from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid

import asyncio
from langchain_core.messages import HumanMessage

def random_uuid():
    return str(uuid.uuid4())

async def run_agent_query(agent, query, recursion_limit, thread_id, timeout_seconds):
    response = await asyncio.wait_for(
        agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=RunnableConfig(
                recursion_limit=recursion_limit,
                thread_id=thread_id,
            ),
        ),
        timeout=timeout_seconds,
    )
    return response

async def run_agent_query_debug(agent, query, recursion_limit, thread_id, timeout_seconds):
    # Step 1: 构造消息
    message = HumanMessage(content=query)
    print("[DEBUG] HumanMessage:", message)

    # Step 2: 构造配置
    config = RunnableConfig(
        recursion_limit=recursion_limit,
        thread_id=thread_id,
    )
    print("[DEBUG] RunnableConfig:", config)

    # Step 3: 构建调用对象
    coro = agent.ainvoke(
        {"messages": [message]},
        config=config
    )
    print("[DEBUG] Awaiting agent.ainvoke...")

    # Step 4: 包裹 timeout，方便定位错误
    try:
        response = await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print("[ERROR] Timeout occurred during agent.ainvoke")
        raise
    except Exception as e:
        print("[ERROR] Exception during agent.ainvoke:", e)
        raise

    # Step 5: 输出响应
    print("[DEBUG] Response:", response)
    return response

async def astream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    stream_mode: str = "messages",
    include_subgraphs: bool = False,
) -> Dict[str, Any]:
    """
    异步流式处理 LangGraph 的执行结果并直接输出的函数。

    Args:
        graph (CompiledStateGraph): 要执行的已编译 LangGraph 对象
        inputs (dict): 传递给图的输入值字典
        config (Optional[RunnableConfig]): 执行配置 (可选)
        node_names (List[str], optional): 要输出的节点名称列表。默认值为空列表
        callback (Optional[Callable], optional): 处理每个数据块的回调函数。默认值为 None
            回调函数接收 {"node": str, "content": Any} 形式的字典作为参数。
        stream_mode (str, optional): 流式处理模式 ("messages" 或 "updates")。默认值为 "messages"
        include_subgraphs (bool, optional): 是否包含子图。默认值为 False

    Returns:
        Dict[str, Any]: 最终结果 (可选)
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    prev_node = ""

    if stream_mode == "messages":
        async for chunk_msg, metadata in graph.astream(
            inputs, config, stream_mode=stream_mode
        ):
            curr_node = metadata["langgraph_node"]
            final_result = {
                "node": curr_node,
                "content": chunk_msg,
                "metadata": metadata,
            }

            # 只有当 node_names 为空或当前节点在 node_names 中时才处理
            if not node_names or curr_node in node_names:
                # 如果有回调函数则执行
                if callback:
                    result = callback({"node": curr_node, "content": chunk_msg})
                    if hasattr(result, "__await__"):
                        await result
                # 没有回调时的默认输出
                else:
                    # 只有当节点改变时才输出分隔线
                    if curr_node != prev_node:
                        print("\n" + "=" * 50)
                        print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                        print("- " * 25)

                    # Claude/Anthropic 模型的 token 数据块处理 - 始终只提取文本
                    if hasattr(chunk_msg, "content"):
                        # 列表形式的 content (Anthropic/Claude 风格)
                        if isinstance(chunk_msg.content, list):
                            for item in chunk_msg.content:
                                if isinstance(item, dict) and "text" in item:
                                    print(item["text"], end="", flush=True)
                        # 字符串形式的 content
                        elif isinstance(chunk_msg.content, str):
                            print(chunk_msg.content, end="", flush=True)
                    # 处理其他形式的 chunk_msg
                    else:
                        print(chunk_msg, end="", flush=True)

                prev_node = curr_node

    elif stream_mode == "updates":
        # 错误修复: 更改解包方式
        # REACT 代理等某些图只返回单个字典
        async for chunk in graph.astream(
            inputs, config, stream_mode=stream_mode, subgraphs=include_subgraphs
        ):
            # 根据返回格式分别处理
            if isinstance(chunk, tuple) and len(chunk) == 2:
                # 预期格式: (namespace, chunk_dict)
                namespace, node_chunks = chunk
            else:
                # 只返回单个字典的情况 (REACT 代理等)
                namespace = []  # 空命名空间 (根图)
                node_chunks = chunk  # chunk 本身就是节点数据块字典

            # 确认是字典并处理条目
            if isinstance(node_chunks, dict):
                for node_name, node_chunk in node_chunks.items():
                    final_result = {
                        "node": node_name,
                        "content": node_chunk,
                        "namespace": namespace,
                    }

                    # 只有当 node_names 不为空时才进行过滤
                    if node_names and node_name not in node_names:
                        continue

                    # 如果有回调函数则执行
                    if callback is not None:
                        result = callback({"node": node_name, "content": node_chunk})
                        # 如果是协程则 await
                        if hasattr(result, "__await__"):
                            await result
                    # 没有回调时的默认输出
                    else:
                        print("\n" + "=" * 50)
                        formatted_namespace = format_namespace(namespace)
                        if formatted_namespace == "root graph":
                            print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                        else:
                            print(
                                f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                            )
                        print("- " * 25)

                        # 输出节点的数据块数据
                        if isinstance(node_chunk, dict):
                            for k, v in node_chunk.items():
                                if isinstance(v, BaseMessage):
                                    v.pretty_print()
                                elif isinstance(v, list):
                                    for list_item in v:
                                        if isinstance(list_item, BaseMessage):
                                            list_item.pretty_print()
                                        else:
                                            print(list_item)
                                elif isinstance(v, dict):
                                    for node_chunk_key, node_chunk_value in v.items():
                                        print(f"{node_chunk_key}:\n{node_chunk_value}")
                                else:
                                    print(f"\033[1;32m{k}\033[0m:\n{v}")
                        elif node_chunk is not None:
                            if hasattr(node_chunk, "__iter__") and not isinstance(
                                node_chunk, str
                            ):
                                for item in node_chunk:
                                    print(item)
                            else:
                                print(node_chunk)
                        print("=" * 50)
            else:
                # 非字典情况，输出整个数据块
                print("\n" + "=" * 50)
                print(f"🔄 Raw output 🔄")
                print("- " * 25)
                print(node_chunks)
                print("=" * 50)
                final_result = {"content": node_chunks}

    # 返回最终结果
    return final_result


async def ainvoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    include_subgraphs: bool = True,
) -> Dict[str, Any]:
    """
    异步流式处理 LangGraph 应用的执行结果并输出的函数。

    Args:
        graph (CompiledStateGraph): 要执行的已编译 LangGraph 对象
        inputs (dict): 传递给图的输入值字典
        config (Optional[RunnableConfig]): 执行配置 (可选)
        node_names (List[str], optional): 要输出的节点名称列表。默认值为空列表
        callback (Optional[Callable], optional): 处理每个数据块的回调函数。默认值为 None
            回调函数接收 {"node": str, "content": Any} 形式的字典作为参数。
        include_subgraphs (bool, optional): 是否包含子图的输出。默认值为 True

    Returns:
        Dict[str, Any]: 最终结果 (最后一个节点的输出)
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # subgraphs 参数用于包含子图的输出
    async for chunk in graph.astream(
        inputs, config, stream_mode="updates", subgraphs=include_subgraphs
    ):
        # 根据返回格式分别处理
        if isinstance(chunk, tuple) and len(chunk) == 2:
            # 预期格式: (namespace, chunk_dict)
            namespace, node_chunks = chunk
        else:
            # 只返回单个字典的情况 (REACT 代理等)
            namespace = []  # 空命名空间 (根图)
            node_chunks = chunk  # chunk 本身就是节点数据块字典

        # 确认是字典并处理条目
        if isinstance(node_chunks, dict):
            for node_name, node_chunk in node_chunks.items():
                final_result = {
                    "node": node_name,
                    "content": node_chunk,
                    "namespace": namespace,
                }

                # 只有当 node_names 不为空时才进行过滤
                if node_names and node_name not in node_names:
                    continue

                # 如果有回调函数则执行
                if callback is not None:
                    result = callback({"node": node_name, "content": node_chunk})
                    # 如果是协程则 await
                    if hasattr(result, "__await__"):
                        await result
                # 没有回调时的默认输出
                else:
                    print("\n" + "=" * 50)
                    formatted_namespace = format_namespace(namespace)
                    if formatted_namespace == "root graph":
                        print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                    else:
                        print(
                            f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                        )
                    print("- " * 25)

                    # 输出节点的数据块数据
                    if isinstance(node_chunk, dict):
                        for k, v in node_chunk.items():
                            if isinstance(v, BaseMessage):
                                v.pretty_print()
                            elif isinstance(v, list):
                                for list_item in v:
                                    if isinstance(list_item, BaseMessage):
                                        list_item.pretty_print()
                                    else:
                                        print(list_item)
                            elif isinstance(v, dict):
                                for node_chunk_key, node_chunk_value in v.items():
                                    print(f"{node_chunk_key}:\n{node_chunk_value}")
                            else:
                                print(f"\033[1;32m{k}\033[0m:\n{v}")
                    elif node_chunk is not None:
                        if hasattr(node_chunk, "__iter__") and not isinstance(
                            node_chunk, str
                        ):
                            for item in node_chunk:
                                print(item)
                        else:
                            print(node_chunk)
                    print("=" * 50)
        else:
            # 非字典情况，输出整个数据块
            print("\n" + "=" * 50)
            print(f"🔄 Raw output 🔄")
            print("- " * 25)
            print(node_chunks)
            print("=" * 50)
            final_result = {"content": node_chunks}

    # 返回最终结果
    return final_result
