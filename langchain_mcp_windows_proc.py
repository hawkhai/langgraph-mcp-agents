# langchain_mcp_windows_proc.py

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
import psutil
import os
import win32gui
import win32process

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # 请替换为您的API密钥

def normalize_name(name: str) -> list:
    """生成包含各种可能变体的进程名列表"""
    name = name.lower()
    candidates = [name]
    if not name.endswith('.exe'):
        candidates.append(name + '.exe')
    return candidates

# 使用langchain的工具装饰器
@tool
def list_processes() -> str:
    """列出当前所有进程和对应的名称"""
    proc_list = [(p.pid, p.name()) for p in psutil.process_iter()]
    return "\n".join([f"PID: {pid}, Name: {name}" for pid, name in proc_list])

@tool
def get_process_name_by_window(name: str) -> str:
    """通过窗口标题模糊匹配并获取对应的进程名。"""
    matches = []
    def enum_windows_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if name.lower() in title.lower():
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    matches.append((title, pid))
                except Exception:
                    pass
    win32gui.EnumWindows(enum_windows_callback, None)
    if not matches:
        return f"未找到包含关键词 '{name}'（及其变体）的窗口标题，无法获取进程名。"
    elif len(matches) == 1:
        title, pid = matches[0]
        try:
            proc = psutil.Process(pid)
            return f"找到进程名：{proc.name()} (pid={pid})"
        except Exception as e:
            return f"尝试获取进程名失败：{str(e)}"
    else:
        return f"找到多个匹配窗口：\n" + '\n'.join(f"{title} (pid={pid})" for title, pid in matches)

@tool
def kill_process(name: str) -> str:
    """
    结束包含指定关键词或变体（如 .exe）的进程。
    如果找到多个匹配项，会返回列表供用户选择。
    """
    candidates = normalize_name(name)
    matches = []

    for p in psutil.process_iter(['pid', 'name']):
        pname = p.info['name'].lower()
        if any(c in pname for c in candidates):
            matches.append(p)

    if not matches:
        return f"未找到包含关键词 '{name}'（及其变体）的进程"
    elif len(matches) == 1:
        p = matches[0]
        try:
            p.kill()
            return f"已成功结束进程：{p.name()} (pid={p.pid})"
        except Exception as e:
            return f"尝试结束进程失败：{str(e)}"
    else:
        return f"找到多个匹配进程：\n" + '\n'.join(f"{p.name()} (pid={p.pid})" for p in matches)

@tool
def kill_process_by_window(name: str) -> str:
    """
    通过窗口标题模糊匹配并结束对应的进程。
    
    输入参数:
    - name: 窗口标题中的关键词（不区分大小写，例如 'notepad'、'浏览器'）

    返回:
    - 成功或失败的结果，包括被结束的窗口及对应 PID。
    """
    matches = []

    def enum_windows_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if name.lower() in title.lower():
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    matches.append((title, pid))
                except Exception:
                    pass

    win32gui.EnumWindows(enum_windows_callback, None)

    if not matches:
        return f"未找到包含“{name}”的窗口标题，无法结束任何进程。"

    killed = []
    failed = []

    for title, pid in matches:
        try:
            proc = psutil.Process(pid)
            proc.kill()
            killed.append(f"已结束：{title} (pid={pid})")
        except Exception as e:
            failed.append(f"失败：{title} (pid={pid})，错误信息：{str(e)}")

    result = ""
    if killed:
        result += "\n".join(killed)
    if failed:
        result += "\n" + "\n".join(failed)
    return result.strip()
    
    
@tool
def get_process_memory(name: str) -> str:
    """获取某个进程的内存使用情况（MB）"""
    for p in psutil.process_iter():
        if p.name().lower() == name.lower():
            mem = p.memory_info().rss / (1024 * 1024)
            return f"进程 {name} 占用内存：{mem:.2f} MB"
    return f"未找到名为 {name} 的进程"

if __name__ == '__main__':
    # 创建LLM
    # 如果使用 OpenAI
    # llm = ChatOpenAI(temperature=0)
    
    # 如果使用通义千问
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain_core.messages import HumanMessage
    llm = ChatTongyi(
        model="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),  # 替换为你的 key
        temperature=0.7,
        top_p=0.8,
        enable_search=False  # 可选参数，仅支持某些模型
    )
    # 测试模型是否工作正常
    # print("测试模型调用...")
    # response = llm.invoke([HumanMessage(content="你好，请介绍一下自己")])
    # print(f"模型回复: {response.content}")
    # print("测试完成\n")
    # exit()

    # 创建工具列表
    tools = [list_processes, kill_process, get_process_memory, kill_process_by_window, get_process_name_by_window]
    
    # 创建记忆
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 使用标准初始化方法创建代理
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 更通用的代理类型
        verbose=True,
        memory=memory
    )

    # 与代理交互
    print("进程管理助手已启动，您可以询问关于进程的问题...")
    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        try:
            response = agent_chain.invoke({"input": query})
            print(response["output"])
        except Exception as e:
            print(f"发生错误: {e}")