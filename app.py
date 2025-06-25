import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Load environment variables (get API keys and settings from .env file)
load_dotenv(override=True)

# config.json file path setting
CONFIG_FILE_PATH = "config.json"

# Function to load settings from JSON file
def load_config_from_json():
    """
    Loads settings from config.json file.
    Creates a file with default settings if it doesn't exist.

    Returns:
        dict: Loaded settings
    """
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["./mcp_server_time.py"],
            "transport": "stdio"
        }
    }

    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Create file with default settings if it doesn't exist
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"Error loading settings file: {str(e)}")
        return default_config

# Function to save settings to JSON file
def save_config_to_json(config):
    """
    Saves settings to config.json file.

    Args:
        config (dict): Settings to save

    Returns:
        bool: Save success status
    """
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving settings file: {str(e)}")
        return False

# Initialize login session variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Main app uses wide layout
st.set_page_config(page_title="MCP 工具智能代理", page_icon="🧠", layout="wide")

st.sidebar.divider()  # Add divider

# Existing page title and description
st.title("💬 MCP 工具智能代理")
st.markdown("✨ 请输入问题，智能代理将使用 MCP 工具来回答。")

SYSTEM_PROMPT = """<ROLE>
你是一位智能代理，能够使用工具来回答问题。
你将被给予一个问题，并使用工具来回答。
选择最相关的工具来回答问题。
如果你无法回答问题，请尝试使用不同的工具来获取上下文。
你的答案应该非常礼貌和专业。
</ROLE>

----

<INSTRUCTIONS>
步骤 1：分析问题
- 分析用户的问题和最终目标。
- 如果用户的问题包含多个子问题，请将它们分解为较小的子问题。

步骤 2：选择最相关的工具
- 选择最相关的工具来回答问题。
- 如果你无法回答问题，请尝试使用不同的工具来获取上下文。

步骤 3：回答问题
- 用相同的语言回答问题。
- 你的答案应该非常礼貌和专业。

步骤 4：提供答案来源（如果适用）
- 如果你使用了工具，请提供答案来源。
- 有效来源是网站（URL）或文档（PDF 等）。

指南：
- 如果你使用了工具，你的答案应该基于工具的输出（工具的输出比你自己的知识更重要）。
- 如果你使用了工具，并且来源是有效的 URL，请提供答案来源（URL）。
- 如果来源不是 URL，请跳过提供来源。
- 用相同的语言回答问题。
- 答案应该简洁明了。
- 避免在输出中包含除答案和来源以外的任何信息。
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(简洁的答案)

**来源**（如果适用）
- (来源 1：有效 URL)
- (来源 2：有效 URL)
- ...
</OUTPUT_FORMAT>
"""

OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
    "qwen-plus-latest": {"max_tokens": 16000},
}

# Initialize session state
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # Session initialization flag
    st.session_state.agent = None  # Storage for ReAct agent object
    st.session_state.history = []  # List for storing conversation history
    st.session_state.mcp_client = None  # Storage for MCP client object
    st.session_state.timeout_seconds = (
        120  # Response generation time limit (seconds), default 120 seconds
    )
    st.session_state.selected_model = (
        "qwen-plus-latest"  # Default model selection
    )
    st.session_state.recursion_limit = 100  # Recursion call limit, default 100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- Function Definitions ---


async def cleanup_mcp_client():
    """
    Safely terminates the existing MCP client.

    Properly releases resources if an existing client exists.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            # Simply set to None as we're not using context manager anymore
            # The client will be garbage collected
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback

            # st.warning(f"Error while terminating MCP client: {str(e)}")
            # st.warning(traceback.format_exc())


def print_message():
    """
    Displays chat history on the screen.

    Distinguishes between user and assistant messages on the screen,
    and displays tool call information within the assistant message container.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # Create assistant message container
            with st.chat_message("assistant", avatar="🤖"):
                # Display assistant message content
                st.markdown(message["content"])

                # Check if the next message is tool call information
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # Display tool call information in the same container as an expander
                    with st.expander("🔧 工具调用信息", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # Increment by 2 as we processed two messages together
                else:
                    i += 1  # Increment by 1 as we only processed a regular message
        else:
            # Skip assistant_tool messages as they are handled above
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    Creates a streaming callback function.

    This function creates a callback function to display responses generated from the LLM in real-time.
    It displays text responses and tool call information in separate areas.

    Args:
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information

    Returns:
        callback_func: Streaming callback function
        accumulated_text: List to store accumulated text responses
        accumulated_tool: List to store accumulated tool call information
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            # If content is in list form (mainly occurs in Claude models)
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # Process text type
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                # Process tool use type
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander(
                        "🔧 工具调用信息", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
            # Process if tool_calls attribute exists (mainly occurs in OpenAI models)
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 工具调用信息", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if content is a simple string
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # Process if invalid tool call information exists
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 工具调用信息（无效）", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if tool_call_chunks attribute exists
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander(
                    "🔧 工具调用信息", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if tool_calls exists in additional_kwargs (supports various model compatibility)
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 工具调用信息", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
        # Process if it's a tool message (tool response)
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    Processes user questions and generates responses.

    This function passes the user's question to the agent and streams the response in real-time.
    Returns a timeout error if the response is not completed within the specified time.

    Args:
        query: Text of the question entered by the user
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information
        timeout_seconds: Response generation time limit (seconds)

    Returns:
        response: Agent's response object
        final_text: Final text response
        final_tool: Final tool call information
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 请求时间超过 {timeout_seconds} 秒。请稍候再试。"
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 代理尚未初始化。"},
                "🚫 代理尚未初始化。",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ 发生错误：{str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    Initializes MCP session and agent.

    Args:
        mcp_config: MCP tool configuration information (JSON). Uses default settings if None

    Returns:
        bool: Initialization success status
    """
    with st.spinner("🔄 正在连接到 MCP 服务器..."):
        # First safely clean up existing client
        await cleanup_mcp_client()

        if mcp_config is None:
            # Load settings from config.json file
            mcp_config = load_config_from_json()
        client = MultiServerMCPClient(mcp_config)
        # Use the recommended approach instead of context manager
        tools = await client.get_tools()
        st.session_state.tool_count = len(tools)
        st.session_state.mcp_client = client

        # Initialize appropriate model based on selection
        selected_model = st.session_state.selected_model

        if selected_model in [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]:
            model = ChatAnthropic(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        elif selected_model in [
            "qwen-plus-latest",
        ]:
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
                openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问兼容 OpenAI 的 URL
            )
        else:  # Use OpenAI model
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True


# --- Sidebar: System Settings Section ---
with st.sidebar:
    st.subheader("⚙️ 系统设置")

    # Model selection feature
    # Create list of available models
    available_models = []

    # Check Anthropic API key
    has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
    if has_anthropic_key:
        available_models.extend(
            [
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ]
        )

    # Check OpenAI API key
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    if has_openai_key:
        available_models.extend(["gpt-4o", "gpt-4o-mini"])

    # "qwen-plus-latest"
    has_openai_key = os.environ.get("DASHSCOPE_API_KEY") is not None
    if has_openai_key:
        available_models.extend(["qwen-plus-latest",])

    # Display message if no models are available
    if not available_models:
        st.warning(
            "⚠️ 未配置 API 密钥。请在 .env 文件中添加 ANTHROPIC_API_KEY 或 OPENAI_API_KEY。"
        )
        # Add Claude model as default (to show UI even without keys)
        available_models = ["claude-3-7-sonnet-latest"]

    # Model selection dropdown
    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 选择要使用的模型",
        options=available_models,
        index=(
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        ),
        help="Anthropic 模型需要设置 ANTHROPIC_API_KEY，OpenAI 模型需要设置 OPENAI_API_KEY 环境变量。",
    )

    # Notify when model is changed and session needs to be reinitialized
    if (
        previous_model != st.session_state.selected_model
        and st.session_state.session_initialized
    ):
        st.warning(
            "⚠️ 模型已更改。点击'应用设置'按钮以应用更改。"
        )

    # Add timeout setting slider
    st.session_state.timeout_seconds = st.slider(
        "⏱️ 响应生成时间限制（秒）",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="设置代理生成响应的最大时间。复杂任务可能需要更多时间。",
    )

    st.session_state.recursion_limit = st.slider(
        "⏱️ 递归调用限制（次数）",
        min_value=10,
        max_value=200,
        value=st.session_state.recursion_limit,
        step=10,
        help="设置递归调用限制。设置过高的值可能导致内存问题。",
    )

    st.divider()  # Add divider

    # Tool settings section
    st.subheader("🔧 工具设置")

    # Manage expander state in session state
    if "mcp_tools_expander" not in st.session_state:
        st.session_state.mcp_tools_expander = False

    # MCP tool addition interface
    with st.expander("🧰 添加 MCP 工具", expanded=st.session_state.mcp_tools_expander):
        # Load settings from config.json file
        loaded_config = load_config_from_json()
        default_config_text = json.dumps(loaded_config, indent=2, ensure_ascii=False)

        # Create pending config based on existing mcp_config_text if not present
        if "pending_mcp_config" not in st.session_state:
            try:
                st.session_state.pending_mcp_config = loaded_config
            except Exception as e:
                st.error(f"Failed to set initial pending config: {e}")

        # UI for adding individual tools
        st.subheader("添加工具（JSON 格式）")
        st.markdown(
            """
        请插入**一个工具**的 JSON 格式配置。

        ⚠️ **重要**：JSON 必须用大括号（`{}`）包围。
        """
        )

        # Provide clearer example
        example_json = {
            "github": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@smithery-ai/github",
                    "--config",
                    '{"githubPersonalAccessToken":"your_token_here"}',
                ],
                "transport": "stdio",
            }
        }

        default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

        new_tool_json = st.text_area(
            "工具 JSON",
            default_text,
            height=250,
        )

        # Add button
        if st.button(
            "添加工具",
            type="primary",
            key="add_tool_button",
            use_container_width=True,
        ):
            try:
                # Validate input
                if not new_tool_json.strip().startswith(
                    "{"
                ) or not new_tool_json.strip().endswith("}"):
                    st.error("JSON 必须以大括号（{}）开始和结束。")
                    st.markdown('正确格式：`{ "工具名称": { ... } }`')
                else:
                    # Parse JSON
                    parsed_tool = json.loads(new_tool_json)

                    # Check if it's in mcpServers format and process accordingly
                    if "mcpServers" in parsed_tool:
                        # Move contents of mcpServers to top level
                        parsed_tool = parsed_tool["mcpServers"]
                        st.info(
                            "检测到 'mcpServers' 格式。正在自动转换。"
                        )

                    # Check number of tools entered
                    if len(parsed_tool) == 0:
                        st.error("请至少输入一个工具。")
                    else:
                        # Process all tools
                        success_tools = []
                        for tool_name, tool_config in parsed_tool.items():
                            # Check URL field and set transport
                            if "url" in tool_config:
                                # Set transport to "sse" if URL exists
                                tool_config["transport"] = "sse"
                                st.info(
                                    f"在 '{tool_name}' 工具中检测到 URL，将传输方式设置为 'sse'。"
                                )
                            elif "transport" not in tool_config:
                                # Set default "stdio" if URL doesn't exist and transport isn't specified
                                tool_config["transport"] = "stdio"

                            # Check required fields
                            if (
                                "command" not in tool_config
                                and "url" not in tool_config
                            ):
                                st.error(
                                    f"'{tool_name}' 工具配置需要 'command' 或 'url' 字段。"
                                )
                            elif "command" in tool_config and "args" not in tool_config:
                                st.error(
                                    f"'{tool_name}' 工具配置需要 'args' 字段。"
                                )
                            elif "command" in tool_config and not isinstance(
                                tool_config["args"], list
                            ):
                                st.error(
                                    f"'{tool_name}' 工具中的 'args' 字段必须是数组（[]）格式。"
                                )
                            else:
                                # Add tool to pending_mcp_config
                                st.session_state.pending_mcp_config[tool_name] = (
                                    tool_config
                                )
                                success_tools.append(tool_name)

                        # Success message
                        if success_tools:
                            if len(success_tools) == 1:
                                st.success(
                                    f"{success_tools[0]} 工具已添加。点击'应用设置'按钮以应用。"
                                )
                            else:
                                tool_names = ", ".join(success_tools)
                                st.success(
                                    f"总共 {len(success_tools)} 个工具（{tool_names}）已添加。点击'应用设置'按钮以应用。"
                                )
                            # Collapse expander after adding
                            st.session_state.mcp_tools_expander = False
                            st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON 解析错误：{e}")
                st.markdown(
                    f"""
                **修复方法**：
                1. 检查 JSON 格式是否正确。
                2. 所有键必须用双引号（"）包围。
                3. 字符串值也必须用双引号（"）包围。
                4. 在字符串中使用双引号时，必须转义（\\\"）。
                """
                )
            except Exception as e:
                st.error(f"发生错误：{e}")

    # Display registered tools list and add delete buttons
    with st.expander("📋 已注册工具列表", expanded=True):
        try:
            pending_config = st.session_state.pending_mcp_config
        except Exception as e:
            st.error("不是有效的 MCP 工具配置。")
        else:
            # Iterate through keys (tool names) in pending config
            for tool_name in list(pending_config.keys()):
                col1, col2 = st.columns([8, 2])
                col1.markdown(f"- **{tool_name}**")
                if col2.button("删除", key=f"delete_{tool_name}"):
                    # Delete tool from pending config (not applied immediately)
                    del st.session_state.pending_mcp_config[tool_name]
                    st.success(
                        f"{tool_name} 工具已删除。点击'应用设置'按钮以应用。"
                    )

    st.divider()  # Add divider

# --- Sidebar: System Information and Action Buttons Section ---
with st.sidebar:
    st.subheader("📊 系统信息")
    st.write(
        f"🛠️ MCP 工具数量：{st.session_state.get('tool_count', '初始化中...')}"
    )
    selected_model_name = st.session_state.selected_model
    st.write(f"🧠 当前模型：{selected_model_name}")

    # Move Apply Settings button here
    if st.button(
        "应用设置",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # Display applying message
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 正在应用更改。请稍候...")
            progress_bar = st.progress(0)

            # Save settings
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )

            # Save settings to config.json file
            save_result = save_config_to_json(st.session_state.pending_mcp_config)
            if not save_result:
                st.error("❌ 保存设置文件失败。")

            progress_bar.progress(15)

            # Prepare session initialization
            st.session_state.session_initialized = False
            st.session_state.agent = None

            # Update progress
            progress_bar.progress(30)

            # Run initialization
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # Update progress
            progress_bar.progress(100)

            if success:
                st.success("✅ 新设置已应用。")
                # Collapse tool addition expander
                if "mcp_tools_expander" in st.session_state:
                    st.session_state.mcp_tools_expander = False
            else:
                st.error("❌ 应用设置失败。")

        # Refresh page
        st.rerun()

    st.divider()  # Add divider

    # Action buttons section
    st.subheader("🔄 操作")

    # Reset conversation button
    if st.button("重置对话", use_container_width=True, type="primary"):
        # Reset thread_id
        st.session_state.thread_id = random_uuid()

        # Reset conversation history
        st.session_state.history = []

        # Notification message
        st.success("✅ 对话已重置。")

        # Refresh page
        st.rerun()

# --- Initialize default session (if not initialized) ---
if not st.session_state.session_initialized:
    st.info(
        "MCP 服务器和代理尚未初始化。请点击左侧边栏中的'应用设置'按钮进行初始化。"
    )


# --- Print conversation history ---
print_message()

# --- User input and processing ---
user_query = st.chat_input("💬 输入您的问题")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning(
            "⚠️ MCP 服务器和代理尚未初始化。请点击左侧边栏中的'应用设置'按钮进行初始化。"
        )
