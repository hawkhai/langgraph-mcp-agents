#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Agent Tkinter 桌面应用
基于 Streamlit 版本重新实现的桌面版 MCP 工具智能代理
"""

import asyncio
import json
import logging
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import nest_asyncio

# 平台特定的事件循环策略设置（Windows 平台）
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 应用 nest_asyncio: 允许在已运行的事件循环中进行嵌套调用
nest_asyncio.apply()

# 设置编码
if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 导入现有模块
from utils import astream_graph, ainvoke_graph, random_uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from dotenv import load_dotenv

# 加载环境变量（从 .env 文件获取 API 密钥和设置）
load_dotenv(override=True)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 系统提示信息
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

# 模型输出令牌限制信息
OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
    "qwen-plus-latest": {"max_tokens": 16000},
}

# 系统提示模板
SYSTEM_INFO = """<ROLE>
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


class MCPAgentApp:
    """MCP Agent Tkinter 桌面应用主类"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MCP 工具智能代理")
        self.root.geometry("1200x800")
        
        # 应用状态
        self.session_initialized = False
        self.agent = None
        self.mcp_client = None
        self.conversation_history = []
        self.thread_id = random_uuid()  # 使用与 app.py 相同的方式
        self.tool_count = 0  # 初始化工具数量
        
        # 配置变量
        self.selected_model = tk.StringVar(value="qwen-plus-latest")
        self.timeout_seconds = tk.IntVar(value=120)  # 与 app.py 一致
        self.recursion_limit = tk.IntVar(value=100)  # 与 app.py 一致
        self.mcp_config = {}
        
        # 创建 UI
        self.create_widgets()
        self.load_config()
        
        # 创建和重用全局事件循环（创建一次并持续使用）
        self.loop = None
        self.start_async_loop()

    def start_async_loop(self):
        """在单独线程中启动异步事件循环"""
        def run_loop():
            # 创建新的事件循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"事件循环错误: {e}")
            finally:
                self.loop.close()
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        
        # 等待事件循环启动
        import time
        time.sleep(0.1)
    
    def create_widgets(self):
        """创建主界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧设置面板
        self.create_settings_panel(main_frame)
        
        # 创建右侧聊天面板
        self.create_chat_panel(main_frame)
    
    def create_settings_panel(self, parent):
        """创建左侧设置面板"""
        # 设置面板框架
        settings_frame = ttk.LabelFrame(parent, text="⚙️ 系统设置", padding=10)
        settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        settings_frame.configure(width=300)
        
        # 模型选择
        ttk.Label(settings_frame, text="🤖 选择模型:").pack(anchor=tk.W, pady=(0, 5))
        
        # 根据可用的API密钥确定可用模型
        available_models = []
        
        # 检查 Anthropic API 密钥
        has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
        if has_anthropic_key:
            available_models.extend([
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest", 
                "claude-3-5-haiku-latest",
            ])
            
        # 检查 OpenAI API 密钥
        has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
        if has_openai_key:
            available_models.extend(["gpt-4o", "gpt-4o-mini"])
            
        # 检查千问 API 密钥
        has_dashscope_key = os.environ.get("DASHSCOPE_API_KEY") is not None
        if has_dashscope_key:
            available_models.extend(["qwen-plus-latest"])
            
        # 如果没有可用模型，显示警告并添加默认模型
        if not available_models:
            available_models = ["claude-3-7-sonnet-latest"]  # 默认模型用于显示UI
            
        model_combo = ttk.Combobox(settings_frame, textvariable=self.selected_model, 
                                  values=available_models, state="readonly")
        model_combo.pack(fill=tk.X, pady=(0, 5))
        
        # API密钥提示
        api_help = ttk.Label(settings_frame, text="💡 提示: Anthropic 模型需要 ANTHROPIC_API_KEY，\nOpenAI 模型需要 OPENAI_API_KEY，\n千问模型需要 DASHSCOPE_API_KEY", 
                           font=("Arial", 8), foreground="gray")
        api_help.pack(anchor=tk.W, pady=(0, 10))
        
        # 超时设置
        ttk.Label(settings_frame, text="⏱️ 响应生成时间限制（秒）:").pack(anchor=tk.W)
        timeout_frame = ttk.Frame(settings_frame)
        timeout_frame.pack(fill=tk.X, pady=(0, 5))
        
        timeout_scale = tk.Scale(timeout_frame, from_=60, to=300, orient=tk.HORIZONTAL, 
                               variable=self.timeout_seconds, resolution=10)
        timeout_scale.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="💡 设置代理生成响应的最大时间。复杂任务可能需要更多时间。", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W, pady=(0, 10))
        
        # 递归限制设置
        ttk.Label(settings_frame, text="🔄 递归调用限制（次数）:").pack(anchor=tk.W)
        recursion_frame = ttk.Frame(settings_frame)
        recursion_frame.pack(fill=tk.X, pady=(0, 5))
        
        recursion_scale = tk.Scale(recursion_frame, from_=10, to=200, orient=tk.HORIZONTAL, 
                                 variable=self.recursion_limit, resolution=10)
        recursion_scale.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="💡 设置递归调用限制。设置过高的值可能导致内存问题。", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W, pady=(0, 10))
        
        # 工具配置按钮
        ttk.Button(settings_frame, text="🔧 配置工具", 
                  command=self.open_tool_config).pack(fill=tk.X, pady=5)
        
        # 应用设置按钮
        ttk.Button(settings_frame, text="✅ 应用设置", 
                  command=self.apply_settings).pack(fill=tk.X, pady=5)
        
        # 重置对话按钮
        ttk.Button(settings_frame, text="🔄 重置对话", 
                  command=self.reset_conversation).pack(fill=tk.X, pady=5)
        
        # 系统信息
        info_frame = ttk.LabelFrame(settings_frame, text="📊 系统信息", padding=5)
        info_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.status_label = ttk.Label(info_frame, text="状态: 未初始化")
        self.status_label.pack(anchor=tk.W)
        
        self.tool_count_label = ttk.Label(info_frame, text="工具数量: 0")
        self.tool_count_label.pack(anchor=tk.W)
    
    def create_chat_panel(self, parent):
        """创建右侧聊天面板"""
        chat_frame = ttk.LabelFrame(parent, text="💬 对话窗口", padding=10)
        chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 聊天历史显示区域
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=25)
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 用户输入区域
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X)
        
        self.user_input = ttk.Entry(input_frame, font=("Arial", 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind('<Return>', lambda e: self.send_message())
        
        send_button = ttk.Button(input_frame, text="发送", command=self.send_message)
        send_button.pack(side=tk.RIGHT)
    
    def load_config(self):
        """加载配置文件"""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.mcp_config = json.load(f)
                logger.info(f"已加载配置: {len(self.mcp_config)} 个工具")
            else:
                # 使用与 app.py 相同的默认配置
                default_config = {
                    "get_current_time": {
                        "command": "python",
                        "args": ["./mcp_server_time.py"],
                        "transport": "stdio"
                    }
                }
                self.mcp_config = default_config
                self.save_config()  # 保存默认配置
                logger.info("未找到配置文件，使用默认配置")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self.mcp_config = {}
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open("config.json", 'w', encoding='utf-8') as f:
                json.dump(self.mcp_config, f, indent=2, ensure_ascii=False)
            logger.info("配置已保存")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def open_tool_config(self):
        """打开工具配置窗口"""
        ToolConfigWindow(self)
    
    def apply_settings(self):
        """应用设置"""
        try:
            # 检查事件循环是否就绪
            if self.loop is None:
                self.append_to_chat("系统", "❌ 事件循环尚未就绪，请稍后重试。", "error")
                return
                
            # 保存配置
            self.save_config()
            
            # 显示初始化开始消息
            self.append_to_chat("系统", "🔄 正在初始化 MCP 服务器和代理，请稍候...", "system")
            
            # 在后台线程中运行异步初始化
            def init_async():
                try:
                    # 确保事件循环正在运行
                    if not self.loop.is_running():
                        self.root.after(0, lambda: self.append_to_chat("系统", "❌ 事件循环未运行", "error"))
                        return
                        
                    # 使用 run_coroutine_threadsafe 在事件循环中运行协程
                    future = asyncio.run_coroutine_threadsafe(
                        self.initialize_session_async(), self.loop
                    )
                    result = future.result(timeout=60)  # 增加到60秒超时
                    
                    if result:
                        success_msg = f"✅ 初始化成功！已连接 {getattr(self, 'tool_count', 0)} 个工具。现在可以开始对话了。"
                        self.root.after(0, lambda: self.append_to_chat("系统", success_msg, "system"))
                        self.root.after(0, self.update_status)
                    else:
                        self.root.after(0, lambda: self.append_to_chat("系统", "❌ 初始化失败，请检查配置和网络连接。", "error"))
                        
                except asyncio.TimeoutError:
                    error_msg = "❌ 初始化超时，请检查网络连接或工具配置。"
                    self.root.after(0, lambda: self.append_to_chat("系统", error_msg, "error"))
                except Exception as e:
                    error_msg = f"❌ 初始化异常: {str(e)}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    self.root.after(0, lambda: self.append_to_chat("系统", error_msg, "error"))
                    
            threading.Thread(target=init_async, daemon=True).start()
            
        except Exception as e:
            logger.error(f"应用设置错误: {e}\n{traceback.format_exc()}")
            self.append_to_chat("系统", f"❌ 应用设置错误: {str(e)}", "error")
    
    async def initialize_session_async(self):
        """异步初始化会话"""
        try:
            # 清理现有客户端
            await self.cleanup_mcp_client()
            
            # 创建 MCP 客户端
            self.mcp_client = MultiServerMCPClient(self.mcp_config)
            
            # 获取工具
            tools = await self.mcp_client.get_tools()
            self.tool_count = len(tools)  # 记录工具数量
            
            # 创建 Agent
            model_name = self.selected_model.get()
            
            # 根据模型创建不同的 LLM
            if model_name in [
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest", 
                "claude-3-5-haiku-latest",
            ]:
                llm = ChatAnthropic(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                )
            elif model_name in ["qwen-plus-latest"]:
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            else:  # OpenAI models
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                )
            
            # 创建 ReAct Agent
            self.agent = create_react_agent(
                llm,
                tools,
                checkpointer=MemorySaver(),
                prompt=SYSTEM_PROMPT,
            )
            
            # 标记会话已初始化
            self.session_initialized = True
            
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def cleanup_mcp_client(self):
        """安全终止现有的MCP客户端"""
        if self.mcp_client is not None:
            try:
                # 简单设置为None，让垃圾回收处理
                self.mcp_client = None
            except Exception as e:
                logger.error(f"清理MCP客户端时出错: {e}")
    
    def reset_conversation(self):
        """重置对话历史"""
        # 重置线程ID
        self.thread_id = random_uuid()
        
        # 清空聊天历史
        self.chat_history.delete(1.0, tk.END)
        
        # 添加重置消息
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.insert(tk.END, f"[{timestamp}] 系统: ✅ 对话已重置。\n")
        self.chat_history.tag_add("msg_system", "end-2l", "end-1l")
        self.chat_history.tag_config("msg_system", foreground="orange")
        
        logger.info("对话已重置")
    
    def send_message(self):
        """发送用户消息"""
        message = self.user_input.get().strip()
        if not message:
            return
        
        if not self.session_initialized:
            self.append_to_chat("系统", "⚠️ MCP 服务器和代理尚未初始化。请点击'应用设置'按钮进行初始化。", "error")
            return
            
        if self.loop is None or not self.loop.is_running():
            self.append_to_chat("系统", "❌ 事件循环未就绪，请稍后重试。", "error")
            return
        
        # 清空输入框
        self.user_input.delete(0, tk.END)
        
        # 显示用户消息
        self.append_to_chat("用户", message, "user")
        
        # 处理查询
        def process_async():
            try:
                # 使用 run_coroutine_threadsafe 在事件循环中运行协程
                future = asyncio.run_coroutine_threadsafe(
                    self.process_query_async(message), self.loop
                )
                resp, final_text, final_tool = future.result(timeout=self.timeout_seconds.get())
                
                if "error" in resp:
                    # 显示错误消息
                    self.root.after(0, lambda: self.append_to_chat("系统", resp["error"], "error"))
                else:
                    # 确保显示最终完整内容（防止防抖动错过最后的更新）
                    if final_text:
                        self.root.after(0, lambda: self.update_streaming_text(final_text))
                    if final_tool:
                        self.root.after(0, lambda: self.update_tool_info(final_tool))
                    
                    # 清理临时状态
                    def cleanup_streaming_state():
                        if hasattr(self, '_current_assistant_line'):
                            delattr(self, '_current_assistant_line')
                        if hasattr(self, '_current_tool_line'):
                            delattr(self, '_current_tool_line')
                    
                    self.root.after(0, cleanup_streaming_state)
                
            except asyncio.TimeoutError:
                error_msg = f"❌ 查询超时（超过 {self.timeout_seconds.get()} 秒）"
                self.root.after(0, lambda: self.append_to_chat("系统", error_msg, "error"))
                # 清理临时状态
                def cleanup_on_error():
                    if hasattr(self, '_current_assistant_line'):
                        delattr(self, '_current_assistant_line')
                    if hasattr(self, '_current_tool_line'):
                        delattr(self, '_current_tool_line')
                self.root.after(0, cleanup_on_error)
            except Exception as e:
                error_msg = f"❌ 处理异常: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.append_to_chat("系统", error_msg, "error"))
                # 清理临时状态
                def cleanup_on_error():
                    if hasattr(self, '_current_assistant_line'):
                        delattr(self, '_current_assistant_line')
                    if hasattr(self, '_current_tool_line'):
                        delattr(self, '_current_tool_line')
                self.root.after(0, cleanup_on_error)
        
        # 在后台线程中运行处理
        threading.Thread(target=process_async, daemon=True).start()
    
    async def process_query_async(self, query: str):
        """异步处理用户查询，与 app.py 的 process_query 函数逻辑一致"""
        try:
            if self.agent:
                # 记录当前助手消息开始位置，用于流式更新
                self._current_assistant_line = None
                self._current_tool_line = None
                
                # 获取流式回调
                streaming_callback, accumulated_text_obj, accumulated_tool_obj = self.get_streaming_callback()
                
                try:
                    # 使用 asyncio.wait_for 进行超时控制
                    response = await asyncio.wait_for(
                        astream_graph(
                            self.agent,
                            {"messages": [HumanMessage(content=query)]},
                            callback=streaming_callback,
                            config=RunnableConfig(
                                recursion_limit=self.recursion_limit.get(),
                                thread_id=self.thread_id,
                            ),
                        ),
                        timeout=self.timeout_seconds.get(),
                    )
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ 请求时间超过 {self.timeout_seconds.get()} 秒。请稍候再试。"
                    return {"error": error_msg}, error_msg, ""
                
                final_text = "".join(accumulated_text_obj)
                final_tool = "".join(accumulated_tool_obj)
                return response, final_text, final_tool
            else:
                error_msg = "🚫 代理尚未初始化。"
                return {"error": error_msg}, error_msg, ""
        except Exception as e:
            import traceback
            error_msg = f"❌ 发生错误：{str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"error": error_msg}, error_msg, ""
    
    def get_streaming_callback(self):
        """
        创建流式回调函数，用于处理 LLM 生成的流式响应
        
        Returns:
            callback_func: 流式回调函数
            accumulated_text: 累积的文本响应列表
            accumulated_tool: 累积的工具调用信息列表
        """
        accumulated_text = []
        accumulated_tool = []
        
        # 添加防抖动机制
        last_text_update = [0]  # 使用列表以便在闭包中修改
        last_tool_update = [0]
        update_delay = 100  # 毫秒
        
        def callback_func(message: dict):
            nonlocal accumulated_text, accumulated_tool
            message_content = message.get("content", None)
            
            if isinstance(message_content, AIMessageChunk):
                content = message_content.content
                # 如果内容是列表形式（主要出现在 Claude 模型中）
                if isinstance(content, list) and len(content) > 0:
                    message_chunk = content[0]
                    # 处理文本类型
                    if message_chunk["type"] == "text":
                        accumulated_text.append(message_chunk["text"])
                        # 防抖动更新 - 减少频繁更新
                        current_time = int(datetime.now().timestamp() * 1000)
                        if current_time - last_text_update[0] > update_delay:
                            last_text_update[0] = current_time
                            full_text = "".join(accumulated_text)
                            self.root.after(0, lambda text=full_text: self.update_streaming_text(text))
                    # 处理工具使用类型
                    elif message_chunk["type"] == "tool_use":
                        if "partial_json" in message_chunk:
                            accumulated_tool.append(message_chunk["partial_json"])
                        else:
                            tool_call_chunks = message_content.tool_call_chunks
                            if tool_call_chunks:
                                tool_call_chunk = tool_call_chunks[0]
                                accumulated_tool.append(
                                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                                )
                        # 防抖动更新工具信息
                        current_time = int(datetime.now().timestamp() * 1000)
                        if current_time - last_tool_update[0] > update_delay:
                            last_tool_update[0] = current_time
                            full_tool = "".join(accumulated_tool)
                            self.root.after(0, lambda tool=full_tool: self.update_tool_info(tool))
                # 处理如果 tool_calls 属性存在（主要出现在 OpenAI 模型中）
                elif (
                    hasattr(message_content, "tool_calls")
                    and message_content.tool_calls
                    and len(message_content.tool_calls[0]["name"]) > 0
                ):
                    tool_call_info = message_content.tool_calls[0]
                    accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                    current_time = int(datetime.now().timestamp() * 1000)
                    if current_time - last_tool_update[0] > update_delay:
                        last_tool_update[0] = current_time
                        full_tool = "".join(accumulated_tool)
                        self.root.after(0, lambda tool=full_tool: self.update_tool_info(tool))
                # 处理如果内容是简单字符串
                elif isinstance(content, str):
                    accumulated_text.append(content)
                    current_time = int(datetime.now().timestamp() * 1000)
                    if current_time - last_text_update[0] > update_delay:
                        last_text_update[0] = current_time
                        full_text = "".join(accumulated_text)
                        self.root.after(0, lambda text=full_text: self.update_streaming_text(text))
                # 处理如果存在无效的工具调用信息
                elif (
                    hasattr(message_content, "invalid_tool_calls")
                    and message_content.invalid_tool_calls
                ):
                    tool_call_info = message_content.invalid_tool_calls[0]
                    accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                    current_time = int(datetime.now().timestamp() * 1000)
                    if current_time - last_tool_update[0] > update_delay:
                        last_tool_update[0] = current_time
                        full_tool = "".join(accumulated_tool)
                        self.root.after(0, lambda tool=full_tool: self.update_tool_info(tool))
                # 处理如果 tool_call_chunks 属性存在
                elif (
                    hasattr(message_content, "tool_call_chunks")
                    and message_content.tool_call_chunks
                ):
                    tool_call_chunk = message_content.tool_call_chunks[0]
                    accumulated_tool.append(
                        "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                    )
                    current_time = int(datetime.now().timestamp() * 1000)
                    if current_time - last_tool_update[0] > update_delay:
                        last_tool_update[0] = current_time
                        full_tool = "".join(accumulated_tool)
                        self.root.after(0, lambda tool=full_tool: self.update_tool_info(tool))
                # 处理如果 tool_calls 存在于 additional_kwargs 中（支持各种模型兼容性）
                elif (
                    hasattr(message_content, "additional_kwargs")
                    and "tool_calls" in message_content.additional_kwargs
                ):
                    tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                    accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                    current_time = int(datetime.now().timestamp() * 1000)
                    if current_time - last_tool_update[0] > update_delay:
                        last_tool_update[0] = current_time
                        full_tool = "".join(accumulated_tool)
                        self.root.after(0, lambda tool=full_tool: self.update_tool_info(tool))
            # 处理如果是工具消息（工具响应）
            elif hasattr(message_content, '__class__') and 'ToolMessage' in str(message_content.__class__):
                accumulated_tool.append(
                    "\n```json\n" + str(message_content.content) + "\n```\n"
                )
                current_time = int(datetime.now().timestamp() * 1000)
                if current_time - last_tool_update[0] > update_delay:
                    last_tool_update[0] = current_time
                    full_tool = "".join(accumulated_tool)
                    self.root.after(0, lambda tool=full_tool: self.update_tool_info(tool))
        
        return callback_func, accumulated_text, accumulated_tool
    
    def update_streaming_text(self, text):
        """更新流式文本显示"""
        try:
            if not hasattr(self, '_current_assistant_line') or self._current_assistant_line is None:
                # 第一次调用，创建新的助手消息行
                timestamp = datetime.now().strftime("%H:%M:%S")
                self._current_assistant_line = self.chat_history.index(tk.END + "-1l")
                self.chat_history.insert(tk.END, f"[{timestamp}] 助手: {text}\n")
                self.chat_history.tag_add("msg_assistant", self._current_assistant_line, tk.END + "-1l")
                self.chat_history.tag_config("msg_assistant", foreground="blue")
            else:
                # 更新现有的助手消息行
                try:
                    # 找到当前行的内容，保留时间戳，只更新消息内容
                    current_line = self.chat_history.get(self._current_assistant_line, self._current_assistant_line + "+1l")
                    if "] 助手: " in current_line:
                        timestamp_part = current_line.split("] 助手: ")[0] + "] 助手: "
                    else:
                        timestamp_part = f"[{datetime.now().strftime('%H:%M:%S')}] 助手: "
                    
                    # 删除当前行并插入新内容
                    self.chat_history.delete(self._current_assistant_line, self._current_assistant_line + "+1l")
                    self.chat_history.insert(self._current_assistant_line, f"{timestamp_part}{text}\n")
                    self.chat_history.tag_add("msg_assistant", self._current_assistant_line, self._current_assistant_line + "+1l")
                    self.chat_history.tag_config("msg_assistant", foreground="blue")
                except tk.TclError:
                    # 如果行索引无效，重新创建
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self._current_assistant_line = self.chat_history.index(tk.END + "-1l")
                    self.chat_history.insert(tk.END, f"[{timestamp}] 助手: {text}\n")
                    self.chat_history.tag_add("msg_assistant", self._current_assistant_line, tk.END + "-1l")
                    self.chat_history.tag_config("msg_assistant", foreground="blue")
            
            self.chat_history.see(tk.END)
        except Exception as e:
            logger.error(f"更新流式文本失败: {e}")

    def update_tool_info(self, tool_info):
        """更新工具调用信息显示"""
        if tool_info.strip():
            try:
                if not hasattr(self, '_current_tool_line') or self._current_tool_line is None:
                    # 第一次调用，创建新的工具消息行
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self._current_tool_line = self.chat_history.index(tk.END + "-1l")
                    self.chat_history.insert(tk.END, f"[{timestamp}] 工具: {tool_info}\n")
                    self.chat_history.tag_add("msg_tool", self._current_tool_line, tk.END + "-1l")
                    self.chat_history.tag_config("msg_tool", foreground="green")
                else:
                    # 更新现有的工具消息行
                    try:
                        current_line = self.chat_history.get(self._current_tool_line, self._current_tool_line + "+1l")
                        if "] 工具: " in current_line:
                            timestamp_part = current_line.split("] 工具: ")[0] + "] 工具: "
                        else:
                            timestamp_part = f"[{datetime.now().strftime('%H:%M:%S')}] 工具: "
                        
                        # 删除当前行并插入新内容
                        self.chat_history.delete(self._current_tool_line, self._current_tool_line + "+1l")
                        self.chat_history.insert(self._current_tool_line, f"{timestamp_part}{tool_info}\n")
                        self.chat_history.tag_add("msg_tool", self._current_tool_line, self._current_tool_line + "+1l")
                        self.chat_history.tag_config("msg_tool", foreground="green")
                    except tk.TclError:
                        # 如果行索引无效，重新创建
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self._current_tool_line = self.chat_history.index(tk.END + "-1l")
                        self.chat_history.insert(tk.END, f"[{timestamp}] 工具: {tool_info}\n")
                        self.chat_history.tag_add("msg_tool", self._current_tool_line, tk.END + "-1l")
                        self.chat_history.tag_config("msg_tool", foreground="green")
                
                self.chat_history.see(tk.END)
            except Exception as e:
                logger.error(f"更新工具信息失败: {e}")
    
    def append_to_chat(self, sender: str, message: str, msg_type: str = "normal"):
        """向聊天窗口添加消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 根据消息类型设置颜色
        if msg_type == "user":
            color = "blue"
        elif msg_type == "assistant":
            color = "green"
        elif msg_type == "system":
            color = "orange"
        elif msg_type == "error":
            color = "red"
        elif msg_type == "success":
            color = "darkgreen"
        else:
            color = "black"
        
        # 插入消息
        self.chat_history.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        self.chat_history.see(tk.END)
        
        # 应用颜色标签
        start_line = float(self.chat_history.index(tk.END)) - 2
        self.chat_history.tag_add(f"msg_{msg_type}", f"{start_line:.1f}", f"{start_line + 1:.1f}")
        self.chat_history.tag_config(f"msg_{msg_type}", foreground=color)
    
    def update_status(self):
        """更新状态信息"""
        if self.session_initialized:
            tool_count = getattr(self, 'tool_count', len(self.mcp_config) if self.mcp_config else 0)
            status = f"状态: ✅ 已连接 | 🛠️ 工具数量: {tool_count} | 🧠 模型: {self.selected_model.get()}"
        else:
            status = "状态: ❌ 未初始化 - 请点击'应用设置'按钮进行初始化"
        
        self.status_label.config(text=status)
    
    def run(self):
        """运行应用"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """应用关闭处理"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()


class ToolConfigWindow:
    """工具配置窗口"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.window = tk.Toplevel(parent_app.root)
        self.window.title("🔧 工具配置")
        self.window.geometry("800x600")
        self.window.transient(parent_app.root)
        self.window.grab_set()
        
        self.create_widgets()
        self.load_current_config()
    
    def create_widgets(self):
        """创建配置窗口组件"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 工具列表
        list_frame = ttk.LabelFrame(main_frame, text="已配置工具", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建树形视图显示工具
        self.tool_tree = ttk.Treeview(list_frame, columns=("transport", "command"), show="tree headings")
        self.tool_tree.heading("#0", text="工具名称")
        self.tool_tree.heading("transport", text="传输方式")
        self.tool_tree.heading("command", text="命令")
        self.tool_tree.pack(fill=tk.BOTH, expand=True)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="➕ 添加工具", command=self.add_tool).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="❌ 删除工具", command=self.delete_tool).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="📄 导入配置", command=self.import_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="💾 保存", command=self.save_config).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="❌ 取消", command=self.window.destroy).pack(side=tk.RIGHT)
    
    def load_current_config(self):
        """加载当前配置到界面"""
        for item in self.tool_tree.get_children():
            self.tool_tree.delete(item)
        
        for tool_name, config in self.parent_app.mcp_config.items():
            transport = config.get("transport", "stdio")
            command = config.get("command", config.get("url", ""))
            self.tool_tree.insert("", tk.END, text=tool_name, values=(transport, command))
    
    def add_tool(self):
        """添加新工具"""
        AddToolDialog(self)
    
    def delete_tool(self):
        """删除选中的工具"""
        selection = self.tool_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的工具")
            return
        
        item = selection[0]
        tool_name = self.tool_tree.item(item, "text")
        
        if messagebox.askyesno("确认", f"确定要删除工具 '{tool_name}' 吗？"):
            if tool_name in self.parent_app.mcp_config:
                del self.parent_app.mcp_config[tool_name]
            self.tool_tree.delete(item)
    
    def import_config(self):
        """导入配置文件"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 合并配置
                self.parent_app.mcp_config.update(config)
                self.load_current_config()
                messagebox.showinfo("成功", "配置导入成功")
            except Exception as e:
                messagebox.showerror("错误", f"导入失败: {str(e)}")
    
    def save_config(self):
        """保存配置"""
        if self.parent_app.save_config():
            messagebox.showinfo("成功", "配置保存成功")
            self.window.destroy()
        else:
            messagebox.showerror("错误", "保存失败")


class AddToolDialog:
    """添加工具对话框"""
    
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.dialog = tk.Toplevel(parent_window.window)
        self.dialog.title("添加工具")
        self.dialog.geometry("600x400")
        self.dialog.transient(parent_window.window)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建对话框组件"""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 工具名称
        ttk.Label(main_frame, text="工具名称:").pack(anchor=tk.W)
        self.name_entry = ttk.Entry(main_frame)
        self.name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # JSON 配置
        ttk.Label(main_frame, text="JSON 配置:").pack(anchor=tk.W)
        self.json_text = scrolledtext.ScrolledText(main_frame, height=15)
        self.json_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 示例配置
        example = {
            "command": "python",
            "args": ["script.py"],
            "transport": "stdio"
        }
        self.json_text.insert(tk.END, json.dumps(example, indent=2, ensure_ascii=False))
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="✅ 添加", command=self.add_tool).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="❌ 取消", command=self.dialog.destroy).pack(side=tk.RIGHT)
    
    def add_tool(self):
        """添加工具到配置"""
        tool_name = self.name_entry.get().strip()
        if not tool_name:
            messagebox.showwarning("警告", "请输入工具名称")
            return
        
        try:
            config_text = self.json_text.get(1.0, tk.END).strip()
            config = json.loads(config_text)
            
            # 验证配置
            if "command" not in config and "url" not in config:
                messagebox.showerror("错误", "配置必须包含 'command' 或 'url' 字段")
                return
            
            # 添加到父应用配置
            self.parent_window.parent_app.mcp_config[tool_name] = config
            self.parent_window.load_current_config()
            
            messagebox.showinfo("成功", f"工具 '{tool_name}' 添加成功")
            self.dialog.destroy()
            
        except json.JSONDecodeError as e:
            messagebox.showerror("错误", f"JSON 格式错误: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"添加失败: {str(e)}")


def main():
    """主函数"""
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    # 创建并运行应用
    app = MCPAgentApp()
    app.run()


if __name__ == "__main__":
    main()
