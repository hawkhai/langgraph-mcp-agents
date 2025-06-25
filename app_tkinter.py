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
import time

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

# 导入模型日志记录模块
from model_logger import get_model_logger, ModelCallTracker, init_model_logging

# 加载环境变量（从 .env 文件获取 API 密钥和设置）
load_dotenv(override=True)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from my_constants import *
from my_dialogs import ToolConfigWindow, AddToolDialog

class MCPAgentApp:
    """MCP Agent Tkinter 桌面应用主类"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MCP 工具智能代理")
        self.root.geometry("1200x800")
        
        # 居中窗口显示（处理屏幕边界）
        self.center_window(self.root)
        
        # 初始化模型日志记录
        self.model_logger = init_model_logging("logs")
        self.model_tracker = ModelCallTracker("logs")
        
        # 生成会话ID
        import uuid
        self.session_id = str(uuid.uuid4())[:8]
        
        # 应用状态
        self.session_initialized = False
        self.agent = None
        self.mcp_client = None
        self.conversation_history = []
        self.thread_id = random_uuid()  # 使用与 app.py 相同的方式
        self.tool_count = 0  # 初始化工具数量
        self.current_model_name = None  # 保存当前使用的模型名称
        self.current_model_provider = None  # 保存当前模型提供商
        
        # 配置变量
        self.selected_model = tk.StringVar(value="qwen-plus-latest")
        self.timeout_seconds = tk.IntVar(value=120)  # 与 app.py 一致
        self.recursion_limit = tk.IntVar(value=100)  # 与 app.py 一致
        self.streaming_enabled = tk.BooleanVar(value=False)  # 默认使用普通返回
        self.mcp_config = {}
        
        # 聊天历史存储
        self.chat_messages = []  # 存储结构化的聊天消息
        
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
        
        # 流式返回设置
        streaming_frame = ttk.Frame(settings_frame)
        streaming_frame.pack(fill=tk.X, pady=(0, 5))
        
        streaming_checkbox = ttk.Checkbutton(streaming_frame, text="流式返回", 
                                            variable=self.streaming_enabled)
        streaming_checkbox.pack(side=tk.LEFT)
        
        ttk.Label(streaming_frame, text="💡 启用流式返回以实时查看代理的思考过程。", 
                 font=("Arial", 8), foreground="gray").pack(side=tk.LEFT, padx=(5, 0))
        
        # 工具配置按钮
        ttk.Button(settings_frame, text="🔧 配置工具", 
                  command=self.open_tool_config).pack(fill=tk.X, pady=5)
        
        # 应用设置按钮
        ttk.Button(settings_frame, text="✅ 应用设置", 
                  command=self.apply_settings).pack(fill=tk.X, pady=5)
        
        # 重置对话按钮
        ttk.Button(settings_frame, text="🔄 重置对话", 
                  command=self.reset_conversation).pack(fill=tk.X, pady=5)
        
        # 日志统计按钮
        ttk.Button(settings_frame, text="📊 查看调用日志", 
                  command=self.show_log_stats).pack(fill=tk.X, pady=5)
        
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
            # 获取当前脚本所在目录，确保无论从哪里运行都能找到配置文件
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.json"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.mcp_config = json.load(f)
                logger.info(f"已加载配置: {len(self.mcp_config)} 个工具")
            else:
                # 使用与 app.py 相同的默认配置
                default_config = {
                    "get_current_time": {
                        "command": "python",
                        "args": [str(script_dir / "mcp_server_time.py")],  # 也使用绝对路径
                        "transport": "stdio"
                    }
                }
                self.mcp_config = default_config
                self.save_config()  # 保存默认配置
                logger.info(f"未找到配置文件 {config_path}，使用默认配置")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self.mcp_config = {}
    
    def save_config(self):
        """保存配置文件"""
        try:
            # 获取当前脚本所在目录，确保无论从哪里运行都能找到配置文件
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.json"
            
            with open(config_path, 'w', encoding='utf-8') as f:
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
            logger.info("开始初始化 MCP 会话...")
            logger.info(f"当前工作目录: {os.getcwd()}")
            logger.info(f"MCP 配置: {self.mcp_config}")
            
            # 清理现有客户端
            await self.cleanup_mcp_client()
            logger.info("已清理现有 MCP 客户端")
            
            # 创建 MCP 客户端
            logger.info("正在创建 MCP 客户端...")
            self.mcp_client = MultiServerMCPClient(self.mcp_config)
            logger.info("MCP 客户端创建成功")
            
            # 获取工具
            logger.info("正在获取工具...")
            tools = await self.mcp_client.get_tools()
            self.tool_count = len(tools)  # 记录工具数量
            logger.info(f"成功获取 {self.tool_count} 个工具")
            
            # 创建 Agent
            model_name = self.selected_model.get()
            logger.info(f"正在为模型 {model_name} 创建 Agent...")
            
            # 保存当前模型信息
            self.current_model_name = model_name
            if model_name in ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]:
                self.current_model_provider = "anthropic"
            elif model_name in ["qwen-plus-latest"]:
                self.current_model_provider = "alibaba"
            else:
                self.current_model_provider = "openai"
            
            logger.info(f"模型提供商: {self.current_model_provider}")
            
            # 根据模型创建不同的 LLM
            if model_name in [
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest", 
                "claude-3-5-haiku-latest",
            ]:
                logger.info("创建 Anthropic LLM...")
                llm = ChatAnthropic(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                )
            elif model_name in ["qwen-plus-latest"]:
                logger.info("创建 Alibaba LLM...")
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    logger.error("DASHSCOPE_API_KEY 环境变量未设置")
                    return False
                logger.info(f"DASHSCOPE_API_KEY 已设置: {'是' if api_key else '否'}")
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                    openai_api_key=api_key,
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            else:  # OpenAI models
                logger.info("创建 OpenAI LLM...")
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                )
            
            logger.info("LLM 创建成功，正在创建 ReAct Agent...")
            # 创建 ReAct Agent
            self.agent = create_react_agent(
                llm,
                tools,
                checkpointer=MemorySaver(),
                prompt=SYSTEM_PROMPT,
            )
            logger.info("ReAct Agent 创建成功")
            
            # 标记会话已初始化
            self.session_initialized = True
            logger.info("MCP 会话初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
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
        self.chat_messages = []
        
        # 添加重置消息
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append_to_chat("系统", "✅ 对话已重置。", "system")
        
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
        
        # 根据流式设置选择处理方式
        if self.streaming_enabled.get():
            # 流式处理模式
            self._send_message_streaming(message)
        else:
            # 普通处理模式
            self._send_message_normal(message)
    
    def _send_message_streaming(self, message: str):
        """流式处理消息"""
        # 清理之前的流式状态
        if hasattr(self, '_current_tool_message_start'):
            delattr(self, '_current_tool_message_start')
        
        logger.info(f"开始处理流式模式消息: '{message}'")
        
        # 显示思考占位符
        self.append_to_chat("助手", "🤔 正在思考...", "assistant")
        
        # 处理查询
        def process_async():
            try:
                # 使用 run_coroutine_threadsafe 在事件循环中运行协程
                logger.info("发起流式协程调用")
                future = asyncio.run_coroutine_threadsafe(
                    self.process_query_async(message), self.loop
                )
                resp, final_text, final_tool = future.result(timeout=self.timeout_seconds.get())
                logger.info(f"流式报返结果: 文本长度={len(final_text) if final_text else 0}, 工具信息长度={len(final_tool) if final_tool else 0}")
                
                if isinstance(resp, dict) and "error" in resp:
                    # 替换思考占位符为错误消息
                    error_msg = resp["error"]
                    logger.info(f"处理错误响应: {error_msg}")
                    self.root.after(0, lambda: self.replace_last_assistant_message(error_msg))
                else:
                    # 替换思考占位符为最终内容
                    if final_text:
                        logger.info(f"更新最终助手消息: 长度={len(final_text)}")
                        self.root.after(0, lambda: self.replace_last_assistant_message(final_text))
                    else:
                        fallback_text = "收到回复但无法解析内容"
                        logger.warning("流式响应没有最终文本，使用回退消息")
                        self.root.after(0, lambda: self.replace_last_assistant_message(fallback_text))
                    
                    # 显示工具信息（如果有）
                    if final_tool:
                        logger.info(f"使用最终工具信息更新: 长度={len(final_tool)}")
                        # 使用update_tool_info代替append_to_chat来避免重复
                        self.root.after(0, lambda tool=final_tool: self.update_tool_info(tool))
                
            except asyncio.TimeoutError:
                error_msg = f"❌ 查询超时（超过 {self.timeout_seconds.get()} 秒）"
                logger.error(error_msg)
                self.root.after(0, lambda: self.replace_last_assistant_message(error_msg))
            except Exception as e:
                error_msg = f"❌ 处理异常: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.replace_last_assistant_message(error_msg))
        
        # 在后台线程中运行处理
        threading.Thread(target=process_async, daemon=True).start()
        
    def _send_message_normal(self, message: str):
        """普通处理消息（不使用流式）"""
        # 处理查询
        def process_async():
            try:
                logger.info(f"开始处理普通模式消息: '{message}'")
                # 使用普通查询处理方法
                future = asyncio.run_coroutine_threadsafe(
                    self.process_query_normal_async(message), self.loop
                )
                
                # 获取处理结果
                resp, final_text, final_tool = future.result(timeout=self.timeout_seconds.get())
                logger.info(f"查询处理完成: 文本长度={len(final_text) if final_text else 0}, 工具信息长度={len(final_tool) if final_tool else 0}")
                
                # 检查是否有错误
                if isinstance(resp, dict) and "error" in resp:
                    # 显示错误消息
                    self.root.after(0, lambda: self.append_to_chat("助手", resp["error"], "error"))
                else:
                    # 显示最终内容
                    if final_text:
                        self.root.after(0, lambda: self.append_to_chat("助手", final_text, "assistant"))
                    else:
                        # 如果没有最终文本，显示回退消息
                        fallback_text = "收到回复但无法解析内容"
                        if isinstance(resp, dict) and "messages" in resp:
                            fallback_text = f"响应包含 {len(resp['messages'])} 个消息"
                        self.root.after(0, lambda: self.append_to_chat("助手", fallback_text, "assistant"))
                
                # 显示工具信息（如果有）
                if final_tool:
                    self.root.after(0, lambda: self.append_to_chat("工具", final_tool, "tool"))
            
            except asyncio.TimeoutError:
                error_msg = f"❌ 查询超时（超过 {self.timeout_seconds.get()} 秒）"
                logger.error(error_msg)
                self.root.after(0, lambda: self.append_to_chat("助手", error_msg, "error"))
            except Exception as e:
                error_msg = f"❌ 处理异常: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.append_to_chat("助手", error_msg, "error"))
    
    async def process_query_async(self, query: str):
        """异步处理用户查询，与 app.py 的 process_query 函数逻辑一致"""
        try:
            if self.agent:
                # 准备输入消息用于日志记录
                input_messages = [{"role": "user", "content": query}]
                
                # 使用保存的模型信息
                model_name = self.current_model_name or "unknown"
                model_provider = self.current_model_provider or "unknown"
                
                # 获取流式回调
                streaming_callback, accumulated_text_obj, accumulated_tool_obj = self.get_streaming_callback()
                
                # 创建带监控的流式回调
                monitored_callback, get_final_record = self.model_logger.create_streaming_wrapper(
                    session_id=self.session_id,
                    thread_id=self.thread_id,
                    model_name=model_name,
                    model_provider=model_provider,
                    input_messages=input_messages,
                    original_callback=streaming_callback
                )
                
                try:
                    # 使用 asyncio.wait_for 进行超时控制
                    response = await asyncio.wait_for(
                        astream_graph(
                            self.agent,
                            {"messages": [HumanMessage(content=query)]},
                            callback=monitored_callback,  # 使用监控回调
                            config=RunnableConfig(
                                recursion_limit=self.recursion_limit.get(),
                                thread_id=self.thread_id,
                            ),
                        ),
                        timeout=self.timeout_seconds.get(),
                    )
                    
                    # 记录最终的调用日志
                    final_record = get_final_record()
                    self.model_logger.log_model_call(final_record)
                    
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ 请求时间超过 {self.timeout_seconds.get()} 秒。请稍候再试。"
                    
                    # 记录超时错误
                    final_record = get_final_record()
                    final_record.error = error_msg
                    self.model_logger.log_model_call(final_record)
                    
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
    
    async def process_query_normal_async(self, query: str):
        """异步处理用户查询 - 普通返回模式（非流式）"""
        try:
            if self.agent:
                # 准备输入消息用于日志记录
                input_messages = [{"role": "user", "content": query}]
                
                # 使用保存的模型信息
                model_name = self.current_model_name or "unknown"
                model_provider = self.current_model_provider or "unknown"
                
                # 记录开始时间
                start_time = time.time()
                
                try:
                    # 使用普通调用（非流式）
                    response = await asyncio.wait_for(
                        self.agent.ainvoke(
                            {"messages": [HumanMessage(content=query)]},
                            config=RunnableConfig(
                                recursion_limit=self.recursion_limit.get(),
                                thread_id=self.thread_id,
                            ),
                        ),
                        timeout=self.timeout_seconds.get(),
                    )
                    
                    logger.info(f"Agent响应类型: {type(response)}")
                    logger.info(f"Agent响应键: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
                    
                    # 处理响应
                    final_text = ""
                    final_tool = ""
                    
                    if "messages" in response:
                        # 只处理最后一条消息，避免累积
                        if len(response["messages"]) > 0:
                            # 获取最后一条消息
                            msg = response["messages"][-1]
                            logger.info(f"处理最后一条消息: 类型={type(msg)}")
                            
                            if hasattr(msg, 'content'):
                                content = msg.content
                                logger.info(f"消息内容类型: {type(content)}")
                                
                                if isinstance(content, str):
                                    final_text = content
                                    logger.info(f"使用字符串内容: {content[:50] if len(content) > 50 else content}")
                                elif isinstance(content, list):
                                    logger.info(f"处理列表内容，长度: {len(content)}")
                                    for content_part in content:
                                        if isinstance(content_part, dict):
                                            if 'text' in content_part:
                                                final_text += content_part['text']
                                                logger.info(f"添加text字段: {content_part['text'][:50]}...")
                                            elif 'content' in content_part:
                                                final_text += str(content_part['content'])
                                                logger.info(f"添加content字段: {str(content_part['content'])[:50]}...")
                                                
                            # 处理工具调用信息
                            if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                                for tool_call in msg.additional_kwargs['tool_calls']:
                                    final_tool += f"🔧 工具调用: {tool_call.get('function', {}).get('name', 'Unknown')}\n"
                                    final_tool += f"参数: {tool_call.get('function', {}).get('arguments', '')}\n\n"
                                    
                            logger.info(f"最终消息文本: '{final_text[:100]}...'")
                        else:
                            logger.warning("响应中的messages列表为空")
                    else:
                        logger.warning("响应中没有找到'messages'键")
                    
                    logger.info(f"最终文本长度: {len(final_text)}, 工具信息长度: {len(final_tool)}")
                    
                    # 记录模型调用
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # 创建简化的日志记录
                    log_record = {
                        'session_id': self.session_id,
                        'thread_id': self.thread_id,
                        'model_name': model_name,
                        'model_provider': model_provider,
                        'input_messages': input_messages,
                        'output_content': final_text,
                        'tool_calls': [],
                        'tool_responses': [],
                        'duration': duration,
                        'timestamp': datetime.now().isoformat(),
                        'error': None
                    }
                    
                    self.model_logger.log_model_call(log_record)
                    
                    return response, final_text, final_tool
                    
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ 请求时间超过 {self.timeout_seconds.get()} 秒。请稍候再试。"
                    
                    # 记录超时错误
                    log_record = {
                        'session_id': self.session_id,
                        'thread_id': self.thread_id,
                        'model_name': model_name,
                        'model_provider': model_provider,
                        'input_messages': input_messages,
                        'output_content': "",
                        'tool_calls': [],
                        'tool_responses': [],
                        'duration': self.timeout_seconds.get(),
                        'timestamp': datetime.now().isoformat(),
                        'error': error_msg
                    }
                    
                    self.model_logger.log_model_call(log_record)
                    
                    return {"error": error_msg}, error_msg, ""
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
        last_update_time = [0]
        update_threshold = 0.2  # 200ms 防抖动
        
        def safe_json_str(obj):
            """安全地将对象转换为 JSON 字符串，处理潜在的嵌套引号和特殊字符"""
            try:
                if isinstance(obj, str):
                    # 如果已经是字符串，尝试解析为 JSON 对象再重新序列化
                    try:
                        parsed = json.loads(obj)
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # 不是有效的 JSON，直接返回经过转义的字符串
                        return obj.replace('```', '\\```')  # 转义可能导致嵌套的 markdown 代码块标记
                else:
                    # 对象转 JSON 字符串
                    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
            except Exception:
                # 如果转换失败，使用 str() 但确保转义关键字符
                return str(obj).replace('```', '\\```')
        
        def callback_func(message: dict):
            nonlocal accumulated_text, accumulated_tool
            message_content = message.get("content", None)
            current_time = time.time()
            
            if isinstance(message_content, AIMessageChunk):
                content = message_content.content
                # 如果内容是列表形式（主要出现在 Claude 模型中）
                if isinstance(content, list) and len(content) > 0:
                    message_chunk = content[0]
                    # 处理文本类型
                    if message_chunk["type"] == "text":
                        # 确保文本安全，防止嵌套格式问题
                        text = message_chunk["text"]
                        accumulated_text.append(text)
                        # 智能防抖动更新
                        if (current_time - last_update_time[0] > update_threshold or 
                            len("".join(accumulated_text)) % 50 == 0):
                            full_text = "".join(accumulated_text)
                            last_update_time[0] = current_time
                            self.root.after(0, lambda text=full_text: self.update_streaming_text(text))
                    # 处理工具使用类型
                    elif message_chunk["type"] == "tool_use":
                        if "partial_json" in message_chunk:
                            # 处理部分 JSON
                            json_content = safe_json_str(message_chunk["partial_json"])
                            accumulated_tool.append(json_content)
                        else:
                            tool_call_chunks = message_content.tool_call_chunks
                            if tool_call_chunks:
                                tool_call_chunk = tool_call_chunks[0]
                                # 安全地转换工具调用为 JSON
                                json_content = safe_json_str(tool_call_chunk)
                                accumulated_tool.append("\n```json\n" + json_content + "\n```\n")
                        # 更新工具信息
                        if accumulated_tool:
                            self.root.after(0, lambda: self.update_tool_info("".join(accumulated_tool)))
                # 处理如果 tool_calls 属性存在（主要出现在 OpenAI 模型中）
                elif (
                    hasattr(message_content, "tool_calls")
                    and message_content.tool_calls
                    and len(message_content.tool_calls[0]["name"]) > 0
                ):
                    tool_call_info = message_content.tool_calls[0]
                    # 安全地转换工具调用为 JSON
                    json_content = safe_json_str(tool_call_info)
                    accumulated_tool.append("\n```json\n" + json_content + "\n```\n")
                    self.root.after(0, lambda: self.update_tool_info("".join(accumulated_tool)))
                # 处理如果内容是简单字符串
                elif isinstance(content, str):
                    accumulated_text.append(content)
                    # 智能防抖动更新
                    if (current_time - last_update_time[0] > update_threshold or 
                        len("".join(accumulated_text)) % 30 == 0):
                        full_text = "".join(accumulated_text)
                        last_update_time[0] = current_time
                        self.root.after(0, lambda text=full_text: self.update_streaming_text(text))
            # 处理如果是工具消息（工具响应）
            elif hasattr(message_content, '__class__') and 'ToolMessage' in str(message_content.__class__):
                # 安全地处理工具消息内容
                json_content = safe_json_str(message_content.content)
                accumulated_tool.append("\n```json\n" + json_content + "\n```\n")
                self.root.after(0, lambda: self.update_tool_info("".join(accumulated_tool)))
            return None
        
        return callback_func, accumulated_text, accumulated_tool
    
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
        
        # 记录消息的开始位置（在插入之前）
        start_pos = self.chat_history.index(tk.END)
        
        # 插入消息
        full_message = f"[{timestamp}] {sender}: {message}\n\n"
        self.chat_history.insert(tk.END, full_message)
        self.chat_history.see(tk.END)
        
        # 记录消息的结束位置（在插入之后）
        end_pos = self.chat_history.index(tk.END)
        
        # 如果是助手消息，保存完整的位置信息
        if msg_type == "assistant":
            self._current_assistant_message_start = start_pos
            self._current_assistant_message_end = end_pos
            logger.debug(f"记录助手消息位置: {start_pos} 到 {end_pos}")
        elif msg_type == "tool":
            self._current_tool_message_start = start_pos
            self._current_tool_message_end = end_pos
        
        # 应用颜色标签
        self.chat_history.tag_add(f"msg_{msg_type}", start_pos, end_pos)
        self.chat_history.tag_config(f"msg_{msg_type}", foreground=color)
        
        # 保存聊天历史
        self.chat_messages.append({
            "sender": sender,
            "message": message,
            "timestamp": timestamp,
            "type": msg_type
        })
    
    def replace_last_assistant_message(self, message: str):
        """替换最后一条助手消息"""
        try:
            logger.info(f"开始替换助手消息，新消息长度: {len(message)}")
            logger.info(f"当前消息历史长度: {len(self.chat_messages)}")
            
            # 打印当前消息历史
            for i, msg in enumerate(self.chat_messages):
                logger.info(f"消息 {i}: {msg['type']} - {msg['sender']} - {msg['message'][:50]}...")
            
            # 找到最后一条助手消息并替换
            found_assistant = False
            for i in range(len(self.chat_messages) - 1, -1, -1):
                if self.chat_messages[i]["type"] == "assistant":
                    logger.info(f"找到助手消息在位置 {i}: {self.chat_messages[i]['message'][:50]}...")
                    # 更新最后一条助手消息
                    old_message = self.chat_messages[i]["message"]
                    self.chat_messages[i]["message"] = message
                    self.chat_messages[i]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    logger.info(f"消息已更新: '{old_message[:50]}...' -> '{message[:50]}...'")
                    found_assistant = True
                    break
            
            if not found_assistant:
                logger.warning("没有找到助手消息，添加新的")
                # 如果没有找到助手消息，添加新的
                self.chat_messages.append({
                    "sender": "助手",
                    "message": message,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "assistant"
                })
            
            # 重建聊天历史显示
            logger.info("开始重建聊天历史")
            self.rebuild_chat_history()
            logger.info("助手消息已成功替换")
            
        except Exception as e:
            logger.error(f"替换助手消息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 回退到添加新消息
            self.append_to_chat("助手", message, "assistant")
    
    def rebuild_chat_history(self):
        """重建聊天历史显示"""
        try:
            # 清空当前显示
            self.chat_history.delete(1.0, tk.END)
            
            # 重新添加所有消息
            for msg_data in self.chat_messages:
                timestamp = msg_data["timestamp"]
                sender = msg_data["sender"]
                message = msg_data["message"]
                msg_type = msg_data["type"]
                
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
                elif msg_type == "tool":
                    color = "purple"
                else:
                    color = "black"
                
                # 插入消息
                start_pos = self.chat_history.index(tk.END)
                full_message = f"[{timestamp}] {sender}: {message}\n\n"
                self.chat_history.insert(tk.END, full_message)
                end_pos = self.chat_history.index(tk.END)
                
                # 应用颜色标签
                self.chat_history.tag_add(f"msg_{msg_type}", start_pos, end_pos)
                self.chat_history.tag_config(f"msg_{msg_type}", foreground=color)
            
            # 滚动到底部
            self.chat_history.see(tk.END)
            
        except Exception as e:
            logger.error(f"重建聊天历史时出错: {e}")
    
    def update_streaming_text(self, text):
        """更新流式文本显示"""
        try:
            logger.info(f"更新流式文本: 文本长度={len(text)}")
            
            # 使用结构化消息列表方式更新
            found_assistant = False
            for i in range(len(self.chat_messages) - 1, -1, -1):
                if self.chat_messages[i]["type"] == "assistant":
                    # 更新最后一条助手消息
                    self.chat_messages[i]["message"] = text
                    self.chat_messages[i]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    found_assistant = True
                    logger.info(f"更新了流式消息 {i}: 新文本长度={len(text)}")
                    break
            
            if not found_assistant:
                # 如果没有找到助手消息，添加新的
                logger.warning("未找到要更新的助手消息，添加新消息")
                self.chat_messages.append({
                    "sender": "助手",
                    "message": text,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "assistant"
                })
            
            # 重建聊天历史显示
            self.rebuild_chat_history()
            
        except Exception as e:
            logger.error(f"更新流式文本出错: {str(e)}\n{traceback.format_exc()}")
    
    def update_tool_info(self, tool_info):
        """更新工具调用信息显示"""
        if not tool_info.strip():
            return
        
        try:
            logger.info(f"更新工具信息: 信息长度={len(tool_info)}")
            
            # 查找最近的工具消息并更新
            current_query_id = self.get_current_query_id_from_tool_info(tool_info)
            
            # 如果能从工具信息中提取查询ID，则查找匹配的工具消息进行更新
            found_matching_tool = False
            if current_query_id:
                # 仅在当前会话中查找带有相同ID的工具消息
                for i in range(len(self.chat_messages) - 1, -1, -1):
                    if self.chat_messages[i]["type"] == "tool":
                        tool_id = self.get_current_query_id_from_tool_info(self.chat_messages[i]["message"])
                        if tool_id and tool_id == current_query_id:
                            # 更新带有相同ID的工具消息
                            logger.info(f"更新工具消息 ID={current_query_id}")
                            self.chat_messages[i]["message"] = tool_info
                            self.chat_messages[i]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                            found_matching_tool = True
                            break
            
            # 如果没有找到匹配的工具消息，添加新的
            if not found_matching_tool:
                logger.info("添加新的工具消息")
                self.chat_messages.append({
                    "sender": "工具",
                    "message": tool_info,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "tool"
                })
            
            # 重建聊天历史
            self.rebuild_chat_history()
        except Exception as e:
            logger.error(f"更新工具信息时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_current_query_id_from_tool_info(self, tool_info):
        """从工具信息中提取查询ID"""
        try:
            # 尝试提取工具调用ID，例如 'id': 'call_7afdae80290d42a1801391'
            import re
            match = re.search(r"'id':\s*'([^']+)'|\"id\":\s*\"([^\"]+)\"", tool_info)
            if match:
                call_id = match.group(1) or match.group(2)
                logger.info(f"从工具信息中提取到查询ID: {call_id}")
                return call_id
            return None
        except Exception:
            return None
    
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
        # 确保窗口显示在合适位置
        self.center_window(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """应用关闭处理"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()
    
    def show_log_stats(self):
        """显示日志统计信息窗口"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 模型调用日志统计")
        stats_window.geometry("800x600")
        stats_window.transient(self.root)
        
        # 相对于主窗口居中显示
        self.center_child_window(self.root, stats_window)
        
        # 创建文本框显示统计信息
        text_frame = ttk.Frame(stats_window, padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # 滚动文本框
        stats_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True)
        
        # 获取并显示统计信息
        try:
            stats_summary = self.model_tracker.get_stats_summary()
            stats_text.insert(tk.END, stats_summary)
            
            # 添加日志文件路径信息
            stats_text.insert(tk.END, f"\n\n📁 日志文件位置:\n")
            stats_text.insert(tk.END, f"  {os.path.abspath('logs')}\n")
            stats_text.insert(tk.END, f"\n💡 提示: 日志以JSON Lines格式存储，每行一条记录")
            
        except Exception as e:
            stats_text.insert(tk.END, f"❌ 获取统计信息失败: {str(e)}")
        
        # 配置文本框为只读
        stats_text.config(state=tk.DISABLED)
        
        # 按钮框架
        button_frame = ttk.Frame(stats_window, padding=10)
        button_frame.pack(fill=tk.X)
        
        # 刷新按钮
        def refresh_stats():
            stats_text.config(state=tk.NORMAL)
            stats_text.delete(1.0, tk.END)
            try:
                stats_summary = self.model_tracker.get_stats_summary()
                stats_text.insert(tk.END, stats_summary)
                stats_text.insert(tk.END, f"\n\n📁 日志文件位置:\n")
                stats_text.insert(tk.END, f"  {os.path.abspath('logs')}\n")
                stats_text.insert(tk.END, f"\n💡 提示: 日志以JSON Lines格式存储，每行一条记录")
            except Exception as e:
                stats_text.insert(tk.END, f"❌ 获取统计信息失败: {str(e)}")
            stats_text.config(state=tk.DISABLED)
        
        ttk.Button(button_frame, text="🔄 刷新", command=refresh_stats).pack(side=tk.LEFT, padx=(0, 5))
        
        # 打开日志文件夹按钮
        def open_log_folder():
            log_path = os.path.abspath("logs")
            if os.path.exists(log_path):
                if sys.platform.startswith('win'):
                    os.startfile(log_path)
                elif sys.platform.startswith('darwin'):  # macOS
                    os.system(f'open "{log_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{log_path}"')
            else:
                messagebox.showwarning("警告", "日志文件夹不存在")
        
        ttk.Button(button_frame, text="📁 打开日志文件夹", command=open_log_folder).pack(side=tk.LEFT, padx=(0, 5))
        
        # 关闭按钮
        ttk.Button(button_frame, text="❌ 关闭", command=stats_window.destroy).pack(side=tk.RIGHT)

    @staticmethod
    def center_window(window):
        """将窗口居中显示在屏幕上
        
        Args:
            window: 要居中的窗口
        """
        # 先刷新以确保获取正确的窗口尺寸
        window.update_idletasks()
        
        # 获取窗口尺寸
        window_width = window.winfo_reqwidth()
        window_height = window.winfo_reqheight()
        
        # 获取屏幕尺寸
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        # 计算居中位置
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)
        
        # 设置窗口位置
        window.geometry(f"+{x}+{y}")

    @staticmethod
    def center_child_window(parent_window, child_window):
        """将子窗口相对于父窗口居中显示
        
        Args:
            parent_window: 父窗口
            child_window: 要居中的子窗口
        """
        # 先刷新以确保获取正确的窗口尺寸
        child_window.update_idletasks()
        
        # 获取父窗口信息
        parent_x = parent_window.winfo_rootx()
        parent_y = parent_window.winfo_rooty()
        parent_width = parent_window.winfo_width()
        parent_height = parent_window.winfo_height()
        
        # 获取子窗口尺寸
        child_width = child_window.winfo_reqwidth()
        child_height = child_window.winfo_reqheight()
        
        # 获取屏幕尺寸
        screen_width = parent_window.winfo_screenwidth()
        screen_height = parent_window.winfo_screenheight()
        
        # 计算子窗口居中位置
        x = parent_x + (parent_width - child_width) // 2
        y = parent_y + (parent_height - child_height) // 2
        
        # 确保窗口不会超出屏幕边界
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + child_width > screen_width:
            x = max(0, screen_width - child_width)
        if y + child_height > screen_height:
            y = max(0, screen_height - child_height)
        
        # 设置子窗口位置
        child_window.geometry(f"+{x}+{y}")



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
