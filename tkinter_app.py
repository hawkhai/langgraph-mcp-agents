#!/usr/bin/env python3
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
import uuid

# 导入现有模块
from utils import astream_graph, ainvoke_graph
from langchain_mcp_windows_proc import MultiServerMCPClient
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 输出令牌配置
OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-20241022": {"max_tokens": 4096},
    "claude-3-5-haiku-20241022": {"max_tokens": 4096},
    "claude-3-7-sonnet-latest": {"max_tokens": 4096},
    "gpt-4o": {"max_tokens": 4096},
    "gpt-4o-mini": {"max_tokens": 16384},
    "qwen2.5-72b-instruct": {"max_tokens": 8192}
}

# 系统提示模板
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
        self.thread_id = str(uuid.uuid4())
        
        # 配置变量
        self.selected_model = tk.StringVar(value="claude-3-5-sonnet-20241022")
        self.timeout_seconds = tk.IntVar(value=120)
        self.recursion_limit = tk.IntVar(value=100)
        self.mcp_config = {}
        
        # 创建 UI
        self.create_widgets()
        self.load_config()
        
        # 异步事件循环
        self.loop = None
        self.start_async_loop()
    
    def start_async_loop(self):
        """在单独线程中启动异步事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    
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
        model_combo = ttk.Combobox(settings_frame, textvariable=self.selected_model, 
                                  values=list(OUTPUT_TOKEN_INFO.keys()), state="readonly")
        model_combo.pack(fill=tk.X, pady=(0, 10))
        
        # 超时设置
        ttk.Label(settings_frame, text="⏱️ 响应时间限制 (秒):").pack(anchor=tk.W, pady=(0, 5))
        timeout_scale = ttk.Scale(settings_frame, from_=60, to=300, 
                                 variable=self.timeout_seconds, orient=tk.HORIZONTAL)
        timeout_scale.pack(fill=tk.X, pady=(0, 5))
        timeout_label = ttk.Label(settings_frame, textvariable=self.timeout_seconds)
        timeout_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 递归限制
        ttk.Label(settings_frame, text="🔄 递归调用限制:").pack(anchor=tk.W, pady=(0, 5))
        recursion_scale = ttk.Scale(settings_frame, from_=10, to=200, 
                                   variable=self.recursion_limit, orient=tk.HORIZONTAL)
        recursion_scale.pack(fill=tk.X, pady=(0, 5))
        recursion_label = ttk.Label(settings_frame, textvariable=self.recursion_limit)
        recursion_label.pack(anchor=tk.W, pady=(0, 10))
        
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
                self.mcp_config = {}
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
        self.append_to_chat("系统", "🔄 正在应用设置...", "info")
        
        def apply_async():
            future = asyncio.run_coroutine_threadsafe(
                self.initialize_session_async(), self.loop
            )
            try:
                success = future.result(timeout=30)
                if success:
                    self.root.after(0, lambda: self.append_to_chat("系统", "✅ 设置应用成功", "success"))
                    self.session_initialized = True
                    self.update_status()
                else:
                    self.root.after(0, lambda: self.append_to_chat("系统", "❌ 设置应用失败", "error"))
            except Exception as e:
                self.root.after(0, lambda: self.append_to_chat("系统", f"❌ 错误: {str(e)}", "error"))
        
        threading.Thread(target=apply_async, daemon=True).start()
    
    async def initialize_session_async(self):
        """异步初始化会话"""
        try:
            # 清理现有客户端
            if self.mcp_client:
                await self.mcp_client.close()
            
            # 创建 MCP 客户端
            self.mcp_client = MultiServerMCPClient()
            
            # 配置服务器
            for tool_name, config in self.mcp_config.items():
                self.mcp_client.add_server(tool_name, config)
            
            # 连接所有服务器
            await self.mcp_client.connect_all()
            
            # 创建 Agent
            model_name = self.selected_model.get()
            
            # 根据模型创建不同的 LLM
            if "claude" in model_name:
                llm = ChatAnthropic(
                    model=model_name,
                    temperature=0.1,
                    max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                )
            elif "qwen" in model_name:
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
            
            # 创建提示模板
            prompt_template = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\n{input}\n\nAgent scratchpad:\n{agent_scratchpad}")
            
            # 获取工具
            tools = await self.mcp_client.get_tools()
            
            # 创建 ReAct Agent
            agent = create_react_agent(llm, tools, prompt_template)
            self.agent = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=self.recursion_limit.get()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_history = []
        self.thread_id = str(uuid.uuid4())
        self.chat_history.delete(1.0, tk.END)
        self.append_to_chat("系统", "✅ 对话已重置", "success")
    
    def send_message(self):
        """发送用户消息"""
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        if not self.session_initialized:
            messagebox.showwarning("警告", "请先点击'应用设置'初始化系统")
            return
        
        # 清空输入框
        self.user_input.delete(0, tk.END)
        
        # 显示用户消息
        self.append_to_chat("用户", user_text, "user")
        
        # 处理消息
        def process_async():
            future = asyncio.run_coroutine_threadsafe(
                self.process_query_async(user_text), self.loop
            )
            try:
                response = future.result(timeout=self.timeout_seconds.get())
                self.root.after(0, lambda: self.append_to_chat("助手", response, "assistant"))
            except asyncio.TimeoutError:
                self.root.after(0, lambda: self.append_to_chat("系统", "⏱️ 请求超时", "error"))
            except Exception as e:
                self.root.after(0, lambda: self.append_to_chat("系统", f"❌ 错误: {str(e)}", "error"))
        
        threading.Thread(target=process_async, daemon=True).start()
    
    async def process_query_async(self, query: str) -> str:
        """异步处理用户查询"""
        try:
            if not self.agent:
                return "❌ Agent 未初始化"
            
            # 使用 astream_graph 进行流式处理
            response_text = ""
            async for chunk in astream_graph(
                self.agent, 
                {"input": query, "thread_id": self.thread_id},
                stream_mode="messages"
            ):
                if hasattr(chunk, 'content') and chunk.content:
                    response_text += str(chunk.content)
            
            return response_text if response_text else "未收到响应"
            
        except Exception as e:
            logger.error(f"查询处理错误: {e}")
            return f"❌ 处理错误: {str(e)}"
    
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
            self.status_label.config(text="状态: 已初始化")
            tool_count = len(self.mcp_config)
            self.tool_count_label.config(text=f"工具数量: {tool_count}")
        else:
            self.status_label.config(text="状态: 未初始化")
            self.tool_count_label.config(text="工具数量: 0")
    
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
    load_dotenv()
    
    # 创建并运行应用
    app = MCPAgentApp()
    app.run()


if __name__ == "__main__":
    main()
