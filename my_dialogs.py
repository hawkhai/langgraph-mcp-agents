#encoding=utf-8


class ToolConfigWindow:
    """工具配置窗口"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.window = tk.Toplevel(parent_app.root)
        self.window.title("🔧 工具配置")
        self.window.geometry("800x600")
        self.window.transient(parent_app.root)
        self.window.grab_set()
        
        # 相对于主窗口居中显示
        MCPAgentApp.center_child_window(parent_app.root, self.window)
        
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
        
        # 相对于父窗口居中显示
        MCPAgentApp.center_child_window(parent_window.window, self.dialog)
        
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
