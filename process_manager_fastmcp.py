# process_manager_fastmcp.py

from fastmcp import FastMCP
import psutil, win32gui, win32process

mcp = FastMCP(name="Process Manager", instructions="Windows 本地进程管理 MCP 工具")

def normalize_name(name: str) -> list:
    """生成包含各种可能变体的进程名列表"""
    name = name.lower()
    candidates = [name]
    if not name.endswith('.exe'):
        candidates.append(name + '.exe')
    return candidates

# 使用langchain的工具装饰器
@mcp.tool
def list_processes() -> str:
    """列出当前所有进程和对应的名称"""
    proc_list = [(p.pid, p.name()) for p in psutil.process_iter()]
    return "\n".join([f"PID: {pid}, Name: {name}" for pid, name in proc_list])

@mcp.tool
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

@mcp.tool
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

@mcp.tool
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


@mcp.tool
def get_process_memory(name: str) -> str:
    """获取某个进程的内存使用情况（MB）"""
    for p in psutil.process_iter():
        if p.name().lower() == name.lower():
            mem = p.memory_info().rss / (1024 * 1024)
            return f"进程 {name} 占用内存：{mem:.2f} MB"
    return f"未找到名为 {name} 的进程"

# I:\pdfai_serv\mcp\Python310-agent\Python310-agent\python.exe process_manager_fastmcp.py
if __name__ == "__main__":
    mcp.run()  # 默认使用 stdio transport
