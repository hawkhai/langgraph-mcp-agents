#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型调用监控和日志记录模块
记录所有模型调用的输入参数、输出结果、工具调用等详细信息
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import threading
from dataclasses import dataclass, asdict
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage


@dataclass
class ModelCallRecord:
    """模型调用记录数据结构"""
    timestamp: str
    session_id: str
    thread_id: str
    model_name: str
    model_provider: str
    input_messages: List[Dict]
    input_tokens: Optional[int]
    output_content: str
    output_tokens: Optional[int]
    tool_calls: List[Dict]
    tool_responses: List[Dict]
    processing_time: float
    error: Optional[str]
    metadata: Dict[str, Any]


class ModelLogger:
    """大模型调用日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建日志文件
        log_file = self.log_dir / f"model_calls_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # 配置JSON Lines格式的日志记录器
        self.logger = logging.getLogger('model_calls')
        self.logger.setLevel(logging.INFO)
        
        # 移除现有的处理器，避免重复
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件处理器 - JSON Lines格式
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        
        # 可选：控制台处理器用于调试
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # 线程锁，确保日志写入的线程安全
        self._lock = threading.Lock()
        
        print(f"📊 大模型调用日志记录器已启动，日志文件: {log_file}")
    
    def log_model_call(self, record: ModelCallRecord):
        """记录模型调用"""
        with self._lock:
            try:
                # 转换为字典并序列化为JSON
                record_dict = asdict(record)
                json_str = json.dumps(record_dict, ensure_ascii=False, separators=(',', ':'))
                self.logger.info(json_str)
                
                # 在控制台显示简要信息
                print(f"📝 [{record.timestamp}] {record.model_provider}/{record.model_name} "
                      f"| 输入:{record.input_tokens or '?'}t | 输出:{record.output_tokens or '?'}t "
                      f"| 耗时:{record.processing_time:.2f}s "
                      f"| 工具:{len(record.tool_calls)}")
                
            except Exception as e:
                print(f"❌ 日志记录失败: {e}")
    
    def create_streaming_wrapper(self, 
                                session_id: str,
                                thread_id: str,
                                model_name: str,
                                model_provider: str,
                                input_messages: List[Dict],
                                original_callback: Optional[Callable] = None):
        """
        创建流式调用包装器，监控整个流式过程
        
        Returns:
            wrapped_callback: 包装后的回调函数
            get_final_record: 获取最终记录的函数
        """
        
        # 记录开始时间
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # 累积数据
        accumulated_content = []
        tool_calls = []
        tool_responses = []
        error_info = None
        metadata = {}
        
        def wrapped_callback(chunk_data: Dict[str, Any]):
            """包装的流式回调函数"""
            nonlocal accumulated_content, tool_calls, tool_responses, error_info, metadata
            
            try:
                node = chunk_data.get("node", "")
                content = chunk_data.get("content", "")
                
                # 记录节点信息
                if "metadata" in chunk_data:
                    metadata.update(chunk_data["metadata"])
                
                # 处理不同类型的消息
                if hasattr(content, 'content'):
                    # AI消息内容
                    if isinstance(content.content, str):
                        accumulated_content.append(content.content)
                    elif isinstance(content.content, list):
                        for item in content.content:
                            if isinstance(item, dict) and "text" in item:
                                accumulated_content.append(item["text"])
                
                # 处理工具调用
                if hasattr(content, 'tool_calls') and content.tool_calls:
                    for tool_call in content.tool_calls:
                        tool_calls.append({
                            "id": getattr(tool_call, 'id', ''),
                            "name": getattr(tool_call, 'name', ''),
                            "args": getattr(tool_call, 'args', {}),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # 处理工具响应
                if isinstance(content, ToolMessage):
                    tool_responses.append({
                        "tool_call_id": getattr(content, 'tool_call_id', ''),
                        "content": getattr(content, 'content', ''),
                        "name": getattr(content, 'name', ''),
                        "timestamp": datetime.now().isoformat()
                    })
                
                # 调用原始回调
                if original_callback:
                    return original_callback(chunk_data)
                    
            except Exception as e:
                error_info = str(e)
                print(f"⚠️ 流式监控回调错误: {e}")
        
        def get_final_record() -> ModelCallRecord:
            """获取最终的调用记录"""
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 计算token数量（估算）
            final_content = "".join(accumulated_content)
            estimated_output_tokens = len(final_content) // 4 if final_content else 0
            estimated_input_tokens = sum(len(str(msg)) for msg in input_messages) // 4
            
            return ModelCallRecord(
                timestamp=timestamp,
                session_id=session_id,
                thread_id=thread_id,
                model_name=model_name,
                model_provider=model_provider,
                input_messages=input_messages,
                input_tokens=estimated_input_tokens,
                output_content=final_content,
                output_tokens=estimated_output_tokens,
                tool_calls=tool_calls,
                tool_responses=tool_responses,
                processing_time=processing_time,
                error=error_info,
                metadata=metadata
            )
        
        return wrapped_callback, get_final_record


class ModelCallTracker:
    """模型调用跟踪器，提供统计信息"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def get_today_stats(self) -> Dict[str, Any]:
        """获取今日调用统计"""
        today = datetime.now().strftime('%Y%m%d')
        log_file = self.log_dir / f"model_calls_{today}.jsonl"
        
        if not log_file.exists():
            return {"total_calls": 0, "total_tokens": 0, "total_time": 0}
        
        stats = {
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_time": 0,
            "models": {},
            "tools_used": set(),
            "errors": 0
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        stats["total_calls"] += 1
                        stats["total_input_tokens"] += record.get("input_tokens", 0) or 0
                        stats["total_output_tokens"] += record.get("output_tokens", 0) or 0
                        stats["total_time"] += record.get("processing_time", 0)
                        
                        # 按模型统计
                        model_key = f"{record.get('model_provider', 'unknown')}/{record.get('model_name', 'unknown')}"
                        if model_key not in stats["models"]:
                            stats["models"][model_key] = 0
                        stats["models"][model_key] += 1
                        
                        # 工具使用统计
                        for tool_call in record.get("tool_calls", []):
                            stats["tools_used"].add(tool_call.get("name", "unknown"))
                        
                        # 错误统计
                        if record.get("error"):
                            stats["errors"] += 1
            
            stats["tools_used"] = list(stats["tools_used"])
            stats["total_tokens"] = stats["total_input_tokens"] + stats["total_output_tokens"]
            
        except Exception as e:
            print(f"❌ 统计数据读取失败: {e}")
        
        return stats
    
    def get_stats_summary(self) -> str:
        """获取统计摘要"""
        stats = self.get_today_stats()
        
        summary = f"""
📊 今日模型调用统计
════════════════════
🔢 总调用次数: {stats['total_calls']}
📝 总输入Token: {stats['total_input_tokens']:,}
📋 总输出Token: {stats['total_output_tokens']:,}
🎯 总Token数: {stats['total_tokens']:,}
⏱️ 总耗时: {stats['total_time']:.2f}秒
⚠️ 错误次数: {stats['errors']}

🤖 使用的模型:
{chr(10).join(f"  • {model}: {count}次" for model, count in stats['models'].items())}

🛠️ 使用的工具:
{chr(10).join(f"  • {tool}" for tool in stats['tools_used'])}
"""
        return summary


# 全局日志记录器实例
_global_logger: Optional[ModelLogger] = None


def get_model_logger() -> ModelLogger:
    """获取全局模型日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ModelLogger()
    return _global_logger


def init_model_logging(log_dir: str = "logs"):
    """初始化模型日志记录"""
    global _global_logger
    _global_logger = ModelLogger(log_dir)
    return _global_logger


# 便捷函数
def log_model_call(**kwargs):
    """记录模型调用的便捷函数"""
    logger = get_model_logger()
    record = ModelCallRecord(**kwargs)
    logger.log_model_call(record)


if __name__ == "__main__":
    # 测试代码
    logger = ModelLogger()
    tracker = ModelCallTracker()
    
    # 模拟一次调用记录
    test_record = ModelCallRecord(
        timestamp=datetime.now().isoformat(),
        session_id="test_session",
        thread_id="test_thread",
        model_name="gpt-4o",
        model_provider="openai",
        input_messages=[{"role": "user", "content": "测试消息"}],
        input_tokens=10,
        output_content="测试回复",
        output_tokens=5,
        tool_calls=[],
        tool_responses=[],
        processing_time=1.5,
        error=None,
        metadata={}
    )
    
    logger.log_model_call(test_record)
    print(tracker.get_stats_summary())
