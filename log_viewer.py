#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型调用日志查看工具
独立的日志查看和分析工具
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse


class LogViewer:
    """日志查看器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def list_log_files(self) -> List[Path]:
        """列出所有日志文件"""
        if not self.log_dir.exists():
            return []
        return sorted(self.log_dir.glob("model_calls_*.jsonl"))
    
    def read_log_file(self, log_file: Path) -> List[Dict[str, Any]]:
        """读取日志文件"""
        records = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")
        return records
    
    def show_summary(self, date: str = None):
        """显示日志摘要"""
        if date:
            log_files = [self.log_dir / f"model_calls_{date}.jsonl"]
        else:
            log_files = self.list_log_files()
        
        print("📊 模型调用日志摘要")
        print("=" * 50)
        
        total_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_time = 0
        models_used = {}
        tools_used = set()
        errors = 0
        
        for log_file in log_files:
            if not log_file.exists():
                continue
                
            print(f"\n📁 日志文件: {log_file.name}")
            records = self.read_log_file(log_file)
            
            file_calls = len(records)
            file_input_tokens = sum(r.get('input_tokens', 0) or 0 for r in records)
            file_output_tokens = sum(r.get('output_tokens', 0) or 0 for r in records)
            file_time = sum(r.get('processing_time', 0) for r in records)
            file_errors = sum(1 for r in records if r.get('error'))
            
            print(f"  • 调用次数: {file_calls}")
            print(f"  • 输入Token: {file_input_tokens:,}")
            print(f"  • 输出Token: {file_output_tokens:,}")
            print(f"  • 总耗时: {file_time:.2f}秒")
            print(f"  • 错误次数: {file_errors}")
            
            total_calls += file_calls
            total_input_tokens += file_input_tokens
            total_output_tokens += file_output_tokens
            total_time += file_time
            errors += file_errors
            
            for record in records:
                model_key = f"{record.get('model_provider', 'unknown')}/{record.get('model_name', 'unknown')}"
                models_used[model_key] = models_used.get(model_key, 0) + 1
                
                for tool_call in record.get('tool_calls', []):
                    tools_used.add(tool_call.get('name', 'unknown'))
        
        print(f"\n📈 总计统计:")
        print(f"  • 总调用次数: {total_calls}")
        print(f"  • 总输入Token: {total_input_tokens:,}")
        print(f"  • 总输出Token: {total_output_tokens:,}")
        print(f"  • 总Token数: {(total_input_tokens + total_output_tokens):,}")
        print(f"  • 总耗时: {total_time:.2f}秒")
        print(f"  • 平均耗时: {(total_time/total_calls if total_calls > 0 else 0):.2f}秒")
        print(f"  • 错误次数: {errors}")
        
        if models_used:
            print(f"\n🤖 使用的模型:")
            for model, count in sorted(models_used.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {model}: {count}次")
        
        if tools_used:
            print(f"\n🛠️ 使用的工具:")
            for tool in sorted(tools_used):
                print(f"  • {tool}")
    
    def show_detailed_log(self, date: str = None, limit: int = 10):
        """显示详细日志记录"""
        if date:
            log_files = [self.log_dir / f"model_calls_{date}.jsonl"]
        else:
            log_files = self.list_log_files()
        
        print("📋 详细日志记录")
        print("=" * 80)
        
        count = 0
        for log_file in log_files:
            if not log_file.exists():
                continue
                
            records = self.read_log_file(log_file)
            for record in reversed(records):  # 最新的在前
                if count >= limit:
                    break
                
                print(f"\n🕐 {record.get('timestamp', 'N/A')}")
                print(f"🔗 会话: {record.get('session_id', 'N/A')} | 线程: {record.get('thread_id', 'N/A')}")
                print(f"🤖 模型: {record.get('model_provider', 'N/A')}/{record.get('model_name', 'N/A')}")
                print(f"📝 输入Token: {record.get('input_tokens', 'N/A')} | 输出Token: {record.get('output_tokens', 'N/A')}")
                print(f"⏱️ 耗时: {record.get('processing_time', 0):.2f}秒")
                
                if record.get('tool_calls'):
                    print(f"🛠️ 工具调用: {len(record['tool_calls'])}个")
                    for tool_call in record['tool_calls']:
                        print(f"   • {tool_call.get('name', 'unknown')}")
                
                if record.get('error'):
                    print(f"❌ 错误: {record['error']}")
                
                # 显示输入消息摘要
                input_msg = record.get('input_messages', [])
                if input_msg:
                    content = input_msg[0].get('content', '')
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"💬 输入: {preview}")
                
                # 显示输出摘要
                output = record.get('output_content', '')
                if output:
                    preview = output[:100] + "..." if len(output) > 100 else output
                    print(f"💭 输出: {preview}")
                
                print("-" * 80)
                count += 1
    
    def export_csv(self, date: str = None, output_file: str = None):
        """导出为CSV格式"""
        import csv
        
        if date:
            log_files = [self.log_dir / f"model_calls_{date}.jsonl"]
        else:
            log_files = self.list_log_files()
        
        if not output_file:
            output_file = f"model_calls_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        fieldnames = [
            'timestamp', 'session_id', 'thread_id', 'model_provider', 'model_name',
            'input_tokens', 'output_tokens', 'processing_time', 'tool_calls_count',
            'error', 'input_preview', 'output_preview'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for log_file in log_files:
                if not log_file.exists():
                    continue
                
                records = self.read_log_file(log_file)
                for record in records:
                    # 准备CSV行数据
                    row = {
                        'timestamp': record.get('timestamp', ''),
                        'session_id': record.get('session_id', ''),
                        'thread_id': record.get('thread_id', ''),
                        'model_provider': record.get('model_provider', ''),
                        'model_name': record.get('model_name', ''),
                        'input_tokens': record.get('input_tokens', 0),
                        'output_tokens': record.get('output_tokens', 0),
                        'processing_time': record.get('processing_time', 0),
                        'tool_calls_count': len(record.get('tool_calls', [])),
                        'error': record.get('error', ''),
                    }
                    
                    # 输入预览
                    input_msgs = record.get('input_messages', [])
                    if input_msgs:
                        content = input_msgs[0].get('content', '')
                        row['input_preview'] = content[:200]
                    
                    # 输出预览
                    output = record.get('output_content', '')
                    row['output_preview'] = output[:200] if output else ''
                    
                    writer.writerow(row)
        
        print(f"✅ 已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="模型调用日志查看工具")
    parser.add_argument('--log-dir', default='logs', help='日志目录路径')
    parser.add_argument('--date', help='指定日期 (格式: YYYYMMDD)')
    parser.add_argument('--summary', action='store_true', help='显示摘要统计')
    parser.add_argument('--detailed', action='store_true', help='显示详细日志')
    parser.add_argument('--limit', type=int, default=10, help='详细日志显示条数限制')
    parser.add_argument('--export-csv', action='store_true', help='导出为CSV文件')
    parser.add_argument('--output', help='导出文件名')
    
    args = parser.parse_args()
    
    viewer = LogViewer(args.log_dir)
    
    if args.export_csv:
        viewer.export_csv(args.date, args.output)
    elif args.detailed:
        viewer.show_detailed_log(args.date, args.limit)
    else:
        viewer.show_summary(args.date)


if __name__ == "__main__":
    main()
