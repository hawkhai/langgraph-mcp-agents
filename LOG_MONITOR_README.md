# 🔍 大模型调用监控和日志系统

本系统为MCP智能代理应用提供完整的大模型调用监控和日志记录功能，帮助您跟踪和分析所有的模型调用情况。

## 🚀 功能特性

### 📊 实时监控
- **完整调用记录**: 记录所有模型调用的输入、输出、工具调用等详细信息
- **性能统计**: 监控响应时间、Token使用量、错误率等关键指标
- **流式监控**: 支持流式响应的实时监控和记录
- **工具调用跟踪**: 详细记录每次工具调用的参数和返回值

### 📝 日志记录
- **JSON Lines格式**: 每条记录一行，便于解析和分析
- **自动分类**: 按日期自动分割日志文件
- **结构化数据**: 完整的调用链路信息，便于分析
- **错误记录**: 详细记录异常和超时情况

### 📈 统计分析
- **实时统计**: 查看今日调用次数、Token消耗、响应时间等
- **模型对比**: 不同模型的使用频率和性能对比
- **工具使用统计**: 各种工具的调用频率分析
- **错误率监控**: 失败调用的统计和分析

## 📁 文件结构

```
logs/                           # 日志目录
├── model_calls_20241225.jsonl  # 按日期命名的调用日志
├── model_calls_20241226.jsonl
└── ...

model_logger.py                 # 核心日志记录模块
log_viewer.py                   # 命令行日志查看工具
logging_config.yaml            # 日志配置文件
LOG_MONITOR_README.md          # 本说明文档
```

## 🛠️ 使用方法

### 1. 在应用中查看统计

运行桌面应用后，点击左侧设置面板中的 **"📊 查看调用日志"** 按钮：

- 查看今日调用统计
- 打开日志文件夹
- 实时刷新数据

### 2. 命令行查看工具

#### 基本统计
```bash
# 查看今日统计摘要
python log_viewer.py --summary

# 查看指定日期的统计
python log_viewer.py --summary --date 20241225
```

#### 详细日志
```bash
# 查看最近10条详细记录
python log_viewer.py --detailed

# 查看最近50条记录
python log_viewer.py --detailed --limit 50

# 查看指定日期的详细记录
python log_viewer.py --detailed --date 20241225
```

#### 导出数据
```bash
# 导出为CSV文件
python log_viewer.py --export-csv

# 导出指定日期的数据
python log_viewer.py --export-csv --date 20241225 --output my_export.csv
```

### 3. 日志格式说明

每条日志记录包含以下字段：

```json
{
  "timestamp": "2024-12-25T14:30:15.123456",
  "session_id": "abc12345",
  "thread_id": "thread_def67890",
  "model_name": "gpt-4o",
  "model_provider": "openai",
  "input_messages": [{"role": "user", "content": "用户输入"}],
  "input_tokens": 15,
  "output_content": "模型回复内容",
  "output_tokens": 42,
  "tool_calls": [
    {
      "id": "call_123",
      "name": "search_tool",
      "args": {"query": "搜索内容"},
      "timestamp": "2024-12-25T14:30:16.123456"
    }
  ],
  "tool_responses": [
    {
      "tool_call_id": "call_123",
      "content": "搜索结果",
      "name": "search_tool",
      "timestamp": "2024-12-25T14:30:17.123456"
    }
  ],
  "processing_time": 2.34,
  "error": null,
  "metadata": {"langgraph_node": "agent"}
}
```

## 📊 统计示例

运行统计查看后的输出示例：

```
📊 今日模型调用统计
════════════════════
🔢 总调用次数: 25
📝 总输入Token: 1,245
📋 总输出Token: 3,678
🎯 总Token数: 4,923
⏱️ 总耗时: 45.67秒
⚠️ 错误次数: 1

🤖 使用的模型:
  • openai/gpt-4o: 15次
  • anthropic/claude-3-5-sonnet-latest: 10次

🛠️ 使用的工具:
  • search_tool
  • calculator
  • file_reader
```

## 🔧 高级配置

### 自定义日志目录
在应用启动时可以指定自定义的日志目录：

```python
from model_logger import init_model_logging
logger = init_model_logging("custom_logs_dir")
```

### 日志轮转配置
修改 `logging_config.yaml` 文件来配置日志轮转：

```yaml
handlers:
  model_calls_file:
    class: logging.handlers.RotatingFileHandler
    maxBytes: 10485760  # 10MB
    backupCount: 5      # 保留5个备份文件
```

## 💡 使用建议

1. **定期清理**: 日志文件会随时间增长，建议定期清理旧文件
2. **监控存储**: 注意日志文件占用的磁盘空间
3. **分析调用**: 通过统计数据优化模型选择和应用性能
4. **错误追踪**: 关注错误日志，及时发现和解决问题
5. **Token控制**: 通过Token统计控制API使用成本

## 🐛 故障排除

### 常见问题

1. **日志文件无法创建**
   - 检查logs目录的写入权限
   - 确保有足够的磁盘空间

2. **统计显示为空**
   - 确认已进行过模型调用
   - 检查日志文件是否存在且格式正确

3. **导出CSV失败**
   - 检查输出目录的写入权限
   - 确认CSV文件名不与现有文件冲突

### 调试模式

启用详细日志输出：

```python
import logging
logging.getLogger('model_calls').setLevel(logging.DEBUG)
```

## 📞 技术支持

如果遇到问题或需要帮助，请：

1. 检查控制台输出的错误信息
2. 查看error.log文件中的详细错误日志
3. 确认所有依赖项已正确安装
4. 验证配置文件格式正确

---

**注意**: 此监控系统会记录所有的输入和输出内容，请确保符合您的数据安全和隐私要求。
