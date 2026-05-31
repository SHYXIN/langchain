# LangChain 学习实践

本目录包含 LangChain 学习的实践代码。

## 环境设置

```bash
# 进入实践目录
cd learn/practice

# 创建虚拟环境
uv venv

# 安装依赖
uv add langchain openai python-dotenv
```

## 快速开始

1. 复制 `.env.example` 为 `.env`
2. 在 `.env` 中填入你的 API 密钥
3. 运行示例代码

## 学习路径

1. `01-hello-langchain.py` - 第一个 LangChain 程序
2. `02-prompt-templates.py` - 提示词模板
3. `03-chains.py` - 链式调用
4. `04-agents.py` - 智能代理
5. `05-memory.py` - 记忆管理
6. `06-rag.py` - 检索增强生成

## 注意事项

- 运行代码需要有效的 API 密钥
- 建议从 OpenAI 或 Anthropic 获取 API 密钥
- 代码会产生 API 调用费用，请注意使用量
