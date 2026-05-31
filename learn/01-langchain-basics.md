# LangChain 基础概念与实践

## 学习目标
完成本模块后，你将能够：
1. 理解 LangChain 的核心组件和架构
2. 使用 LangChain 调用不同的 LLM 模型
3. 创建和使用 Prompt 模板
4. 构建简单的 Chain 链式调用
5. 理解 LangChain 的消息类型系统

---

## 1. LangChain 是什么？

LangChain 是一个用于构建 LLM（大语言模型）应用的框架。它提供了一套标准化的接口和组件，让你能够：

- **统一访问不同 LLM**：通过一致的 API 调用 OpenAI、Anthropic、Google 等不同模型
- **组合复杂工作流**：将多个 LLM 调用、工具调用、数据处理链接在一起
- **管理对话状态**：处理多轮对话的历史记录
- **集成外部数据**：通过 RAG（检索增强生成）让 LLM 访问外部知识

### 核心架构

```
┌─────────────────────────────────────────────────────────┐
│                    LangChain 应用层                       │
├─────────────────────────────────────────────────────────┤
│  Agents  │   Chains   │   Memory   │   Retrieval        │
├─────────────────────────────────────────────────────────┤
│               LangChain Core（核心抽象层）                 │
├─────────────────────────────────────────────────────────┤
│  OpenAI  │  Anthropic │  Google    │  其他 LLM 提供商    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 安装 LangChain

### 基础安装
```bash
# 安装核心包
pip install langchain-core

# 安装主包（包含经典 API）
pip install langchain

# 安装特定 LLM 集成
pip install langchain-openai      # OpenAI
pip install langchain-anthropic   # Anthropic (Claude)
pip install langchain-google-genai # Google
```

### 在本项目中开发
```bash
# 本项目使用 uv 管理依赖
uv sync --all-groups

# 运行测试
make test
```

---

## 3. 消息系统（Messages）

LangChain 的消息系统是所有交互的基础。理解消息类型是学习 LangChain 的第一步。

### 消息类型

```python
from langchain_core.messages import (
    SystemMessage,    # 系统消息：设定 AI 的角色和行为
    HumanMessage,     # 用户消息：用户的输入
    AIMessage,        # AI 消息：模型的回复
    ToolMessage,      # 工具消息：工具调用的结果
    FunctionMessage,  # 函数消息（已弃用，使用 ToolMessage）
)
```

### 消息结构

每条消息包含：
- **content**：消息内容（字符串或内容块列表）
- **role**：消息角色（system/human/ai/tool）
- **additional_kwargs**：额外参数（如工具调用信息）

### 示例：创建消息

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 系统消息：设定 AI 角色
system_msg = SystemMessage(content="你是一个有帮助的 Python 编程助手。")

# 用户消息：用户的问题
human_msg = HumanMessage(content="如何在 Python 中读取文件？")

# AI 消息：模型的回复
ai_msg = AIMessage(content="你可以使用 open() 函数来读取文件...")

# 消息可以组合成列表
messages = [system_msg, human_msg, ai_msg]

# 打印消息
for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

---

## 4. 语言模型（Language Models）

LangChain 提供了统一的接口来调用不同的 LLM。

### 4.1 LLM（基础语言模型）

适用于简单的文本补全任务：

```python
from langchain_openai import OpenAI

# 初始化模型
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,  # 控制随机性：0=确定，1=随机
    api_key="your-api-key"
)

# 调用模型
response = llm.invoke("Python 中如何定义一个类？")
print(response)
```

### 4.2 ChatModel（聊天模型）

适用于对话场景，支持多轮对话：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 初始化聊天模型
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key="your-api-key"
)

# 构建消息
messages = [
    SystemMessage(content="你是一个 Python 专家。"),
    HumanMessage(content="解释什么是装饰器？"),
]

# 调用模型
response = chat_model.invoke(messages)
print(response.content)
```

### 4.3 模型参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| model | 模型名称 | "gpt-4o", "claude-3-opus" |
| temperature | 随机性（0-2） | 0.7（平衡） |
| max_tokens | 最大输出 token 数 | 1024 |
| top_p | 核采样参数 | 1.0 |
| frequency_penalty | 频率惩罚 | 0.0 |
| presence_penalty | 存在惩罚 | 0.0 |

---

## 5. 提示词模板（Prompt Templates）

Prompt 模板让你可以动态生成提示词，避免硬编码。

### 5.1 基础 Prompt 模板

```python
from langchain_core.prompts import PromptTemplate

# 创建模板
template = PromptTemplate(
    input_variables=["language", "task"],
    template="用 {language} 编写代码来 {task}。只返回代码，不要解释。"
)

# 格式化模板
prompt = template.format(language="Python", task="读取 CSV 文件")
print(prompt)
# 输出：用 Python 编写代码来读取 CSV 文件。只返回代码，不要解释。
```

### 5.2 Chat Prompt 模板

用于聊天模型，支持多消息组合：

```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 方式 1：使用元组
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role} 专家。"),
    ("human", "请解释 {concept}。"),
])

# 格式化
messages = chat_prompt.format_messages(
    role="Python",
    concept="生成器（Generator）"
)
print(messages)

# 方式 2：使用模板对象
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个 {role} 专家。"),
    HumanMessagePromptTemplate.from_template("请解释 {concept}。"),
])
```

### 5.3 Few-Shot Prompt 模板

提供示例来引导模型输出格式：

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "big", "output": "small"},
    {"input": "fast", "output": "slow"},
]

# 示例格式
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入: {input}\n输出: {output}"
)

# Few-Shot 模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个词的反义词：",
    suffix="输入: {input}\n输出:",
    input_variables=["input"],
)

# 使用
print(few_shot_prompt.format(input="hot"))
```

---

## 6. 链式调用（Chains）

Chain 是 LangChain 的核心概念，用于将多个步骤链接在一起。

### 6.1 LLMChain（基础链）

最简单的链：Prompt → LLM → 输出

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建组件
prompt = ChatPromptTemplate.from_template("用一句话解释 {topic}。")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()  # 将输出解析为字符串

# 构建链（使用 LCEL - LangChain Expression Language）
chain = prompt | model | parser

# 调用链
result = chain.invoke({"topic": "区块链"})
print(result)
```

### 6.2 链的组合

链可以嵌套和组合：

```python
# 链式组合：输出作为下一个链的输入
prompt1 = ChatPromptTemplate.from_template("生成关于 {topic} 的笑话。")
prompt2 = ChatPromptTemplate.from_template("将以下文本翻译成英文：{joke}")

chain1 = prompt1 | model | StrOutputParser()
chain2 = prompt2 | model | StrOutputParser()

# 组合链
combined_chain = {"joke": chain1} | prompt2 | model | StrOutputParser()

result = combined_chain.invoke({"topic": "程序员"})
print(result)
```

### 6.3 Runnable 接口

LangChain 使用 `Runnable` 接口统一所有组件：

```python
from langchain_core.runnables import RunnableLambda

# 自定义函数可以作为链的一部分
def count_words(text: str) -> str:
    words = text.split()
    return f"文本包含 {len(words)} 个单词"

# 使用 RunnableLambda 包装函数
word_counter = RunnableLambda(count_words)

# 组合到链中
chain = prompt | model | StrOutputParser() | word_counter

result = chain.invoke({"topic": "Python"})
print(result)
```

---

## 7. 输出解析器（Output Parsers）

将 LLM 的结构化输出转换为 Python 对象。

### 7.1 字符串解析器

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.invoke(aimessage)  # 提取字符串内容
```

### 7.2 JSON 解析器

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

# 定义输出结构
class Joke(BaseModel):
    setup: str
    punchline: str

# 创建解析器
parser = JsonOutputParser(pydantic_object=Joke)

# 获取格式指令
format_instructions = parser.get_format_instructions()
print(format_instructions)
```

### 7.3 Pydantic 解析器

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class CodeSolution(BaseModel):
    language: str = Field(description="编程语言")
    code: str = Field(description="代码内容")
    explanation: str = Field(description="代码解释")

parser = PydanticOutputParser(pydantic_object=CodeSolution)

prompt = ChatPromptTemplate.from_template(
    "解决以下编程问题：{problem}\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
result = chain.invoke({"problem": "反转字符串"})
print(result.code)
```

---

## 8. 实战练习

### 练习 1：创建问答链

创建一个链，接收用户问题，返回简洁答案：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 你的代码
qa_chain = (
    ChatPromptTemplate.from_template(
        "简洁回答以下问题（不超过 50 字）：{question}"
    )
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# 测试
questions = [
    "什么是机器学习？",
    "Python 和 JavaScript 有什么区别？",
    "解释 REST API。",
]

for q in questions:
    answer = qa_chain.invoke({"question": q})
    print(f"Q: {q}")
    print(f"A: {answer}\n")
```

### 练习 2：构建翻译链

创建一个链，将文本翻译成指定语言：

```python
# 你的代码
translation_chain = (
    ChatPromptTemplate.from_template(
        "将以下文本翻译成 {language}：\n{text}\n只返回翻译结果。"
    )
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# 测试
text = "Hello, how are you today?"
languages = ["中文", "法语", "日语"]

for lang in languages:
    result = translation_chain.invoke({"language": lang, "text": text})
    print(f"{lang}: {result}")
```

### 练习 3：Few-Shot 分类器

创建一个情感分类器，使用 Few-Shot 示例：

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# 定义示例
examples = [
    {"input": "这部电影太棒了！", "output": "正面"},
    {"input": "服务很差，不会再来了。", "output": "负面"},
    {"input": "一般般，没什么特别的。", "output": "中性"},
]

# 创建 Few-Shot 模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 完整 Prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析专家。将文本分类为：正面、负面或中性。"),
    few_shot_prompt,
    ("human", "{input}"),
])

# 构建链
chain = final_prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# 测试
texts = [
    "产品质量很好，非常满意！",
    "快递太慢了，等了一周。",
    "价格合理，质量还可以。",
]

for text in texts:
    result = chain.invoke({"input": text})
    print(f"文本: {text}")
    print(f"情感: {result}\n")
```

---

## 9. 下一步学习

完成本模块后，你已经掌握了 LangChain 的基础。接下来可以学习：

1. **Agents（智能体）** - 让 LLM 自主决策和调用工具
2. **Memory（记忆）** - 管理对话历史和状态
3. **Retrieval（检索）** - 构建 RAG 应用
4. **Tools（工具）** - 集成外部 API 和函数
5. **Callbacks（回调）** - 监控和日志记录

---

## 10. 常见问题

### Q: LangChain 和直接调用 OpenAI API 有什么区别？
A: LangChain 提供了：
- 统一的接口（切换模型只需改一行代码）
- 链式调用（组合多个步骤）
- 内置的 Prompt 管理
- 对话记忆管理
- 工具集成框架

### Q: temperature 参数如何设置？
A:
- **0.0-0.3**：适合事实性问答、代码生成（确定性高）
- **0.5-0.8**：适合通用对话、内容创作（平衡）
- **1.0-2.0**：适合创意写作、头脑风暴（随机性高）

### Q: 如何处理长文本？
A:
- 使用支持更长上下文的模型（如 gpt-4o、claude-3）
- 使用文本分割器（Text Splitter）
- 使用摘要链（Summarization Chain）

---

## 参考资料

- [LangChain 官方文档](https://docs.langchain.com/)
- [LangChain Python API 参考](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
