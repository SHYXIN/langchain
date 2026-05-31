# LangChain Agents（智能体）

## 学习目标
完成本模块后，你将能够：
1. 理解 Agent 的概念和工作原理
2. 创建和使用 Tool（工具）
3. 构建能自主决策的 ReAct Agent
4. 理解 Agent 的执行循环

---

## 1. 什么是 Agent？

**Agent（智能体）** 是一个能够**自主决策**和**调用工具**来完成任务的系统。

```
普通 Chain（固定流程）:
  输入 → 步骤1 → 步骤2 → 步骤3 → 输出
  （每一步都是预先定义好的）

Agent（自主决策）:
  输入 → [思考 → 行动 → 观察] × N → 输出
  （每一步由 LLM 自主决定）
```

### 核心区别

| 特性 | Chain（链） | Agent（智能体） |
|------|-------------|-----------------|
| 流程 | 固定 | 动态 |
| 决策 | 预先定义 | LLM 自主决定 |
| 工具调用 | 不支持 | 支持 |
| 适用场景 | 简单、可预测的任务 | 复杂、需要判断的任务 |

### Agent 的工作原理（ReAct 模式）

```
┌─────────────────────────────────────────────────────────┐
│                    Agent 执行循环                         │
│                                                         │
│  用户输入 ──→ [思考 Thought]                             │
│                  │                                      │
│                  ▼                                      │
│            [决定行动 Action]                             │
│                  │                                      │
│                  ▼                                      │
│            [执行工具 Tool]                               │
│                  │                                      │
│                  ▼                                      │
│            [观察结果 Observation]                        │
│                  │                                      │
│                  ▼                                      │
│            是否完成？ ──否──→ [继续思考]                  │
│                  │                                      │
│                 是                                      │
│                  ▼                                      │
│             输出最终答案                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Tools（工具）

Tool 是 Agent 可以调用的函数。每个工具需要：
- **name**：工具名称
- **description**：工具描述（LLM 根据这个来决定是否使用）
- **func**：实际的执行函数

### 2.1 使用 @tool 装饰器创建工具

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除和括号。"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    # 这里可以调用真实的天气 API
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
    }
    return weather_data.get(city, f"未找到 {city} 的天气信息")

# 查看工具信息
print(calculator.name)        # calculator
print(calculator.description)  # 计算数学表达式...
print(calculator.args)         # 参数信息
```

### 2.2 使用 StructuredTool 创建复杂工具

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(description="最大结果数", default=5)

def search_web(query: str, max_results: int = 5) -> str:
    """搜索网页（示例函数）"""
    return f"搜索 '{query}' 的前 {max_results} 条结果: ..."

tool = StructuredTool.from_function(
    func=search_web,
    name="web_search",
    description="搜索网页获取最新信息",
    args_schema=SearchInput,
)
```

---

## 3. 创建 Agent

### 3.1 使用 create_tool_caching_agent（推荐）

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, get_current_time]

# 2. 创建模型
llm = ChatOpenAI(model="gpt-4o")

# 3. 创建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，可以使用工具来回答问题。"),
    ("human", "{input}"),
    ("agent_scratchpad", "{agent_scratchpad}"),  # Agent 的思考过程
])

# 4. 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. 创建 AgentExecutor（执行器）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示思考过程
)

# 6. 执行
result = agent_executor.invoke({"input": "现在几点了？"})
print(result["output"])
```

### 3.2 ReAct Agent（经典方式）

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 使用 LangChain Hub 上的 ReAct Prompt
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,  # 最大迭代次数
)
```

---

## 4. Agent 执行过程详解

当 Agent 执行时，会经历以下步骤：

```
用户: "计算 123 + 456 的结果"

第 1 轮:
  Thought: 我需要使用计算器工具来计算
  Action: calculator
  Action Input: "123 + 456"
  Observation: 579

第 2 轮:
  Thought: 我已经得到了计算结果，可以回答用户了
  Final Answer: 123 + 456 = 579
```

### verbose=True 时的输出

```
> Entering new AgentExecutor chain...
  我需要计算 123 + 456
  Action: calculator
  Action Input: 123 + 456
  Observation: 579
  我现在知道了答案
  Final Answer: 123 + 456 的结果是 579

> Finished chain.
```

---

## 5. 自定义 Agent Prompt

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 自定义系统提示
system_template = """你是一个专业的编程助手。

你可以使用以下工具来回答问题：

{tools}

工具名称: {tool_names}

请按照以下格式回答：
- 如果需要调用工具，使用 Action: 工具名称 格式
- 如果已经得到答案，使用 Final Answer: 答案 格式
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
```

---

## 6. 实战练习

### 练习 1：创建数学计算器 Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 你的代码
@tool
def add(a: float, b: float) -> float:
    """计算两个数的和。"""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """计算两个数的积。"""
    return a * b

@tool
def power(base: float, exponent: float) -> float:
    """计算 base 的 exponent 次方。"""
    return base ** exponent

# 创建 Agent 并测试
tools = [add, multiply, power]
# ...
```

### 练习 2：天气查询 Agent

```python
@tool
def get_temperature(city: str) -> str:
    """获取城市的当前温度。"""
    temperatures = {"北京": 25, "上海": 28, "广州": 30}
    temp = temperatures.get(city, 20)
    return f"{city} 当前温度: {temp}°C"

@tool
def get_forecast(city: str, days: int = 3) -> str:
    """获取城市的天气预报。"""
    return f"{city} 未来 {days} 天: 晴, 多云, 小雨"

# 创建 Agent 并测试
```

---

## 7. 进阶：多工具协作

Agent 可以在一次执行中调用多个工具：

```
用户: "北京和上海的温度差是多少？"

执行过程:
  1. 调用 get_temperature("北京") → 25°C
  2. 调用 get_temperature("上海") → 28°C
  3. 计算差值: 28 - 25 = 3
  4. 返回: "北京和上海的温度差是 3°C"
```

---

## 8. 常见问题

### Q: Agent 和 Chain 有什么区别？
A: Chain 是固定流程，Agent 是动态决策。如果你知道每一步该用什么，用 Chain；如果需要 LLM 自己判断，用 Agent。

### Q: Agent 会无限循环吗？
A: 不会。AgentExecutor 有 `max_iterations` 参数（默认 15），超过后会停止。

### Q: 如何提高 Agent 的准确性？
A:
- 工具描述要清晰准确
- 系统提示要明确
- 使用更强大的模型
- 限制 `max_iterations` 避免无效循环

---

## 下一步学习

1. **Memory（记忆）** — 让 Agent 记住对话历史
2. **RAG（检索增强）** — 让 Agent 访问外部知识
3. **LangGraph** — 构建复杂的多 Agent 工作流
