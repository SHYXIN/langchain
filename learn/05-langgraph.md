# 模块 5: LangGraph 工作流

## 学习目标
完成本模块后，你将能够：
1. 理解 LangGraph 的核心概念（节点、边、状态）
2. 构建简单的状态图工作流
3. 构建带条件分支的工作流
4. 构建 Agent 工作流

---

## 1. 什么是 LangGraph？

**LangGraph** 是 LangChain 的工作流编排框架，用于构建有状态的、有向图结构的应用。

```
传统 Chain（线性）:
  A → B → C → D

LangGraph（图结构）:
  ┌──→ B ──→ D ──→ F
  A ──→ C ──→ E ──→ G
       ↑         │
       └─────────┘
```

### 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| **State（状态）** | 在节点之间传递的数据 | 传送带上的包裹 |
| **Node（节点）** | 执行某个操作的函数 | 工厂里的机器 |
| **Edge（边） | 连接节点的路径 | 传送带 |
| **条件边** | 根据状态决定走哪条路径 | 分拣器 |

---

## 2. 最简单的 StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. 定义状态
class MyState(TypedDict):
    message: str
    count: int

# 2. 定义节点函数
def node_a(state: MyState) -> MyState:
    return {"message": state["message"] + " → A", "count": state["count"] + 1}

def node_b(state: MyState) -> MyState:
    return {"message": state["message"] + " → B", "count": state["count"] + 1}

# 3. 构建图
graph = StateGraph(MyState)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"message": "开始", "count": 0})
# {"message": "开始 → A → B", "count": 2}
```

---

## 3. 条件分支

```python
def should_continue(state: MyState) -> str:
    if state["count"] < 3:
        return "continue"
    return "end"

graph.add_conditional_edges(
    "a",
    should_continue,
    {
        "continue": "b",
        "end": END,
    }
)
```

---

## 4. MessagesState — 带对话历史的状态

```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage

# MessagesState 自带 messages 字段
class ChatState(MessagesState):
    user_name: str  # 可以添加自定义字段

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

---

## 5. 构建 Agent 工作流

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"

tools = [search]

# Agent 节点
def agent_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 条件边：是否需要调用工具
def should_use_tools(state):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()
```

---

## 6. 总结

```
LangChain 生态:
┌─────────────────────────────────────────────────┐
│  langchain-core    → 基础抽象（Messages, Tools）  │
│  langchain         → 高级 API（Chains, Agents）   │
│  langgraph         → 工作流编排（StateGraph）     │
│  langchain-openai  → OpenAI 集成                 │
└─────────────────────────────────────────────────┘

选择指南:
  - 简单调用 → ChatModel
  - 固定流程 → Chain（LCEL）
  - 需要决策 → Agent（create_agent）
  - 复杂流程 → LangGraph（StateGraph）
```

---

## 参考资料

- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph API 参考](https://docs.langchain.com/oss/python/langgraph/reference)
