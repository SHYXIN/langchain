# LangGraph 进阶

## 学习目标
完成本模块后，你将能够：
1. 使用 MemorySaver 实现对话持久化和时间旅行
2. 用 interrupt/Command 实现真正的人工介入
3. 用 task/entrypoint 构建函数式工作流
4. 实现多 Agent 协作模式
5. 掌握子图（Subgraph）和状态嵌套

---

## 1. MemorySaver — 状态持久化

LangGraph 默认不保存状态。`MemorySaver` 让图在每次执行后**自动保存快照**，支持：
- **对话恢复**：从中断的地方继续
- **时间旅行**：回退到任意历史状态重新执行
- **断点续传**：长任务中断后可以继续

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点存储器
checkpointer = MemorySaver()

# 编译时传入
app = graph.compile(checkpointer=checkpointer)

# 运行时传入 config（同一 thread_id 共享状态）
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [HumanMessage("你好")]}, config)
# 第二次调用会自动加载之前的状态
result = app.invoke({"messages": [HumanMessage("我叫什么名字？")]}, config)
# AI 会记得之前对话中的名字！
```

---

## 2. 真正的人工介入（Interrupt）

模块 5 中的 `human_review` 是模拟的。真正的 `interrupt` 会**暂停图的执行**，等待外部输入：

```python
from langgraph.types import interrupt, Command

def review_node(state):
    # 暂停执行，向外部发送数据
    answer = interrupt({
        "question": "是否批准这篇文章？",
        "content": state["content"]
    })
    # answer 是外部通过 Command(resume=...) 传回的值
    return {"approved": answer["approved"]}

# 运行时会在这里暂停
try:
    result = app.invoke({"content": "..."}, config)
except GraphInterrupt:
    # 图已暂停，等待人工输入
    pass

# 人工审核后，恢复执行
result = app.invoke(Command(resume={"approved": True}), config)
```

---

## 3. task / entrypoint — 函数式工作流

LangGraph 提供了更简洁的**函数式 API**，不需要手动定义 StateGraph：

```python
from langgraph.func import task, entrypoint
from langgraph.checkpoint.memory import MemorySaver

@task
def fetch_data(url: str) -> dict:
    """获取数据"""
    return {"data": f"数据来自 {url}"}

@task
def process_data(data: dict) -> str:
    """处理数据"""
    return f"处理结果: {data['data']}"

@entrypoint(checkpointer=MemorySaver())
def my_workflow(inputs: dict) -> str:
    # 像写普通函数一样编排工作流
    raw = await fetch_data(inputs["url"])
    result = await process_data(raw)
    return result

# 使用
result = my_workflow.invoke({"url": "https://example.com"})
```

---

## 4. 多 Agent 协作

LangGraph 支持多种多 Agent 模式：

### 4.1 Supervisor 模式（管理者模式）

一个"监督者" Agent 决定由哪个专家 Agent 处理任务：

```
用户 → Supervisor → Agent A（搜索）
                   → Agent B（计算）
                   → Agent C（写作）
```

```python
# 创建专家 Agent
search_agent = create_agent(llm, [search_tool])
calc_agent = create_agent(llm, [calc_tool])

# Supervisor 决定调用哪个 Agent
def supervisor(state):
    response = llm.invoke(state["messages"])
    if response.tool_calls:
        return {"next": response.tool_calls[0]["name"]}
    return {"next": END}

graph.add_conditional_edges("supervisor", supervisor, {
    "search": "search_agent",
    "calc": "calc_agent",
    END: END,
})
```

### 4.2 子图模式（Subgraph）

将一个 Agent 作为另一个 Agent 的子图嵌入：

```python
# 子图
subgraph = StateGraph(SubState)
subgraph.add_node("process", process_node)
subgraph.add_edge(START, "process")
subgraph.add_edge("process", END)
sub_app = subgraph.compile()

# 主图
main_graph = StateGraph(MainState)
main_graph.add_node("sub_agent", sub_app)  # 直接嵌入子图
main_graph.add_edge(START, "sub_agent")
main_graph.add_edge("sub_agent", END)
```

---

## 5. 时间旅行（Time Travel）

利用 MemorySaver 的历史快照，可以**回到过去重新执行**：

```python
# 获取所有历史状态
history = list(app.get_state_history(config))
print(f"共 {len(history)} 个快照")

# 回到第 2 个快照的状态
checkpoint = history[2]
result = app.invoke(None, checkpoint.config)

# 或者修改历史状态后重新执行
result = app.invoke(
    {"messages": [HumanMessage("换个问题")]},
    checkpoint.config
)
```

---

## 6. 流式输出（Streaming）

LangGraph 支持多种流式输出模式：

```python
# 流式输出每个节点的结果
for chunk in app.stream({"messages": [HumanMessage("你好")]}, config):
    print(chunk)

# 流式输出 token（需要 LLM 支持 streaming）
for chunk in app.stream({"messages": [HumanMessage("你好")]}, 
                         config, stream_mode="messages"):
    print(chunk.content, end="", flush=True)
```

---

## 7. 总结

| 进阶概念 | API | 用途 |
|----------|-----|------|
| 状态持久化 | `MemorySaver` | 对话恢复、时间旅行 |
| 人工介入 | `interrupt` / `Command` | 审核、确认、分支决策 |
| 函数式 API | `task` / `entrypoint` | 更简洁的工作流定义 |
| 多 Agent | Supervisor / Subgraph | 复杂任务分解 |
| 时间旅行 | `get_state_history` | 回退、调试 |
| 流式输出 | `stream()` | 实时反馈 |

---

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph 多 Agent 教程](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
- [LangGraph 时间旅行](https://langchain-ai.github.io/langgraph/concepts/time-travel/)
