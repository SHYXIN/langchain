#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一个 LangChain 程序
学习目标：了解 LangChain 的基本结构和模型调用
"""

import os
import sys
import io

# 修复 Windows 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 核心概念 1: Language Model (语言模型)
# ============================================================
# LangChain 提供了统一的接口来调用不同的 LLM 模型
# 目前支持 OpenAI、Anthropic、Google 等多种模型

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 示例 1: 使用 OpenAI GPT-4o-mini（兼容 OpenAI API 格式的第三方服务）
def demo_openai():
    """演示如何使用 OpenAI 模型"""
    print("=" * 60)
    print("示例 1: OpenAI GPT-4o-mini")
    print("=" * 60)

    # 从环境变量读取配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # 兼容第三方 API 服务

    if not api_key:
        print("⚠️  未设置 OPENAI_API_KEY 环境变量，跳过 OpenAI 演示")
        return

    # 创建模型实例
    # base_url: 可自定义 API 端点（如 LongCat、DeepSeek 等兼容 OpenAI 格式的服务）
    # LongCat 支持的模型名称
    model_name = os.getenv("OPENAI_MODEL", "LongCat-2.0-Preview")
    llm_kwargs = {
        "model": model_name,
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
        print(f"使用自定义 API 端点: {base_url}")

    llm = ChatOpenAI(**llm_kwargs)

    # 调用模型
    response = llm.invoke("请用一句话介绍 LangChain")
    print(f"模型回复: {response.content}")
    print(f"模型名称: {response.response_metadata.get('model_name', 'unknown')}")
    print()


# 示例 2: 使用 Anthropic Claude
def demo_anthropic():
    """演示如何使用 Anthropic Claude 模型"""
    print("=" * 60)
    print("示例 2: Anthropic Claude 3.5 Sonnet")
    print("=" * 60)

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    response = llm.invoke("请用一句话介绍 LangChain")
    print(f"模型回复: {response.content}")
    print()


# ============================================================
# 核心概念 2: Messages (消息)
# ============================================================
# LangChain 使用消息对象来表示对话中的不同角色
# - SystemMessage: 系统指令，定义 AI 的行为
# - HumanMessage: 用户输入
# - AIMessage: AI 的回复
# - ToolMessage: 工具的返回结果

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def demo_messages():
    """演示消息类型"""
    print("=" * 60)
    print("示例 3: 消息类型")
    print("=" * 60)

    messages = [
        SystemMessage(content="你是一个专业的 Python 编程助手"),
        HumanMessage(content="如何在 Python 中读取文件？"),
        AIMessage(content="使用 open() 函数..."),
    ]

    for msg in messages:
        print(f"类型: {type(msg).__name__}")
        print(f"内容: {msg.content[:50]}...")
        print()


# ============================================================
# 核心概念 3: Prompt Templates (提示词模板)
# ============================================================
# 提示词模板允许你创建可重用的提示词结构
# 支持变量替换，便于动态生成提示词

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

def demo_prompt_templates():
    """演示提示词模板"""
    print("=" * 60)
    print("示例 4: 提示词模板")
    print("=" * 60)

    # 简单模板
    template = PromptTemplate.from_template(
        "请解释 {concept} 的概念，用 {language} 回答"
    )
    prompt = template.format(concept="机器学习", language="中文")
    print(f"简单模板:\n{prompt}\n")

    # 聊天模板（推荐）
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("human", "{question}"),
    ])
    messages = chat_template.format_messages(
        role="Python 专家",
        question="什么是装饰器？"
    )
    print(f"聊天模板:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    print()


# ============================================================
# 核心概念 4: Chains (链)
# ============================================================
# 链将多个步骤组合在一起，前一步的输出是后一步的输入
# LangChain 使用 LCEL (LangChain Expression Language) 构建链

from langchain_core.output_parsers import StrOutputParser

def demo_chains():
    """演示链式调用"""
    print("=" * 60)
    print("示例 5: 链式调用 (LCEL)")
    print("=" * 60)

    # 构建一个简单的链: 提示词 -> 模型 -> 输出解析
    prompt = ChatPromptTemplate.from_template(
        "请用一句话解释 {topic}"
    )

    # 注意：这里不实际调用 API，只展示链的结构
    chain = prompt | StrOutputParser()

    print("链的结构:")
    print(f"  1. 提示词模板: {prompt}")
    print(f"  2. 输出解析器: StrOutputParser()")
    print(f"  3. 完整链: {chain}")
    print()

    # 链的执行方式:
    # result = chain.invoke({"topic": "量子计算"})
    # 这等价于:
    # formatted_prompt = prompt.invoke({"topic": "量子计算"})
    # model_output = llm.invoke(formatted_prompt)
    # result = output_parser.invoke(model_output)


# ============================================================
# 核心概念 5: Output Parsers (输出解析器)
# ============================================================
# 输出解析器将 LLM 的文本输出转换为结构化数据

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class ProgrammingLanguage(BaseModel):
    """编程语言信息"""
    name: str = Field(description="语言名称")
    year: int = Field(description="首次发布年份")
    creator: str = Field(description="创始人")

def demo_output_parsers():
    """演示输出解析器"""
    print("=" * 60)
    print("示例 6: 输出解析器")
    print("=" * 60)

    parser = JsonOutputParser(pydantic_object=ProgrammingLanguage)

    # 示例 LLM 输出
    llm_output = '{"name": "Python", "year": 1991, "creator": "Guido van Rossum"}'

    # 解析输出
    result = parser.parse(llm_output)
    print(f"解析结果: {result}")
    print(f"类型: {type(result)}")
    print()


def main():
    """主函数 - 运行所有演示"""
    print("\n" + "=" * 60)
    print("  欢迎使用 LangChain 学习教程！")
    print("=" * 60 + "\n")

    # 运行演示
    demo_openai()           # 模型调用
    demo_messages()         # 消息类型
    demo_prompt_templates() # 提示词模板
    demo_chains()           # 链式调用
    demo_output_parsers()   # 输出解析器

    print("=" * 60)
    print("  学习完成！")
    print("=" * 60)
    print("\n下一步: 运行 02-advanced-concepts.py 学习更多高级概念")


if __name__ == "__main__":
    main()
