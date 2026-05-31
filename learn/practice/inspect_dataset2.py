# -*- coding: utf-8 -*-
import json
from datasets import load_dataset

ds = load_dataset("luozhouyang/question-answering-datasets", split="train")

# 搜索客服/服务相关的样本
keywords = ["客服", "退款", "订单", "售后", "投诉", "换货", "维修", "咨询", "快递", "物流", "售后", "服务"]
print("=== 搜索客服/服务相关样本 ===")
found = 0
for i, item in enumerate(ds):
    q = item["question"]
    ctx = item["context"]
    if any(kw in q or kw in ctx for kw in keywords):
        found += 1
        if found <= 10:
            print(f"\n--- 样本 {i} ---")
            print(f"  问题: {q}")
            print(f"  答案: {item['answer']}")
            ctx_preview = ctx[:200].replace('\n', ' ')
            print(f"  上下文(前200): {ctx_preview}...")

print(f"\n共找到 {found} 条客服/服务相关样本（总计 {len(ds)} 条）")

# 看看问题类型分布
print("\n=== 问题类型抽样（按首词分类）{} ===")
from collections import Counter
prefixes = Counter()
for item in ds:
    q = item["question"]
    prefix = q[:2] if len(q) >= 2 else q
    prefixes[prefix] += 1

print("最常见的问题前缀:")
for prefix, count in prefixes.most_common(20):
    # 找该前缀的示例
    for item in ds:
        if item["question"].startswith(prefix):
            print(f"  '{prefix}...' ({count}条) 示例: {item['question']}")
            break
