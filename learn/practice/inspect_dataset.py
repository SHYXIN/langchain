#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datasets import load_dataset

print("Loading dataset: luozhouyang/question-answering-datasets ...")
ds = load_dataset("luozhouyang/question-answering-datasets", split="train")

print(f"\n=== 基本信息 ===")
print(f"总样本数: {len(ds)}")
print(f"字段: {ds.column_names}")
print(f"特征: {ds.features}")

print(f"\n=== 前5条样本 ===")
for i in range(min(5, len(ds))):
    item = ds[i]
    print(f"\n--- 样本 {i+1} ---")
    for k, v in item.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        print(f"  {k}: {val_str}")

# 统计 context 长度
print(f"\n=== Context 长度统计 ===")
lengths = [len(item["context"]) for item in ds]
print(f"  最短: {min(lengths)} 字符")
print(f"  最长: {max(lengths)} 字符")
print(f"  平均: {sum(lengths)/len(lengths):.0f} 字符")

# 统计 question 数量
print(f"\n=== Answer 数量统计 ===")
ans_counts = [len(item["answer"]) if isinstance(item["answer"], list) else 1 for item in ds]
print(f"  最少答案数: {min(ans_counts)}")
print(f"  最多答案数: {max(ans_counts)}")
print(f"  平均答案数: {sum(ans_counts)/len(ans_counts):.1f}")
