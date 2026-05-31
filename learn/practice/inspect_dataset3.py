# -*- coding: utf-8 -*-
import json
from datasets import load_dataset

ds = load_dataset("luozhouyang/question-answering-datasets", split="train")

# Collect question prefixes and domains
domains = {}
for item in ds:
    q = item["question"]
    # Categorize by question type
    if any(kw in q for kw in ["医院", "医生", "治疗", "手术", "药", "病", "疼", "痛"]):
        domains.setdefault("医疗健康", []).append(q)
    elif any(kw in q for kw in ["手机", "电脑", "价格", "多少钱", "iphone", "vivo", "小米", "华为", "苹果"]):
        domains.setdefault("数码/价格", []).append(q)
    elif any(kw in q for kw in ["电影", "上映", "演员", "电视剧", "综艺", "明星"]):
        domains.setdefault("影视娱乐", []).append(q)
    elif any(kw in q for kw in ["游戏", "dnf", "lol", "王者荣耀", "steam"]):
        domains.setdefault("游戏", []).append(q)
    elif any(kw in q for kw in ["学校", "大学", "高考", "考研", "留学"]):
        domains.setdefault("教育", []).append(q)
    elif any(kw in q for kw in ["公司", "工作", "招聘", "工资", "公务员", "职业"]):
        domains.setdefault("工作/职业", []).append(q)

print("=== 数据集领域分布 ===")
for domain, questions in sorted(domains.items(), key=lambda x: -len(x[1])):
    print(f"\n【{domain}】共 {len(questions)} 条")
    for q in questions[:5]:
        print(f"  - {q}")

print(f"\n总计分类: {sum(len(v) for v in domains.values())} / {len(ds)} 条")
print(f"未分类: {len(ds) - sum(len(v) for v in domains.values())} 条")

# Show some random samples with full context
print("\n\n=== 完整样本示例（含 context）===")
for idx in [1, 2, 10, 34, 44, 52]:
    item = ds[idx]
    print(f"\n--- 样本 {idx+1} ---")
    print(f"问题: {item['question']}")
    print(f"答案: {item['answer']}")
    ctx = item['context'][:300]
    print(f"上下文(前300字): {ctx}")
    print(f"上下文长度: {len(item['context'])} 字符")
