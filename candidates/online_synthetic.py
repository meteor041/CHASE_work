#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Synthetic (OS) 方法实现
- 根据当前问题动态生成多个输入-输出样例
- 作为Prompt输入来指导SQL生成
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import re

# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"E:\code\CHASE_work\models\qwen2-7b-instruct"  # 请根据实际模型路径调整
    input_json: str = r"E:\code\CHASE_work\data\train\train.json"
    output_dir: str = r"E:\code\CHASE_work\CHASE_work\candidates\os_results"
    # 文本生成超参
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.7
    # 性能设置
    batch_size: int = 4
    use_fp16: bool = True
    device_map: str = "auto"
    # OS特定参数
    num_general_examples: int = 3
    num_schema_aware_examples: int = 3

# 配置实例
CFG = Config()

def load_model_and_tokenizer(cfg: Config):
    """加载模型和分词器"""
    # 量化配置
    quant_cfg = None
    if not cfg.use_fp16:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
    )
    return tokenizer, model

def batched(iterable: List[Any], n: int):
    """将列表分批切片"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# 生成通用示例的提示模板
GENERAL_EXAMPLES_TEMPLATE = """你是一个专业的数据库专家。你的任务是为给定的数据库生成一些自然语言问题和对应的SQL查询示例。

数据库信息:
{db_schema}

请生成{num_examples}个不同的自然语言问题和对应的SQL查询示例。这些示例应该涵盖不同的SQL特性，如:
1. 简单的SELECT查询
2. 带有WHERE条件的查询
3. 多表JOIN查询
4. 带有GROUP BY和聚合函数的查询
5. 带有ORDER BY的查询
6. 带有LIMIT的查询
7. 带有子查询的复杂查询

对于每个示例，请使用以下格式:
问题: [自然语言问题]
SQL: [对应的SQL查询]

请确保SQL查询是正确的，并且能够在给定的数据库上运行。
"""

# 生成schema-aware示例的提示模板
SCHEMA_AWARE_TEMPLATE = """你是一个专业的数据库专家。你的任务是为给定的数据库和相关列生成一些自然语言问题和对应的SQL查询示例。

数据库信息:
{db_schema}

相关列:
{relevant_columns}

请生成{num_examples}个不同的自然语言问题和对应的SQL查询示例。这些示例应该特别关注上述相关列，并且涵盖不同的SQL操作。

对于每个示例，请使用以下格式:
问题: [自然语言问题]
SQL: [对应的SQL查询]

请确保SQL查询是正确的，并且能够在给定的数据库上运行。
"""

# 使用few-shot示例生成SQL的提示模板
FEW_SHOT_TEMPLATE = """你是一个专业的数据库专家。你的任务是将自然语言问题转换为SQL查询。

数据库信息:
{db_schema}

以下是一些示例:

{examples}

现在，请为以下问题生成SQL查询:
问题: {question}

SQL: 
"""

def filter_columns_by_question(question: str, db_schema: str, generator) -> List[str]:
    """根据问题过滤出相关的列"""
    prompt = f"""你是一个专业的数据库专家。你的任务是从数据库模式中找出与给定问题相关的列。

数据库信息:
{db_schema}

问题:
{question}

请列出与这个问题最相关的列名（包括表名前缀），每行一个列名。只输出列名，不要有其他解释。
"""
    
    response = generator(prompt, max_new_tokens=512, do_sample=False)
    text = response[0]["generated_text"].strip()
    
    # 解析列名
    columns = []
    for line in text.split('\n'):
        if line.strip():
            columns.append(line.strip())
    
    return columns

def generate_examples_by_sql_features(db_schema: str, num_examples: int, generator) -> List[Tuple[str, str]]:
    """生成涵盖不同SQL特性的示例"""
    prompt = GENERAL_EXAMPLES_TEMPLATE.format(
        db_schema=db_schema,
        num_examples=num_examples
    )
    
    response = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.8)
    text = response[0]["generated_text"].strip()
    
    # 解析示例
    examples = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].startswith("问题:"):
            question = lines[i][3:].strip()
            i += 1
            # 寻找SQL行
            while i < len(lines) and not lines[i].startswith("SQL:"):
                i += 1
            if i < len(lines):
                sql = lines[i][4:].strip()
                examples.append((question, sql))
        i += 1
    
    return examples[:num_examples]  # 确保不超过请求的数量

def generate_examples_by_schema(db_schema: str, relevant_columns: List[str], num_examples: int, generator) -> List[Tuple[str, str]]:
    """生成基于特定schema的示例"""
    prompt = SCHEMA_AWARE_TEMPLATE.format(
        db_schema=db_schema,
        relevant_columns='\n'.join(relevant_columns),
        num_examples=num_examples
    )
    
    response = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.8)
    text = response[0]["generated_text"].strip()
    
    # 解析示例
    examples = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].startswith("问题:"):
            question = lines[i][3:].strip()
            i += 1
            # 寻找SQL行
            while i < len(lines) and not lines[i].startswith("SQL:"):
                i += 1
            if i < len(lines):
                sql = lines[i][4:].strip()
                examples.append((question, sql))
        i += 1
    
    return examples[:num_examples]  # 确保不超过请求的数量

def format_few_shot_prompt(examples: List[Tuple[str, str]], question: str, db_schema: str) -> str:
    """格式化few-shot提示"""
    examples_text = ""
    for i, (q, sql) in enumerate(examples, 1):
        examples_text += f"示例 {i}:\n问题: {q}\nSQL: {sql}\n\n"
    
    return FEW_SHOT_TEMPLATE.format(
        db_schema=db_schema,
        examples=examples_text,
        question=question
    )

def extract_sql(text: str) -> str:
    """从生成的文本中提取SQL"""
    # 尝试找到SQL:后面的内容
    if "SQL:" in text:
        return text.split("SQL:", 1)[1].strip()
    return text.strip()

def online_synthetic_icl(question: str, db_schema: str, generator, 
                         num_general: int = 3, num_schema_aware: int = 3):
    """主函数：使用在线合成示例方法生成SQL"""
    # 步骤1: 使用常见SQL特征生成通用示例
    general_examples = generate_examples_by_sql_features(db_schema, num_general, generator)
    
    # 步骤2: 使用过滤列生成schema-aware示例
    relevant_columns = filter_columns_by_question(question, db_schema, generator)
    schema_examples = generate_examples_by_schema(db_schema, relevant_columns, num_schema_aware, generator)
    
    # 步骤3: 组合所有示例 + 当前问题进入Prompt
    prompt = format_few_shot_prompt(general_examples + schema_examples, question, db_schema)
    
    # 步骤4: 生成SQL
    response = generator(prompt, max_new_tokens=1024, do_sample=False)
    return extract_sql(response[0]["generated_text"])

def process_data():
    """处理数据并生成SQL"""
    # 创建输出目录
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    # 加载数据
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载模型和分词器
    tokenizer, model = load_model_and_tokenizer(CFG)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    
    results = []
    
    # 处理每个样本
    for item in tqdm(data, desc="处理样本"):
        question = item.get("question", "")
        db_schema = item.get("db_schema", "")
        
        # 使用OS方法生成SQL
        try:
            generated_sql = online_synthetic_icl(
                question, 
                db_schema, 
                generator,
                num_general=CFG.num_general_examples,
                num_schema_aware=CFG.num_schema_aware_examples
            )
            
            # 保存结果
            result = {
                "id": item.get("id", ""),
                "question": question,
                "db_schema": db_schema,
                "generated_sql": generated_sql,
                "gold_sql": item.get("query", "")  # 原始正确SQL
            }
            results.append(result)
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
    
    # 保存所有结果
    output_path = os.path.join(CFG.output_dir, "os_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    process_data()