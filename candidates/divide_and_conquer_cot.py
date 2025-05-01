#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide-and-Conquer CoT 方法实现
- 将复杂自然语言问题分解为多个子问题
- 对每个子问题分别生成SQL片段
- 最后合并这些片段，构造完整的SQL
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"E:\code\CHASE_work\models\qwen2-7b-instruct"  # 请根据实际模型路径调整
    input_json: str = r"E:\code\CHASE_work\data\train\train.json"
    output_dir: str = r"E:\code\CHASE_work\CHASE_work\candidates\dac_results"
    # 文本生成超参
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.7
    # 性能设置
    batch_size: int = 4
    use_fp16: bool = True
    device_map: str = "auto"

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
        local_files_only=True,
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

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
    
def decompose_question(question: str, db_schema: str, generator) -> List[str]:
    # 分解步骤的提示词模板
    DECOMPOSE_TEMPLATE = load_prompt_template(r"CHASE_work\template\decompose_template.txt")

    """将问题分解为多个子问题"""
    prompt = DECOMPOSE_TEMPLATE.format(
        question=question,
        db_schema=db_schema
    )
    
    response = generator(prompt, max_new_tokens=512, do_sample=False)
    text = response[0]["generated_text"]
    
    # 解析子问题
    sub_questions = []
    for line in text.strip().split('\n'):
        if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
            # 移除序号前缀
            sub_q = line.strip()
            for i in range(1, 10):
                prefix = f"{i}. "
                if sub_q.startswith(prefix):
                    sub_q = sub_q[len(prefix):]
                    break
            sub_questions.append(sub_q)
    
    return sub_questions

def generate_partial_sql(sub_question: str, db_schema: str, generator) -> str:
     # 生成SQL片段的提示词模板
    PARTIAL_SQL_TEMPLATE = load_prompt_template(r"CHASE_work\template\partial_sql_template.txt")
    
    """为子问题生成SQL片段"""
    prompt = PARTIAL_SQL_TEMPLATE.format(
        sub_question=sub_question,
        db_schema=db_schema
    )
    
    response = generator(prompt, max_new_tokens=512, do_sample=False)
    return response[0]["generated_text"].strip()

def assemble_sql(question: str, db_schema: str, sub_questions: List[str], 
                partial_sqls: List[str], generator) -> str:
    # 组合SQL的提示词模板
    ASSEMBLE_TEMPLATE = load_prompt_template(r"CHASE_work\template\assemble_template.txt")
    
    """组合SQL片段为完整SQL"""
    # 格式化子问题和SQL片段
    sub_qs_and_sqls = ""
    for i, (q, sql) in enumerate(zip(sub_questions, partial_sqls), 1):
        sub_qs_and_sqls += f"{i}. 子问题: {q}\n   SQL片段: {sql}\n\n"
    
    prompt = ASSEMBLE_TEMPLATE.format(
        question=question,
        db_schema=db_schema,
        sub_questions_and_sqls=sub_qs_and_sqls
    )
    
    response = generator(prompt, max_new_tokens=1024, do_sample=False)
    return response[0]["generated_text"].strip()

def optimize_sql(sql: str, generator) -> str:
    """优化SQL查询（可选）"""
    # 这里可以添加SQL优化逻辑，如去除冗余等
    # 简单实现，直接返回
    return sql

def divide_and_conquer_sql(question: str, db_schema: str, generator):
    """主函数：使用分而治之方法生成SQL"""
    # 步骤1: 分解问题
    sub_questions = decompose_question(question, db_schema, generator)
    
    # 步骤2: 生成每个子问题的SQL片段
    partial_sqls = []
    for q in sub_questions:
        partial_sql = generate_partial_sql(q, db_schema, generator)
        partial_sqls.append(partial_sql)
    
    # 步骤3: 汇总构造最终SQL
    final_sql = assemble_sql(question, db_schema, sub_questions, partial_sqls, generator)
    
    return optimize_sql(final_sql, generator)

def process_data():
    """处理数据并生成SQL"""
    # 创建输出目录(若已存在,则不用创建)
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    # 加载数据
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载模型和分词器
    tokenizer, model = load_model_and_tokenizer(CFG)

    # 创建生成器
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
        
    # 处理每个样本
    results = []
    for i, item in enumerate(tqdm(data, desc="Processing")):
        try:
            question = item.get("question", "")
            db_schema = item.get("db_schema", "")  # 假设数据中有db_schema字段
            db_id = item.get("db_id", f"db_{i}")
            
            # 使用Divide-and-Conquer方法生成SQL
            sql = divide_and_conquer_sql(question, db_schema, generator)
            
            # 保存结果
            result = {
                "db_id": db_id,
                "question": question,
                "sql": sql,
            }
            results.append(result)
            
            # 写入单个文件
            output_file = os.path.join(CFG.output_dir, f"{db_id}_{i}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
    
    # 写入汇总文件
    with open(os.path.join(CFG.output_dir, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果保存在: {CFG.output_dir}")

if __name__ == "__main__":
    process_data()