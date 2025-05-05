#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Synthetic (OS) 方法实现
- 根据当前问题动态生成多个输入-输出样例
- 作为Prompt输入来指导SQL生成
"""
import multiprocessing
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
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
    model_name: str = r"/home/yangliu26/qwen3-8b"  # 请根据实际模型路径调整
    input_json: str = r"/home/yangliu26/data/train/schema_linking_result.json"
    output_dir: str =  r"/home/yangliu26/CHASE/candidates/os_results"
    
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
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

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

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def generate_examples_by_sql_features(db_schema: str, num_examples: int, generator) -> List[Tuple[str, str]]:
    # GENERAL_EXAMPLES_TEMPLATE是生成通用示例的提示模板
    GENERAL_EXAMPLES_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/general_examples_template.txt")
    """生成涵盖不同SQL特性的示例"""
    prompt = GENERAL_EXAMPLES_TEMPLATE.format(
        db_schema=db_schema,
        num_examples=num_examples
    )
    
    response = generator(prompt, max_new_tokens=1024, do_sample=True, temperature=0.8)
    text = response[0]["generated_text"].strip()
    
    # 解析示例
    examples = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].startswith("Question:"):
            question = lines[i][9:].strip()
            i += 1
            # 寻找SQL行
            while i < len(lines) and not lines[i].startswith("SQL:"):
                i += 1
            if i < len(lines):
                sql = lines[i][4:].strip()
                examples.append((question, sql))
        i += 1
    
    return examples[:num_examples]  # 确保不超过请求的数量

def generate_examples_by_schema(db_schema: str, num_examples: int, generator) -> List[Tuple[str, str]]:
    SCHEMA_AWARE_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/schema_aware_template.txt")
    """生成基于特定schema的示例"""
    prompt = SCHEMA_AWARE_TEMPLATE.format(
        db_schema=db_schema,
        num_examples=num_examples
    )
    
    response = generator(prompt, max_new_tokens=1024, do_sample=True, temperature=0.8)
    text = response[0]["generated_text"].strip()
    
    # 解析示例
    examples = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].startswith("Question:"):
            question = lines[i][9:].strip()
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
    FEW_SHOT_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/os_few_shot_template.txt")
    """格式化few-shot提示"""
    examples_text = ""
    for i, (q, sql) in enumerate(examples, 1):
        examples_text += f"Example {i}:\nQuestion: {q}\nSQL: {sql}\n"
    
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
    print("步骤1: 使用常见SQL特征生成通用示例")
    general_examples = generate_examples_by_sql_features(db_schema, num_general, generator)
    
    print("步骤2: 生成schema-aware示例")
    schema_examples = generate_examples_by_schema(db_schema, num_schema_aware, generator)
    
    print("步骤3: 组合所有示例 + 当前问题进入Prompt")
    prompt = format_few_shot_prompt(general_examples + schema_examples, question, db_schema)
    
    print("步骤4: 生成SQL")
    response = generator(prompt, max_new_tokens=1024, do_sample=False)
    return general_examples, schema_examples, prompt, extract_sql(response[0]["generated_text"])

def process_single_item(item: Dict[str, Any], generator) -> Dict[str, Any]:
    question = item.get("question", "")
    db_schema = item.get("schema_linking", "")
    
    # 使用OS方法生成SQL
    try:
        general_examples, schema_examples, prompt, generated_sql = online_synthetic_icl(
            question, 
            db_schema, 
            generator,
            num_general=CFG.num_general_examples,
            num_schema_aware=CFG.num_schema_aware_examples
        )
        
        # 保存结果
        return {
            "id": item.get("id", ""),
            "question": question,
            "db_schema": db_schema,
            "sql": generated_sql,
            "general_examples": general_examples,
            "schema_examples": schema_examples,
            "prompt": prompt
        }
    except Exception as e:
        print(f"处理样本 {item.get('id', '')} 时出错: {e}")
        return None
        
def process_data():
    """处理数据并生成SQL"""
    # 创建输出目录
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    # 加载数据
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[0:1]
    print(data)
    # 加载模型和分词器
    tokenizer, model = load_model_and_tokenizer(CFG)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    
    results = []
    
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_item, item, generator) for item in data]
        # 使用tqdm显示进度
        for future in tqdm(futures, total=len(data), desc="处理样本"):
            result = future.result()
            if result is not None:
                results.append(result) 
            
    
    # 保存所有结果
    output_path = os.path.join(CFG.output_dir, "os_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    process_data()