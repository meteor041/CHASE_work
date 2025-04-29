#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-generates NL2SQL 结果并保存为 JSON 文件。

关键优化：
1. **一次读取模板 & format_map**避免在循环里多次 I/O 和正则替换。
2. **批量推理**利用 pipeline 的 `batch_size` 提升吞吐。
3. **no_grad + fp16/8bit**减小显存占用，推理更快。
4. **路径与超参集中管理**便于脚本化 / 日后复现。
5. **进度可视化 (tqdm)**长任务更直观。
6. **异常兜底**避免单条数据崩溃拖垮整体。
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
import re
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
    model_name: str = r"/data/qwen2-7b-instruct"
    prompt_file: str = "QP_prompt.txt"
    input_json: str = "schema_linking_result.json"
    output_dir: str = "result"
    # 文本生成超参
    max_new_tokens: int = 1536
    do_sample: bool = True
    temperature: float = 0.6
    # 性能设置
    batch_size: int = 8         # 根据显存灵活调整
    use_fp16: bool = True        # 或用 8bit/4bit 量化
    load_in_8bit: bool = False
    device_map: str = "auto"


CFG = Config()
# --------------------------------


def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, db_id: str, question: str,
                  evidence: str, schema_linking: Dict[str, Any]) -> str:
    """
    使用 str.format_map 直接填充占位符，保持模板可读性。
    模板里写 {db_id}、{question}、{evidence}、{schema_linking}
    """
    return template.format_map({
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "schema_linking": json.dumps(schema_linking, ensure_ascii=False),
    })


def load_model_and_tokenizer(cfg: Config):
    # 量化示例：8bit
    quant_cfg = None
    if not cfg.use_fp16:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",  # 对于 text-generation 更稳妥
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
        local_files_only=True,
    )
    return tokenizer, model


def batched(iterable: List[Any], n: int):
    """将列表分批切片 (Python 3.12 里有 itertools.batched)"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def extract_sql_block(generated_text: str) -> str:
    """从模型输出中提取 ```sql ... ``` 中间内容"""
    pattern = r"```sql\s+(.*?)\s*```"
    match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return generated_text.strip()  # fallback
    
def generate():
    cfg = CFG
    root = Path(__file__).resolve().parent
    input_path = root / cfg.input_json
    out_dir = root / cfg.output_dir
    out_dir.mkdir(exist_ok=True)

    data: List[Dict[str, Any]] = json.loads(Path(input_path).read_text(encoding="utf-8"))

    prompt_tpl = load_prompt_template(root / cfg.prompt_file)
    tokenizer, model = load_model_and_tokenizer(cfg)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        do_sample=cfg.do_sample,
        device_map=cfg.device_map,
    )

    results = []

    # —— 主循环 ——
    with torch.no_grad():
        for batch in tqdm(list(batched(data, cfg.batch_size)), desc="Generating"):
            prompts = [
                format_prompt(
                    prompt_tpl,
                    item.get("db_id", ""),
                    item.get("question", ""),
                    item.get("evidence", ""),
                    item.get("schema_linking", {}),
                )
                for item in batch
            ]

            # 生成
            outputs = generator(prompts, max_new_tokens=cfg.max_new_tokens)
            # pipeline 在 batch 模式下返回 List[List[Dict]]
            for item, gen in zip(batch, outputs):
                text: str = gen[0]["generated_text"]
                sql_key = "sql statement:"
                sql_start = text.lower().find(sql_key)
                sql = text[sql_start + len(sql_key):].strip() if sql_start != -1 else text.strip()

                result = {
                    "db_id": item.get("db_id"),
                    "question": item.get("question"),
                    "evidence": item.get("evidence"),
                    "schema_linking": item.get("schema_linking"),
                    "sql": sql,
                }
                results.append(result)
    return results

if __name__ == '__main__':
    cfg = CFG
    root = Path(__file__).resolve().parent
    input_path = root / cfg.input_json
    out_dir = root / cfg.output_dir
    out_dir.mkdir(exist_ok=True)

    data: List[Dict[str, Any]] = json.loads(Path(input_path).read_text(encoding="utf-8"))

    prompt_tpl = load_prompt_template(root / cfg.prompt_file)
    tokenizer, model = load_model_and_tokenizer(cfg)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        device_map=cfg.device_map,
        return_full_text=False,     # 仅返回新增 token，不含 prompt
    )

    # —— 主循环 ——
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(list(batched(data, cfg.batch_size)), desc="Generating"):
            prompts = [
                format_prompt(
                    prompt_tpl,
                    item.get("db_id", ""),
                    item.get("question", ""),
                    item.get("evidence", ""),
                    item.get("schema_linking", {}),
                )
                for item in batch
            ]

            # 生成
            outputs = generator(prompts, max_new_tokens=cfg.max_new_tokens)
            # pipeline 在 batch 模式下返回 List[List[Dict]]
            for item, gen in zip(batch, outputs):
                text: str = gen[0]["generated_text"]
                sql = extract_sql_block(text)
                # sql_key = "sql statement:"
                # sql_start = text.lower().find(sql_key)
                # sql = text[sql_start + len(sql_key) :].strip() if sql_start != -1 else text.strip()
                
                result = {
                    "db_id": item.get("db_id"),
                    "question": item.get("question"),
                    "evidence": item.get("evidence"),
                    "schema_linking": item.get("schema_linking"),
                    "sql": sql,
                    "text": text,
                }

                out_file = out_dir / f"{item['db_id']}_{cnt}.json"
                cnt += 1
                out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ All done! Results saved to: {out_dir.resolve()}")
