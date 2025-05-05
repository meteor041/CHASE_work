import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

class SimpleNL2SQL:
    def __init__(self, model_path: str = "/home/yangliu26/qwen3-8b"):
        print("正在加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("模型加载完成")
        
    def generate_sql(self, question: str, hint: str, schema: str) -> str:
        """生成单条SQL"""
        prompt = f"""请根据以下信息生成SQL查询语句:

问题: {question}

提示: {hint}

数据库表结构:
{schema}

请直接输出SQL语句，不要有任何解释。
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取生成的SQL
        sql = response[len(prompt):].strip()
        return sql

def main():
    # 设置输入输出路径
    input_path = "/home/yangliu26/data/train/train.json"
    output_path = "simple_nl2sql_results.json"
    
    # 初始化模型
    nl2sql = SimpleNL2SQL()
    
    # 读取输入数据
    print("正在读取数据...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 生成SQL
    print("开始生成SQL...")
    results = []
    for item in tqdm(data):
        sql = nl2sql.generate_sql(
            question=item["question"],
            hint=item.get("hint", ""),  # 如果没有hint则使用空字符串
            schema=item["db_schema"]
        )
        results.append(sql)
    
    # 保存结果
    print("正在保存结果...")
    with open(output_path, "w", encoding="utf-8") as f:
        for sql in results:
            f.write(sql + "\n")
    
    print(f"处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    main()