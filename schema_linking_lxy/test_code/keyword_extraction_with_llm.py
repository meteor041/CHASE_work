from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
# 这里假设使用零样本/少样本prompt方式，实际可根据具体模型调整

model_path = r"/data/qwen2-7b-instruct"
class keyword_extractor():
    def __init__(self, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # FP16
            device_map="auto",         # accelerate 自动分配 GPU
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id  # 用专用 <pad>
        
        
    def extract_keywords(self, question: str, prompt_template: str = None) -> List[str]:
        """
        利用LLM和few-shot prompt抽取自然语言问题中的关键词。
        """
        if prompt_template == None:
            prompt_template = (
            "Input question: {question}\n"
            "Please strictly follow these requirements:\n"
            "1. Only extract the key entities, attributes, numbers, or filter conditions from the question as keywords;\n"
            "2. Output only the following standard JSON format without adding any extra explanation or text;\n"
            "3. If no keywords can be extracted or the question is unclear, return an empty list.\n\n"
            "Standard output format:\n"
            "{{\n"
            '  "keywords": ["<keyword1>", "<keyword2>", "..."]\n'
            "}}\n"
            "【Special Note】：\n"
            "- Do not output anything other than the JSON content\n"
            "- Keep the keywords concise; retain numbers and units if necessary\n"
            "- If the format is incorrect or extra content is output, the answer will be considered invalid\n"
            )
        prompt = prompt_template.format(question=question)
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        out = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            return_dict_in_generate=True,  # 关键！返回 structured outputs
            return_legacy_cache=True  # <<< 这样旧版兼容模式
        )
        # 只取 newly generated tokens
        generated_tokens = out.sequences[:, inputs.input_ids.shape[1]:]  # 只保留新增部分
        result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(result)
        # 读取结果
        # 1. 预处理: 去除 ```json ``` 包裹
        import re
        cleaned = result.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = re.sub(r"^```(json)?\s*", "", cleaned)  # 去掉开头
            cleaned = re.sub(r"\s*```$", "", cleaned) 
        # print("cleaned: ", cleaned)
        import json
        try:
            parsed = json.loads(cleaned)
            if isinstance(cleaned, list):
                cleaned = cleaned[0]
            keywords = parsed.get("keywords", [])
        except json.JSONDecodeError:
            keywords = []
        return keywords

if __name__ == "__main__":
    q = "有哪些销售额超过50000的店铺？"
    ex = keyword_extractor(model_path)
    print(ex.extract_keywords(q))