from typing import List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import json
import re

class KeywordExtractor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            enable_thinking=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        # 保险起见，再显式关一次 generation_config
        self.model.generation_config.enable_thinking = False  
        
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def build_prompt(self, question: str, prompt_template: str = None) -> str:
        if prompt_template is None:
            prompt_template = """
You are a highly precise extraction assistant.

Task:
Given an input question, **only extract** the key entities, attributes, numbers, or filtering conditions that are essential for understanding the question.

Output Rules:
- Return **only** the extracted keywords in a **strict JSON object** format.
- **Do not** include explanations, comments, examples, greetings, or any extra text.
- **Do not** mention \"Example\" or \"Assistant:\" or anything else beyond the JSON.
- If no keywords can be extracted, return an empty list in JSON format.

Strict Output Format:
```json
{{
  "keywords": ["<keyword1>", "<keyword2>", "..."]
}}
```

Input Question:
{question}

Respond with **only** the strict JSON object as specified above.
"""
        return prompt_template.format(question=question)

    def extract_json_block(self, text: str) -> dict | None:
        """
        从文本中抓取第一个合法的 JSON 对象并反序列化。
        返回 dict；若没找到返回 None。
        """
        # 非贪婪匹配最外层 {...}
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

    
    def clean_output(self, result: str) -> str:
        # 去除包裹的 ```json ``` 和其他markdown
        cleaned = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", result.strip(), flags=re.DOTALL)
        # 尝试提取第一个出现的大括号 {} 或数组 []
        match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", cleaned)
        if match:
            print(f"匹配成功:{match.group(0)}")
            return match.group(0)
        print(f"匹配失败:{cleaned}")
        return cleaned  # fallback

    def parse_keywords(self, cleaned: str) -> List[str]:
        try:
            parsed = self.extract_json_block(cleaned)
            if isinstance(parsed, list):
                parsed = parsed[0]
            return parsed.get("keywords", [])
        except json.JSONDecodeError as e:
            print(f"[Parsing Failed] Content: {cleaned}")
            print(f"[Error Info] {e}")
            return []

    async def extract_keywords(self, question: str, prompt_template: str = None, timeout_sec: int = 20) -> List[str]:
        prompt = self.build_prompt(question, prompt_template)
        inputs = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        try:
            output = await asyncio.wait_for(self._generate(inputs), timeout=timeout_sec)
            # print(output)
            cleaned = self.clean_output(output)
            keywords = self.parse_keywords(cleaned)
            return keywords
        except asyncio.TimeoutError:
            print("[Timeout Warning] Response generation timeout, skipping this question.")
            return []

    async def _generate(self, inputs) -> str:
        out = self.model.generate(
            **inputs,
            max_new_tokens=128,
            # temperature=0.01,
            # top_p=0.8,
            do_sample=False,
            return_dict_in_generate=True,
            enable_thinking=False,
        )
        generated_tokens = out.sequences[:, inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return result

    async def batch_extract(self, questions: List[str], prompt_template: str = None) -> List[List[str]]:
        tasks = [self.extract_keywords(q, prompt_template) for q in questions]
        results = await asyncio.gather(*tasks)
        return results


if __name__ == "__main__":
    model_path = "/home/yangliu26/qwen3-8b"
    extractor = KeywordExtractor(model_path)

    async def main():
        questions = [
            "Which stores have sales exceeding 50,000?",
            "What are the popular restaurants in Beijing in 2023?",
            "List the training institutions that offer courses in Python and deep learning."
        ]
        results = await extractor.batch_extract(questions)
        for q, keywords in zip(questions, results):
            print(f"Question: {q}\nKeywords: {keywords}\n")

    asyncio.run(main())
