from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import json
import argparse
import os

def load_model_and_tokenizer(model_path, lora_path):
    """加载基础模型、LoRA权重和分词器"""
    print(f"正在加载基础模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"正在加载分类模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    print(f"正在加载LoRA权重: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_better_sql(model, tokenizer, prompt, device):
    """预测哪个SQL更好"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return "A" if prediction == 0 else "B", confidence

def process_json_file(json_path, model, tokenizer, device, output_path=None):
    """处理JSON文件中的SQL对"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for i, item in enumerate(data):
        prompt = item.get('prompt', '')
        
        if not prompt:
            print(f"警告: 项目 {i} 没有prompt字段，跳过")
            continue
        
        print(f"\n处理第 {i+1}/{len(data)} 个样本...")
        
        # 预测更好的SQL
        better_sql, confidence = predict_better_sql(model, tokenizer, prompt, device)
        
        # 提取原始标签（如果有）
        original_label = item.get('label', None)
        
        # 构建结果
        result = {
            'id': item.get('id', i),
            'prompt': prompt,
            'prediction': better_sql,
            'confidence': confidence,
            'original_label': original_label,
            'correct': original_label == better_sql if original_label else None
        }
        
        results.append(result)
        
        # 打印结果
        print(f"预测: {better_sql} (置信度: {confidence:.4f})")
        if original_label:
            print(f"原始标签: {original_label}")
            print(f"预测{'正确' if result['correct'] else '错误'}")
    
    # 计算准确率（如果有原始标签）
    if all(r['original_label'] is not None for r in results):
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        print(f"\n总体准确率: {accuracy:.4f}")
    
    # 保存结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用训练好的LoRA模型选择更好的SQL")
    parser.add_argument("--json_path", type=str, required=True, help="包含SQL对的JSON文件路径")
    parser.add_argument("--model_path", type=str, default="E:\\code\\CHASE_work\\models\\qwen3-8b", help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default="E:\\code\\CHASE_work\\CHASE_work\\pairwise\\pairwise_selector_model\\qwen3-8b-lora", help="LoRA权重路径")
    parser.add_argument("--output_path", type=str, help="输出结果的JSON文件路径")
    
    args = parser.parse_args()
    
    # 如果未指定输出路径，则在输入文件旁边创建
    if not args.output_path:
        input_dir = os.path.dirname(args.json_path)
        input_filename = os.path.basename(args.json_path)
        output_filename = f"results_{input_filename}"
        args.output_path = os.path.join(input_dir, output_filename)
    
    # 加载模型和分词器
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.lora_path)
    
    # 处理JSON文件
    process_json_file(args.json_path, model, tokenizer, device, args.output_path)

if __name__ == "__main__":
    main()