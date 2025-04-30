from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import json

# # 加载/处理数据
# def format_pairwise_example(example):
#     # 从prompt中提取问题和数据库模式
#     prompt_parts = example['prompt'].split('**************************')
#     schema = prompt_parts[1].strip()
#     question = prompt_parts[2].strip()
#     candidates = prompt_parts[3:5]
    
#     return {
#         "text": example['prompt'],
#         "label": 0 if example["label"] == "A" else 1
#     }

# 加载 JSON 数据集
def load_json_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return raw
    # return Dataset.from_list([format_pairwise_example(ex) for ex in raw])

dataset = load_json_dataset("../pairwise_datas.json")
dataset = dataset.train_test_split(test_size=0.1)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 编码函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 训练参数
training_args = TrainingArguments(
    output_dir="./pairwise_selector_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

# trainer.train()
