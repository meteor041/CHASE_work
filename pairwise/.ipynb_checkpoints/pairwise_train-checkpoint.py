from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import json
import os
def format_pairwise_example(example):
    return {
        "text": example['prompt'],
        "label": 0 if example["label"] == "A" else 1
    }
    
# 加载 JSON 数据集
def load_json_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return Dataset.from_list([format_pairwise_example(ex) for ex in raw])

dataset = load_json_dataset("/home/yangliu26/CHASE/utils/pairwise_datas.json")
dataset = dataset.train_test_split(test_size=0.1)

model_name = "/home/yangliu26/qwen3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 编码函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print("train size:", len(tokenized_dataset["train"]))
print("test size:", len(tokenized_dataset["test"]))

# from transformers import is_torch_tpu_available

if not torch.distributed.is_initialized():
    try:
        import deepspeed
        deepspeed.init_distributed()
        print(f"✅ 手动初始化完成: Rank {torch.distributed.get_rank()}")
    except Exception as e:
        print(f"❌ DeepSpeed 手动初始化失败: {e}")
    
print("正在初始化训练参数")
# 训练参数
training_args = TrainingArguments(
    output_dir="./pairwise_selector_model/qwen3-8b/",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    bf16=True,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    ddp_find_unused_parameters=False,
    deepspeed="/home/yangliu26/CHASE/pairwise/ds_config.json"
)
print("初始训练化参数完成")

print("⚙️ 正在初始化 Trainer ...")
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)
print("🔥 Calling trainer.train()...")

print("🔥 Calling trainer.train()...")
trainer.train()
print("✅ training completed")
