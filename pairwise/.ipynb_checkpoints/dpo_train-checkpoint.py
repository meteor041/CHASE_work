from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# 模型和数据路径
model_name = "/your/local/path/to/qwen3-8b"  # 暂时使用qwen-3-8B
data_path = "qwen_sql_dpo_data.jsonl" # 训练数据来源

# 加载数据
dataset = load_dataset("json", data_files=data_path, split="train")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)

# DPO配置
dpo_config = DPOConfig(
    beta=0.1, # 温度参数
    max_prompt_length=512, # 最长指令长度,用于限制输入提示
    max_length=1024, # 最大总长度
    label_smoothing=0.0, # 标签平滑,这里设置为0,保持原始偏好信息的准确性
)

training_args = TrainingArguments(
    output_dir="./qwen3-dpo-sql",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    bf16=True,
    optim="adamw_torch",
    report_to="none"
)

# 创建Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 如果无对比模型，可以为 None
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dpo_config=dpo_config
)

# 开始训练
trainer.train()
