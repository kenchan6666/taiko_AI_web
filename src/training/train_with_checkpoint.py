from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
import os

# **📌 1. 加载模型和 tokenizer**
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# **📌 2. 加载数据集**
train_dataset = datasets.load_dataset("json", data_files="data/taiko_dataset.jsonl", split="train")

# **📌 3. Tokenize 数据（开启多线程）**
def tokenize_function(examples):
    return tokenizer(
        examples["notes"],  # **使用 "notes" 作为输入字段**
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)  # **开启 4 线程加速**

# **📌 4. 训练参数**
training_args = TrainingArguments(
    output_dir="./model/taiko_model_trained",
    evaluation_strategy="steps",  # **每 save_steps 评估一次**
    save_strategy="steps",  # **每隔 X 步保存模型**
    save_steps=500,  # **每 500 步保存**
    save_total_limit=3,  # **最多保留 3 个 checkpoint**
    per_device_train_batch_size=4,  # **增大 batch size，加快训练**
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),  # **如果 GPU 支持 FP16，则启用**
    dataloader_num_workers=4,  # **多线程数据加载**
    report_to="none"  # **关闭 Wandb / Tensorboard**
)

# **📌 5. 训练器**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# **📌 6. 检查是否有 Checkpoint**
checkpoint_path = "./model/taiko_model_trained/checkpoint-latest"
if os.path.exists(checkpoint_path):
    print(f"🔄 继续从 {checkpoint_path} 训练...")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("🚀 开始新的训练...")
    trainer.train()
