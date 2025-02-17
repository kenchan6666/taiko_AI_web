from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import datasets
import json

# 🔹 选择 GPT-2 作为模型
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔹 GPT-2 没有 pad_token，需要手动设置
tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name)


# 读取 JSONL 数据并转换为文本格式
def preprocess_data(file_path):
    data_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # 只保留标准难度（忽略 Edit）
            if data["course"].lower() == "edit":
                continue

            # 构造 GPT-2 适合的文本格式
            text = (
                f"Title: {data['title']}\n"
                f"BPM: {data['bpm']}\n"
                f"Level: {data['level']}\n"
                f"Notes: {data['notes']}\n"
                f"---\n"
            )
            data_list.append({"text": text})

    return data_list


# 预处理数据
dataset = preprocess_data("data/taiko_dataset_test.jsonl")

# 使用 datasets 库加载数据
train_dataset = datasets.Dataset.from_dict({"text": [d["text"] for d in dataset]})


# 🔹 Tokenize 数据
def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # 关键：确保 labels 和 input_ids 一致
    return tokenized_inputs


train_dataset = train_dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./model/taiko_model_trained_test",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# 训练模型
trainer.train()

