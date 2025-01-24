from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import datasets
import torch

# **加载 GPT-Neo 1.3B**
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ **增加 Pad Token**
tokenizer.pad_token = tokenizer.eos_token

# **加载数据集**
dataset = datasets.load_dataset("json", data_files="data/taiko_dataset.jsonl", split="train")

# **Tokenize 数据**
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["notes"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # ✅ 让 `labels = input_ids`
    return tokenized

if __name__ == "__main__":
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=1)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=1)

    # **训练参数**
    training_args = TrainingArguments(
        output_dir="./model/taiko_model_2.0",
        per_device_train_batch_size=1,  # ✅ 降低 batch size，避免 OOM
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        eval_strategy="steps",  # ✅ 修正 `evaluation_strategy`
        eval_steps=500,
        save_steps=500,
        fp16=True,
        dataloader_num_workers=4,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # **训练器**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # **开始训练**
    trainer.train()

    # ✅ **保存最终训练好的模型**
    trainer.save_model("./model/taiko_model_2.0")
    tokenizer.save_pretrained("./model/taiko_model_2.0")

    print("✅ 训练完成，模型已保存到 `model/taiko_model_2.0`")



