from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载训练好的模型和 tokenizer
MODEL_PATH = "model/taiko_model_trained_test/checkpoint-114"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "BPM:120 OFFSET:0 LEVEL:8"
    generated_text = generate_text(prompt)
    print("\n🎵 生成的谱面内容：\n")
    print(generated_text)
