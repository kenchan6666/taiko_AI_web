from transformers import AutoModelForCausalLM, AutoTokenizer
from process import extract_audio_features

MODEL_PATH = "./model/taiko_model_trained"

def generate_chart(audio_path, bpm, level):
    """AI 生成谱面，考虑 LEVEL 难度"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    audio_features = extract_audio_features(audio_path, 0)

    input_text = f"BPM: {bpm}, Level: {level}, Audio Features: {audio_features[:10]}, Generate Taiko drum chart."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    result = generate_chart("data/track1.mp3", 163, 8)
    print("Generated Taiko Chart:", result)
