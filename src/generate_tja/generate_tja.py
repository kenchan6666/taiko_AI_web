import torch
import librosa
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 📌 载入 GPT-2 训练好的模型
MODEL_PATH = "model/taiko_model_trained_test/checkpoint-114"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ **关键修正：增加 Pad Token**
tokenizer.pad_token = tokenizer.eos_token


# **音频分析**
def extract_bpm(audio_path):
    """使用 librosa 自动检测 BPM"""
    y, sr = librosa.load(audio_path, sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return int(tempo) if tempo else 143  # **默认 BPM 143**


# **清理生成的 TJA 文本**
def clean_generated_text(text):
    """确保生成的 TJA 格式正确，移除无效字符"""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()

        # 移除不符合 TJA 规范的文本
        if line.startswith("生成一个太鼓谱面"):
            continue  # **跳过无效的 GPT-2 复述**
        if "BPMCHANGE" in line or "SCROLL" in line:
            continue  # **跳过异常的控制指令**
        if len(line) < 3:
            continue  # **过滤太短的无效内容**

        # 只保留有效的谱面数据 (1~9, 逗号)
        valid_line = ''.join(c for c in line if c in "0123456789,")
        if valid_line:
            clean_lines.append(valid_line)

    return "\n".join(clean_lines)


# **生成完整 TJA**
def generate_tja(audio_path, offset=0):
    """🎵 生成完整的 Taiko TJA 谱面（包含 4 个难度）"""
    bpm = extract_bpm(audio_path)
    song_name = audio_path.split("/")[-1].replace(".mp3", "").replace(".wav", "")
    tja_filename = audio_path.replace(".mp3", ".tja").replace(".wav", ".tja")

    # **TJA 头部信息**
    tja_content = f"""TITLE: {song_name}
BPM: {bpm}
OFFSET: {offset}
WAVE: {audio_path.split('/')[-1]}

"""

    # **不同难度的 `TJA` 生成**
    difficulties = {
        "Easy": 3,
        "Normal": 5,
        "Hard": 7,
        "Oni": 9
    }

    for course, level in tqdm(difficulties.items(), desc="🎵 生成谱面中..."):
        print(f"🎵 生成 {course} ({level}) 谱面中...")

        prompt = f"请生成一个太鼓谱面, 难度: {course}, BPM: {bpm}, OFFSET: {offset}, LEVEL: {level}, #START\n"

        # **GPT-2 生成节奏数据**
        input_data = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # ✅ 限制最大 Token 长度，防止溢出
            padding="max_length"
        )

        max_tokens = 300  # ✅ 限制最大生成 Token
        with torch.no_grad():
            output = model.generate(
                input_ids=input_data["input_ids"],
                max_new_tokens=max_tokens,
                temperature=1.0,  # ✅ 控制随机性
                do_sample=True,
                top_k=50,  # ✅ 采样提升质量
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id  # ✅ 解决 IndexError
            )

        # **清理生成的谱面**
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_notes = clean_generated_text(generated_text)

        # **确保谱面不为空**
        if len(generated_notes.strip()) == 0:
            print(f"⚠️ 生成的 {course} 谱面为空，使用占位符")
            generated_notes = "100000000,010000000,001000000,000100000,000010000,000001000,"

        # **合并 TJA 内容**
        tja_content += f"""COURSE:{course}
LEVEL:{level}
#START
{generated_notes}
#END

"""

    # **保存 TJA 文件**
    with open(tja_filename, "w", encoding="utf-8") as f:
        f.write(tja_content)

    print(f"✅ 谱面已生成：{tja_filename}")
    return tja_filename


# **🎯 测试**
if __name__ == "__main__":
    audio_file = "audio/ReadyNow.mp3"
    generate_tja(audio_file, offset=0)


