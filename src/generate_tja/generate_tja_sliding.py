import torch
import librosa
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "model/taiko_model_trained"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def extract_audio_features(audio_path, segment_duration=5.0, overlap_ratio=0.2):
    """🎵 滑动窗口分割音频，并提取 Mel 频谱"""
    y, sr = librosa.load(audio_path, sr=22050)
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(segment_samples * overlap_ratio)

    mel_features = []
    start = 0
    while start < len(y):
        end = min(start + segment_samples, len(y))
        segment = y[start:end]
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_flat = mel_db.mean(axis=1).tolist()[:100]
        mel_features.append(mel_flat)
        start += segment_samples - overlap_samples  # 滑动窗口

    return mel_features


def generate_tja(audio_path, bpm=120, offset=0):
    """🎯 生成 Taiko TJA 谱面（使用滑动窗口）"""
    audio_segments = extract_audio_features(audio_path)

    difficulty_levels = {"Easy": 3, "Normal": 5, "Hard": 7, "Oni": 9}
    difficulty_max_tokens = {"Easy": 150, "Normal": 250, "Hard": 400, "Oni": 512}

    tja_content = f"""TITLE: Generated Taiko Chart
BPM: {bpm}
OFFSET: {offset}
WAVE: {audio_path.split('/')[-1]}
"""

    for difficulty, level in difficulty_levels.items():
        print(f"🎵 生成 {difficulty} ({level}) 谱面中...")

        segment_notes = []
        for i, segment_features in enumerate(audio_segments):
            print(f"🧩 处理片段 {i + 1}/{len(audio_segments)}")
            prompt = f"BPM:{bpm} OFFSET:{offset} LEVEL:{level} AUDIO:{segment_features}"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs["position_ids"] = torch.arange(0, inputs["input_ids"].size(1)).unsqueeze(0)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=difficulty_max_tokens[difficulty])

            segment_notes.append(tokenizer.decode(output[0], skip_special_tokens=True))

        final_chart = "\n".join(segment_notes)

        tja_content += f"""
COURSE:{difficulty}
LEVEL:{level}
#START
{final_chart}
#END
"""

    tja_filename = audio_path.rsplit(".", 1)[0] + "_sliding.tja"
    with open(tja_filename, "w", encoding="utf-8") as f:
        f.write(tja_content)

    print(f"✅ 谱面已生成：{tja_filename}")
    return tja_filename


if __name__ == "__main__":
    audio_file = "audio/sample.mp3"
    generate_tja(audio_file, bpm=120, offset=-2.0)
