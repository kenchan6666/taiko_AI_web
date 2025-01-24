import os
import json
import librosa
import numpy as np
import chardet
import subprocess
import os
import pathlib

FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"  # 确保路径正确


def find_tja_and_audio(root_folder):
    """递归搜索根目录，找到所有 TJA 文件和对应的音频文件"""
    dataset = []

    for subdir, _, files in os.walk(root_folder):
        tja_file = None
        audio_file = None

        # 找到 TJA 和 音频文件
        for file in files:
            if file.endswith(".tja"):
                tja_file = os.path.join(subdir, file)
            elif file.endswith((".mp3", ".ogg", ".wav")):
                audio_file = os.path.join(subdir, file)

        # 确保找到完整的配对
        if tja_file and audio_file:
            dataset.append({"tja": tja_file, "audio": audio_file})

    return dataset


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']


def parse_tja(file_path):
    """解析 TJA 文件，提取 BPM、Offset 和 鼓点数据"""
    encoding = detect_encoding(file_path)  # 自动检测编码
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.readlines()
    except UnicodeDecodeError:
        print(f"⚠️ {file_path} 编码检测失败，尝试 UTF-8 重新打开")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()

    metadata = {
        "title": None,
        "bpm": None,
        "offset": None,
        "audio": None,
        "courses": {}
    }

    current_course = None
    current_level = None
    notes = []
    in_chart = False

    for line in content:
        line = line.strip()

        # 解析元信息
        if line.startswith("TITLE:"):
            metadata["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("BPM:"):
            metadata["bpm"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("OFFSET:"):
            metadata["offset"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("WAVE:"):
            metadata["audio"] = line.split(":", 1)[1].strip()

        # 解析 COURSE
        elif line.startswith("COURSE:"):
            current_course = line.split(":", 1)[1].strip()
            if current_course.lower() == "edit":
                print(f"⚠️ 忽略 Edit: {file_path}")
                current_course = None  # 忽略 Edit
            else:
                metadata["courses"][current_course] = {"level": None, "balloon": [], "notes": []}  # 预初始化

        # 解析 LEVEL
        elif line.startswith("LEVEL:"):
            if current_course:
                try:
                    current_level = int(line.split(":", 1)[1].strip())
                    metadata["courses"][current_course]["level"] = current_level
                except ValueError:
                    print(f"⚠️ 解析 LEVEL 失败: {line}")

        # 解析 BALLOON
        elif line.startswith("BALLOON:"):
            if current_course:
                balloon_values = line.split(":", 1)[1].strip()
                metadata["courses"][current_course]["balloon"] = [int(x) for x in balloon_values.split(",") if x.isdigit()]

        # 解析谱面数据
        elif line.startswith("#START"):
            in_chart = True
            notes = []
        elif line.startswith("#END"):
            in_chart = False
            if current_course:
                metadata["courses"][current_course]["notes"] = notes
        elif in_chart and line:
            notes.append(line.replace(",", "").strip())

    return metadata


def normalize_path(file_path):
    """修正 Windows 复杂路径 (中文, 日文, 符号)"""
    return str(pathlib.Path(file_path).resolve())

def convert_audio_to_wav(audio_path):
    """使用 FFmpeg 将音频转换为 WAV"""
    wav_path = audio_path.rsplit(".", 1)[0] + ".wav"

    if not os.path.exists(wav_path):
        try:
            subprocess.run([FFMPEG_PATH, "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "22050", wav_path], check=True)
            return wav_path
        except subprocess.CalledProcessError:
            print(f"FFmpeg 处理失败: {audio_path}")
            return None
    return wav_path

def extract_audio_features(audio_path, offset):
    """提取 Mel 频谱，修复路径问题"""
    audio_path = normalize_path(audio_path)

    # 如果是 MP3/OGG，转换成 WAV
    if audio_path.endswith((".mp3", ".ogg")):
        audio_path = convert_audio_to_wav(audio_path)
        if audio_path is None:
            return None

    try:
        y, sr = librosa.load(audio_path, sr=22050, offset=offset)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        return mel.mean(axis=1).tolist()
    except Exception as e:
        print(f"音频加载失败: {audio_path}, 错误: {e}")
        return None


def prepare_dataset(root_folder, output_file):
    """自动查找 TJA 和 音频文件，并转换成 JSONL 格式"""
    dataset = []
    file_pairs = find_tja_and_audio(root_folder)

    for pair in file_pairs:
        tja_file, audio_file = pair["tja"], pair["audio"]
        metadata = parse_tja(tja_file)
        audio_features = extract_audio_features(audio_file, metadata.get("offset", 0))

        for difficulty, data in metadata["courses"].items():
            dataset.append({
                "title": metadata["title"],
                "bpm": metadata["bpm"],
                "offset": metadata.get("offset", 0),
                "level": data["level"],
                "course": difficulty,
                "audio_features": audio_features,
                "notes": " ".join(data["notes"])
            })

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")  # 解决日文乱码


if __name__ == "__main__":
    DATASET_PATH = "dataset"  #   dataset
    OUTPUT_FILE = "data/taiko_dataset.jsonl"

    print(f"Processing dataset in {DATASET_PATH}...")
    prepare_dataset(DATASET_PATH, OUTPUT_FILE)
    print(f"Dataset saved to {OUTPUT_FILE}")

