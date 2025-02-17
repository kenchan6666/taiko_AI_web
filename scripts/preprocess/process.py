#!/usr/bin/env python3
"""
预处理太鼓达人数据：
- 遍历目录，查找 TJA 文件与对应音频文件
- 检测文件名编码问题，必要时重命名
- 解析 TJA 文件，提取元数据和谱面数据
- 使用 FFmpeg 将音频转换为 WAV 格式，并利用 librosa 提取 Mel 频谱特征
- 整理数据后保存为 JSONL 文件
"""

import os
import json
import librosa
import numpy as np
import chardet
import subprocess
import pathlib
import argparse
import logging
from typing import List, Dict, Optional

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

FFMPEG_PATH = "/usr/bin/ffmpeg"  # 请确认该路径正确


def find_tja_and_audio(root_folder: str) -> List[Dict[str, str]]:
    """
    遍历根目录，查找所有 TJA 文件和对应的音频文件（mp3, ogg, wav）。
    如果文件名存在乱码，则进行重命名。

    :param root_folder: 数据集根目录
    :return: 包含字典列表，每个字典包含 "tja" 和 "audio" 文件路径
    """
    dataset = []
    file_index = 1  # 用于重命名问题文件

    for subdir, _, files in os.walk(root_folder):
        tja_file = None
        audio_file = None

        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(".tja"):
                tja_file = file_path
            elif file.endswith((".mp3", ".ogg", ".wav")):
                audio_file = file_path

        if tja_file and audio_file:
            valid_tja_file = rename_if_needed(tja_file, file_index)
            valid_audio_file = rename_if_needed(audio_file, file_index)
            dataset.append({"tja": valid_tja_file, "audio": valid_audio_file})
            file_index += 1

    logging.info(f"找到 {len(dataset)} 组 TJA 和音频文件")
    return dataset


def rename_if_needed(file_path: str, index: int) -> str:
    """
    检查文件名是否为有效 UTF-8 编码，若有问题则重命名文件。

    :param file_path: 原始文件路径
    :param index: 重命名时使用的序号
    :return: 有效的文件路径（可能已重命名）
    """
    try:
        file_path.encode('utf-8')
        return file_path
    except UnicodeEncodeError:
        new_name = f"renamed_file_{index}{os.path.splitext(file_path)[1]}"
        new_path = os.path.join(os.path.dirname(file_path), new_name)
        os.rename(file_path, new_path)
        logging.info(f"重命名文件 {file_path} -> {new_path}")
        return new_path


def detect_encoding(file_path: str) -> Optional[str]:
    """
    检测文件编码

    :param file_path: 文件路径
    :return: 编码名称，如 'utf-8'；如果无法检测，则返回 None
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
        logging.debug(f"检测 {file_path} 编码: {encoding}")
        return encoding
    except Exception as e:
        logging.error(f"检测 {file_path} 编码失败: {e}")
        return None


def parse_tja(file_path: str) -> Dict:
    """
    解析 TJA 文件，提取元数据和谱面数据。

    :param file_path: TJA 文件路径
    :return: 包含标题、BPM、OFFSET、音频文件名及各 COURSE 下数据的字典
    """
    encoding = detect_encoding(file_path) or 'utf-8'
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.readlines()
    except UnicodeDecodeError:
        logging.warning(f"{file_path} 使用 UTF-8 fallback 编码")
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
    notes = []
    in_chart = False

    for line in content:
        line = line.strip()
        if line.startswith("TITLE:"):
            metadata["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("BPM:"):
            try:
                metadata["bpm"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                logging.warning(f"无法解析 BPM 值: {line}")
        elif line.startswith("OFFSET:"):
            try:
                metadata["offset"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                logging.warning(f"无法解析 OFFSET 值: {line}")
        elif line.startswith("WAVE:"):
            metadata["audio"] = line.split(":", 1)[1].strip()
        elif line.startswith("COURSE:"):
            current_course = line.split(":", 1)[1].strip()
            if current_course.lower() == "edit":
                logging.info(f"忽略 Edit 模式: {file_path}")
                current_course = None
            else:
                metadata["courses"][current_course] = {"level": None, "balloon": [], "notes": []}
        elif line.startswith("LEVEL:") and current_course:
            try:
                metadata["courses"][current_course]["level"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                logging.warning(f"LEVEL 解析错误: {line}")
        elif line.startswith("BALLOON:") and current_course:
            balloon_values = line.split(":", 1)[1].strip()
            if balloon_values:
                try:
                    metadata["courses"][current_course]["balloon"] = [int(x) for x in balloon_values.split(",") if
                                                                      x.strip().isdigit()]
                except Exception as e:
                    logging.warning(f"BALLOON 解析错误: {line}, {e}")
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


def normalize_path(file_path: str) -> str:
    """
    将文件路径转换为绝对路径

    :param file_path: 文件路径
    :return: 绝对路径字符串
    """
    return os.path.abspath(str(pathlib.Path(file_path).resolve()))


def convert_audio_to_wav(audio_path: str) -> Optional[str]:
    """
    使用 FFmpeg 将音频文件转换为 WAV 格式（采样率 22050）

    :param audio_path: 原始音频文件路径
    :return: WAV 文件路径，如果转换失败返回 None
    """
    audio_path = normalize_path(audio_path)
    wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
    if os.path.exists(wav_path):
        logging.info(f"WAV 文件已存在: {wav_path}")
        return wav_path

    try:
        subprocess.run([FFMPEG_PATH, "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "22050", wav_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"转换成功: {audio_path} -> {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg 转换失败: {audio_path}, 错误: {e}")
        return None


def extract_audio_features(audio_path: str, offset: float = 0) -> Optional[List[float]]:
    """
    提取音频文件的 Mel 频谱特征，并返回每个 Mel 频段的均值

    :param audio_path: 音频文件路径
    :param offset: 加载音频时的偏移量（秒）
    :return: Mel 特征向量列表；若处理失败返回 None
    """
    audio_path = normalize_path(audio_path)
    if audio_path.endswith((".mp3", ".ogg")):
        audio_path = convert_audio_to_wav(audio_path)
        if audio_path is None:
            return None
    try:
        y, sr = librosa.load(audio_path, sr=22050, offset=offset)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        # 返回每个 Mel 频段的均值（全局特征）；如果需要时序信息，可返回 mel 矩阵
        return mel.mean(axis=1).tolist()
    except Exception as e:
        logging.error(f"音频处理失败: {audio_path}, 错误: {e}")
        return None


def prepare_dataset(input_folder: str, output_file: str) -> None:
    """
    遍历 input_folder 中所有 TJA 和音频文件，解析数据、提取特征后保存为 JSONL 格式

    :param input_folder: 输入数据集根目录
    :param output_file: 输出 JSONL 文件路径
    """
    dataset = []
    file_pairs = find_tja_and_audio(input_folder)

    for pair in file_pairs:
        tja_file, audio_file = pair["tja"], pair["audio"]
        metadata = parse_tja(tja_file)
        audio_features = extract_audio_features(audio_file, metadata.get("offset", 0))

        # 针对每个 COURSE 生成一条数据记录
        for difficulty, course_data in metadata["courses"].items():
            record = {
                "title": metadata.get("title", ""),
                "bpm": metadata.get("bpm", None),
                "offset": metadata.get("offset", 0),
                "level": course_data.get("level", None),
                "course": difficulty,
                "audio_features": audio_features,
                "notes": " ".join(course_data.get("notes", []))
            }
            dataset.append(record)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"数据集保存成功：{output_file}")
    except Exception as e:
        logging.error(f"保存数据集失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="太鼓达人数据预处理")
    parser.add_argument("--input", required=True, help="输入数据集根目录")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    args = parser.parse_args()

    logging.info(f"开始处理数据集：{args.input}")
    prepare_dataset(args.input, args.output)
    logging.info("预处理完成。")


if __name__ == "__main__":
    main()

