import librosa
import pathlib
audio_path = r"C:\Users\陈逸楠\PycharmProjects\taiko_AI\dataset\24 karats TRIBE OF GOLD\24 karats TRIBE OF GOLD.ogg"
audio_path = str(pathlib.Path(audio_path).resolve())  # 处理路径
y, sr = librosa.load(audio_path, sr=22050)
print(f"音频加载成功！采样率: {sr}")
