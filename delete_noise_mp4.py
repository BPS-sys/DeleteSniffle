from pipeline import build_audiosep, separate_audio
import torch
import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from moviepy.editor import *


input_file = 'movie.mp4'
noaudio_mp4_file = 'no_audio.mp4'
audio_file = 'audio.mp3'
text = "sniffle"
seped_file = 'sep.mp3'
deleted_noise_file = 'deleted_noise.mp3'
output_file = 'result.mp4'

# mp3ファイル生成
audio = AudioFileClip(input_file)
audio.write_audiofile(audio_file)


# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデル読み込み
model = build_audiosep(
      config_yaml='config/audiosep_base.yaml', 
      checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt', 
      device=device)

_, freeq = librosa.load(audio_file, sr=None)

# 音源分離を実行した48000Hzファイルを生成
separate_audio(model, audio_file, text, seped_file, freeq=freeq, device=device)


# 音源AとBを読み込み
A, sr_A = librosa.load(audio_file, sr=None)
B, sr_B = librosa.load(seped_file, sr=None)

print(sr_A)
print(sr_B)

# サンプルレートが一致していることを確認
if sr_A != sr_B:
    raise ValueError("音源AとBのサンプルレートが一致していません。")


# FFTを適用して周波数領域に変換
A_fft = np.fft.fft(A)
B_fft = np.fft.fft(B)

# 周波数領域で減算
result_fft = A_fft - B_fft

# IFFTで時間領域に戻す
result = np.fft.ifft(result_fft)

# 実部のみを取得
result = np.real(result)

# 音声ファイルとして保存
wavfile.write(deleted_noise_file, sr_A, result.astype(np.float32))


clip = VideoFileClip(input_file)
rewrite_audio = AudioFileClip(deleted_noise_file)
video_without_audio = clip.without_audio()

final_video = video_without_audio.set_audio(rewrite_audio)
final_video.write_videofile(output_file)

