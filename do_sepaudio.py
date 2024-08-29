from pipeline import build_audiosep, separate_audio
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_audiosep(
      config_yaml='config/audiosep_base.yaml', 
      checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt', 
      device=device)

audio_file = 'sniffle_sample.m4a'
text = "sniffle"
output_file='separated_audio.wav'

# AudioSep processes the audio at 32 kHz sampling rate  
separate_audio(model, audio_file, text, output_file, device)



import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile

# 音源AとBを読み込み
A, sr_A = librosa.load('sniffle_sample.m4a', sr=None)
B, sr_B = librosa.load('separated_audio.wav', sr=None)

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
wavfile.write('result.wav', sr_A, result.astype(np.float32))
