from TTS.api import TTS
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import convolve, lfilter

# 1. TTSモデル読み込み（日本語 Tacotron2）
tts = TTS(model_name="tts_models/ja/kokoro/tacotron2-DDC")

# 2. HAL 9000風セリフ（間を意識）
text = "こんにちは、デイブ。\n調子はいかがですか。"

# 3. 音声生成（WAVファイル出力）
wav_path = "temp.wav"
tts.tts_to_file(text=text, file_path=wav_path)

# 4. 音声読み込み
y, sr = librosa.load(wav_path, sr=None)

# 5. ピッチを下げて低音化
y_low = librosa.effects.pitch_shift(y, sr, n_steps=-5)

# 6. 簡易リバーブ追加
reverb_ir = np.zeros(int(sr * 0.05))  # 50msリバーブ
reverb_ir[0] = 0.6
reverb_ir[int(sr*0.015)] = 0.3
reverb_ir[int(sr*0.03)] = 0.1
y_reverb = convolve(y_low, reverb_ir, mode='full')[:len(y_low)]

# 7. ディレイ（わずかなエコー）追加
delay_samples = int(sr * 0.07)  # 70ms遅延
echo_gain = 0.25
y_echo = np.copy(y_reverb)
y_echo[delay_samples:] += echo_gain * y_reverb[:-delay_samples]

# 8. ローパスフィルターで声を冷たく
b = [0.2, 0.2, 0.2, 0.2, 0.2]  # 単純移動平均フィルター
a = [1]
y_cold = lfilter(b, a, y_echo)

# 9. 音量正規化
y_norm = y_cold / np.max(np.abs(y_cold))

# 10. WAV出力
output_file = "hal9000_jp_final_effect.wav"
sf.write(output_file, y_norm, sr)

print(f"HAL 9000風日本語音声（無感情・不気味演出付き）を生成しました：{output_file}")
