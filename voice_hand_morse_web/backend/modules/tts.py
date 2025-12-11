import pyttsx3
from gtts import gTTS
import io
import base64
import soundfile as sf
import numpy as np

# -----------------------------
# gTTS（オンライン）メモリ上で生成
# -----------------------------
def tts_gtts_mem(text):
    mp3_fp = io.BytesIO()
    tts = gTTS(text=text, lang="ja")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode()

# -----------------------------
# pyttsx3（オフライン）メモリ上で生成
# -----------------------------
def tts_pyttsx3_mem(text):
    """
    pyttsx3で音声をリアルタイム取得（WAVファイル不要）
    """
    engine = pyttsx3.init()
    # バッファに保存するために wave と numpy を使う
    import tempfile
    import os
    import wave

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        engine.save_to_file(text, tmpfile.name)
        engine.runAndWait()
        tmpfile_path = tmpfile.name

    # WAV を読み込んで Base64 化
    data, samplerate = sf.read(tmpfile_path, dtype='int16')
    with io.BytesIO() as buf:
        sf.write(buf, data, samplerate, format='WAV')
        buf.seek(0)
        b64_audio = base64.b64encode(buf.read()).decode()

    os.remove(tmpfile_path)
    return b64_audio
