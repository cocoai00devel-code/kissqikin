from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from modules.speech_recognizer import recognize_audio
from modules.tts import tts_pyttsx3_mem, tts_gtts_mem

app = FastAPI()

# 静的ファイルを /static にマウント
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# index.html を返す
@app.get("/")
async def read_index():
    return FileResponse("../frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_bytes()
        with open("temp.wav", "wb") as f:
            f.write(data)

        # 音声認識
        text = recognize_audio("temp.wav")

        # TTS（切替可能）
        use_online = False  # True: gTTS, False: pyttsx3
        if use_online:
            b64_audio = tts_gtts_mem(text)
        else:
            b64_audio = tts_pyttsx3_mem(text)

        await ws.send_text(b64_audio)
