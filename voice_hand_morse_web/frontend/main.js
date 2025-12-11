const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (event) => {
    const b64_audio = event.data;
    const audioBlob = new Blob([Uint8Array.from(atob(b64_audio), c=>c.charCodeAt(0))], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
};

document.getElementById("start").onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = async (e) => {
        if (e.data.size > 0) {
            const arrayBuffer = await e.data.arrayBuffer();
            ws.send(arrayBuffer);
        }
    };

    mediaRecorder.start();

    setTimeout(() => {
        mediaRecorder.stop();
    }, 5000); // 5秒録音
};
