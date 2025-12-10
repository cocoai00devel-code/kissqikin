// // imakunAI.js #


// /// imakunAI.js #

// /* --------------------------------------------------------------------------------- */
// /* 1. Canvasアニメーションとレスポンシブ対応                                         */
// /* --------------------------------------------------------------------------------- */

// const canvas = document.getElementById("waveCanvas");
// const ctx = canvas.getContext("2d");

// // Canvasの初期サイズ設定は、resizeCanvas関数で処理するため、ここでは変数宣言に留める
// let bars = [];
// // 複数の定義があったため、BAR_COUNTに統一し、定数として定義し直す
// const BAR_COUNT = 40; 
// const BAR_WIDTH = 8;
// let dataArray;

// let animationFrameId;
// let isSpeaking = false;
// let isRecording = false;
// let currentWaveColor = 'rgba(50, 200, 255, 0.7)'; // 初期色を定義
// let rainbowHue = 0; // レインボーアニメーション用の色相を保持

// const WAVE_COLORS = {
//     default: 'rgba(50, 200, 255, 0.7)',
//     positive: 'rgba(50, 255, 50, 0.7)',
//     anger: 'rgba(255, 50, 50, 0.7)',
//     rage: 'rgba(150, 50, 255, 0.7)',
//     negative: 'rgba(50, 100, 255, 0.7)',
//     sadness: 'rgba(0, 0, 150, 0.7)'
// };

// /**
//  * バーのデータを再計算する
//  */
// function createBars() {
//     bars = [];
//     // 中央揃えでバーを配置
//     const startX = canvas.width / 2 - (BAR_COUNT * BAR_WIDTH) / 2;
//     for (let i = 0; i < BAR_COUNT; i++) {
//         bars.push({
//             x: startX + i * BAR_WIDTH,
//             height: 10,
//             color: "#00ffff" // 初期色は使用されないが、初期化
//         });
//     }
// }

// /**
//  * Canvasサイズをウィンドウにフィットさせ、バーを再計算する
//  */
// function resizeCanvas() {
//     canvas.width = window.innerWidth;
//     canvas.height = window.innerHeight;
//     // サイズ変更時にバーを再計算
//     createBars();
// }

// /**
//  * バーをアニメーションさせて描画する (drawWaveとdrawBarsのロジックを統合)
//  */
// function animateBars() {
//     // Canvasをクリア
//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // analyserが存在し、音声入力があれば周波数データを取得
//     if (analyser && dataArray && (isSpeaking || isRecording)) {
//         analyser.getByteFrequencyData(dataArray);
//     }

//     // 描画色を決定するロジック
//     let barColor = currentWaveColor;

//     // レインボーモードの場合、動的に色を計算
//     if (currentWaveColor === 'rainbow') {
//         rainbowHue = (rainbowHue + 3) % 360;
//         barColor = `hsla(${rainbowHue}, 100%, 70%, 0.9)`;
//     }

//     // 決定した色を塗りつぶし色として設定
//     ctx.fillStyle = barColor;
//     const currentWaveY = canvas.height / 2;

//     bars.forEach((bar, i) => {
//         let height = bar.height;

//         // 音声入力中またはAI応答中の場合、波形を動かす
//         if (isRecording && dataArray) {
//             // 周波数データを単純にマッピング
//             const dataIndex = Math.floor(i * (dataArray.length / BAR_COUNT));
//             const rawHeight = dataArray[dataIndex] || 0;
//             // 0-255を最大高さ（例: 200）にスケール
//             height = (rawHeight / 255) * 200 + 5; 
//         } else if (isSpeaking) {
//             // AI応答中は、シンプルなサイン波で波形を動かす
//             const waveAmplitude = 100;
//             const waveFrequency = 0.05;
//             const waveOffset = Math.sin(Date.now() * 0.005 + i * waveFrequency) * waveAmplitude;
//             height = 10 + Math.abs(waveOffset);
//         } else {
//             // 待機中は最小の高さ
//             height = 10;
//         }

//         // バーの高さと位置を更新
//         bars[i].height = height;

//         // 描画
//         // barWidthは定数、BAR_WIDTHも定数。一貫性のためにBAR_WIDTHを使用
//         ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
//     });

//     animationFrameId = requestAnimationFrame(animateBars);
// }

// // 【★ 修正点1: window.addEventListenerの重複を解消し、一つに統合 ★】
// window.addEventListener("load", () => {
//     resizeCanvas();
//     animateBars();
//     initAudioAndSTT(); // マイク初期化とSTTを自動で開始
//     setStandbyStatus();
//     document.getElementById('ui').style.opacity = 1;
// });
// window.addEventListener("resize", resizeCanvas);
// window.addEventListener("orientationchange", () => {
//     // 回転後の値が安定してから再計算
//     setTimeout(resizeCanvas, 300);
// });

// /* --------------------------------------------------------------------------------- */
// /* 2. 感情・色判定ロジック                                                           */
// /* --------------------------------------------------------------------------------- */

// /**
//  * AIの回答テキストに基づいて波形の色を変更する関数
//  * @param {string} responseText LLMからの回答テキスト
//  */
// function setWaveColorBasedOnResponse(responseText) {
//     const text = responseText.toLowerCase();

//     // 絵文字抽出関数（ローカルまたはグローバルで定義されている前提）
//     const extractEmojis = (t) => {
//         const emojiRegex = /\p{Emoji_Presentation}|\p{Emoji}\p{Emoji_Modifier}*|\p{Emoji_Component}|\u200d/gu;
//         const matches = t.match(emojiRegex);
//         return matches ? matches.join('') : '';
//     };

//     // 判定ロジックは提示された内容を維持（優先順位順）

//     // 1. 【本気の怒り・裏切り (紫)】
//     const rageKeywords = ['裏切り', '許さない', '報復', 'どうしてくれる', '絶交', '失望'];
//     const rageEmojis = ['😡', '😠', '🤬', '👿', '😾', '💀', '🔪', '💣'];
//     if (rageKeywords.some(k => text.includes(k)) || rageEmojis.some(e => text.includes(e))) {
//         currentWaveColor = WAVE_COLORS.rage;
//         console.log("波形の色を【本気の怒り・裏切り】の紫に変更しました。");
//         return;
//     }

//     // 5. 【最高にハッピー (レインボー)】
//     const superHappyKeywords = ['最高にハッピー', '神', '究極', 'パーフェクト', '完璧', '奇跡', '感無量', 'レジェンド'];
//     const superHappyEmojis = ['🤩', '✨', '🥳', '💯', '👑', '🥇', '🚀', '🌈', '🎉🎉🎉'];
//     if (superHappyKeywords.some(k => text.includes(k)) || superHappyEmojis.some(e => text.includes(e))) {
//         currentWaveColor = 'rainbow';
//         console.log("波形の色を【最高にハッピー】のレインボーに変更しました。");
//         return;
//     }

//     // 2. 【お怒り (赤)】
//     const angerKeywords = ['怒り', 'ふざけるな', 'やめろ', 'だめだ', '不可能だ', '違います', '否定', 'ありえない'];
//     const angerEmojis = ['😤', '💢', '🔥', '💥', '👹', '😫', '😩'];
//     if (angerKeywords.some(k => text.includes(k)) || angerEmojis.some(e => text.includes(e))) {
//         currentWaveColor = WAVE_COLORS.anger;
//         console.log("波形の色を【お怒り】の赤に変更しました。");
//         return;
//     }

//     // 3. 【悲しい・号泣 (濃いブルー)】
//     const sadnessKeywords = ['悲しい', '泣く', 'ごめんなさい', 'つらい', '寂しい', '涙', '耐えられない', '最悪', 'しんどい'];
//     const sadnessEmojis = ['😭', '😢', '🥺', '💧', '😥', '💔', '🌧️', '☔'];
//     if (sadnessKeywords.some(k => text.includes(k)) || sadnessEmojis.some(e => text.includes(e))) {
//         currentWaveColor = WAVE_COLORS.sadness;
//         console.log("波形の色を【悲しい・号泣】の濃いブルーに変更しました。");
//         return;
//     }

//     // 4. 【ネガティブ (ブルー)】
//     const negativeKeywords = ['エラー', '失敗', 'できません', '警告', '問題', '懸念', '不明', '確認', '無理', '難しい'];
//     const negativeEmojis = ['😞', '😟', '😨', '🥶', '😰', '😵'];
//     if (negativeKeywords.some(k => text.includes(k)) || negativeEmojis.some(e => text.includes(e))) {
//         currentWaveColor = WAVE_COLORS.negative;
//         console.log("波形の色を【ネガティブ】のブルーに変更しました。");
//         return;
//     }

//     // 6. 【ポジティブ (緑)】
//     const positiveKeywords = ['ありがとう', '成功', '完了', '問題ありません', '良い', 'できます', '素晴らしい', '助かる', '了解', 'OK', 'ハッピー'];
//     const positiveEmojis = ['😄', '😊', '😆', '👍', '👏', '✅', '🌟'];
//     if (positiveKeywords.some(k => text.includes(k)) || positiveEmojis.some(e => text.includes(e))) {
//         currentWaveColor = WAVE_COLORS.positive;
//         console.log("波形の色を【ポジティブ】の緑に変更しました。");
//         return;
//     }

//     // どの条件にも合致しない場合はデフォルト色に戻す
//     currentWaveColor = WAVE_COLORS.default;
//     console.log("波形の色をデフォルトの水色に戻しました。");
// }


// /* --------------------------------------------------------------------------------- */
// /* 3. 機密保持/開発者ツールの無効化 (重複を解消し整理)                                */
// /* --------------------------------------------------------------------------------- */

// // 1. 右クリック（コンテキストメニュー）を禁止する
// document.addEventListener('contextmenu', function (e) {
//     e.preventDefault();
//     console.log("右クリックは禁止されています。");
//     return false;
// });
// document.body.addEventListener('contextmenu', function (e) {
//     e.preventDefault();
//     return false;
// });

// // 2. キーボードショートカットを禁止する
// // 【★ 修正点2: document.onkeydownの重複定義を解消し、一つのロジックに統合 ★】
// document.onkeydown = function (e) {
//     const key = e.key;
//     const lowerKey = key.toLowerCase();

//     // F12キー (開発者ツール)
//     if (key === 'F12' || e.keyCode === 123) {
//         e.preventDefault();
//         return false;
//     }

//     // Ctrl/Cmd/Option/Alt キー状態のチェックを容易にする
//     const isCmdOrCtrl = e.ctrlKey || e.metaKey;
//     const isShift = e.shiftKey;
//     const isAltOrOption = e.altKey;

//     // 開発者ツールのショートカット (I, J, C)
//     if (
//         (isCmdOrCtrl && isShift && (lowerKey === 'i' || lowerKey === 'j' || lowerKey === 'c')) || // Ctrl/Cmd + Shift + I/J/C
//         (e.metaKey && isAltOrOption && (lowerKey === 'i' || lowerKey === 'j' || lowerKey === 'c')) // Cmd + Option + I/J/C (macOS)
//     ) {
//         e.preventDefault();
//         return false;
//     }

//     // その他、コンテンツ保護のためのショートカット (U, S, P)
//     if (isCmdOrCtrl && (lowerKey === 'u' || lowerKey === 's' || lowerKey === 'p')) {
//         e.preventDefault();
//         return false;
//     }
// };

// /* --------------------------------------------------------------------------------- */
// /* 4. 音声読み上げ/認識/API連携関連                                                  */
// /* --------------------------------------------------------------------------------- */

// // DOM要素の取得
// const statusArea = document.getElementById("status-area");
// const sendBtn = document.getElementById("sendBtn");
// const input = document.getElementById("messageInput");
// // const transcriptBox = document.getElementById('transcript'); // 未使用のため削除
// const ui = document.getElementById('ui');
// const tapArea = document.getElementById('tapArea');

// // API設定
// const LLM_API_URL = "https://atjmuwnwmtjw-hello.hf.space/llm/generate";
// const MQTT_API_URL = "https://atjmuwnwmtjw-hello.hf.space/iot/control";

// // 状態管理変数
// const synth = window.speechSynthesis;
// let audioContext, analyser, mediaStream;
// let recognition = null;
// let currentTextToSpeak = '';

// // --- ヘルパー関数 (色の補間) ---
// function hexToRgb(hex) {
//     const bigint = parseInt(hex.slice(1), 16);
//     const r = (bigint >> 16) & 255;
//     const g = (bigint >> 8) & 255;
//     const b = bigint & 255;
//     return [r, g, b];
// }

// function rgbToHex(r, g, b) {
//     const toHex = (c) => ('0' + Math.max(0, Math.min(255, c)).toString(16)).slice(-2);
//     return '#' + toHex(Math.round(r)) + toHex(Math.round(g)) + toHex(Math.round(b));
// }

// function startColorTransition(startColor, endColor, duration = 2000) {
//     const startTime = performance.now();
//     const startRgb = hexToRgb(startColor);
//     const endRgb = hexToRgb(endColor);

//     function interpolate(currentTime) {
//         const elapsed = currentTime - startTime;
//         const progress = Math.min(1, elapsed / duration);

//         const r = startRgb[0] + (endRgb[0] - startRgb[0]) * progress;
//         const g = startRgb[1] + (endRgb[1] - startRgb[1]) * progress;
//         const b = startRgb[2] + (endRgb[2] - startRgb[2]) * progress;

//         const currentColor = rgbToHex(r, g, b);

//         statusArea.style.color = currentColor;
//         statusArea.style.boxShadow = `0 0 20px ${currentColor}80`;

//         if (progress < 1) {
//             requestAnimationFrame(interpolate);
//         }
//     }
//     requestAnimationFrame(interpolate);
// }

// /* ---------- UI helpers ---------- */

// function updateStatus(message, color = '#00ffff') {
//     statusArea.innerHTML = message;
//     statusArea.style.color = color;
//     statusArea.style.boxShadow = `0 0 20px ${color}80`;
// }

// function setStandbyStatus() {
//     const standbyMsg = `
//         イマジナリーナンバー
//         通称GAIイマさんAI
//         AIアシスタント待機中...
//     `;
//     updateStatus(standbyMsg.trim(), '#00ffff');
// }

// /* ---------- TTS (Speech Synthesis) ---------- */

// // 【外部定義】絵文字抽出関数
// const extractEmojis = (text) => {
//     const emojiRegex = /\p{Emoji_Presentation}|\p{Emoji}\p{Emoji_Modifier}*|\p{Emoji_Component}|\u200d/gu;
//     const matches = text.match(emojiRegex);
//     return matches ? matches.join('') : '';
// };


// /**
//  * LLM応答など、AIからの正式な応答を読み上げ、終了後にSTTを再起動する
//  * @param {string} text 読み上げるテキスト
//  */
// function speak(text) {
//     if (!text) return;

//     currentTextToSpeak = text;

//     if (synth.speaking) synth.cancel();

//     isSpeaking = true;

//     // LLM応答に基づいて波形の色を設定
//     setWaveColorBasedOnResponse(text);

//     const u = new SpeechSynthesisUtterance(text);
//     u.lang = 'ja-JP';
//     u.rate = 1.0;

//     u.onstart = () => {
//         const display = text.length > 20 ? text.substring(0, 20) + '...' : text;
//         // 【★ 修正点3: onstart内のextractEmojisを修正したロジックを採用 ★】
//         const onlyEmojis = extractEmojis(text); 
        
//         const formattedStatus = `
//         ---==(_____[　イマジナリーナンバー通称GAIイマさんAI応答:  ?&!! ${onlyEmojis}　]_____)==--- __(V._.V)__
//                       「${display}」
//         `;
//         // TTS中は色をデフォルトの応答色に設定（波形の色とは別）
//         updateStatus(formattedStatus.trim(), '#00ffaa');
//     };
    
//     u.onend = () => {
//         isSpeaking = false;
//         currentTextToSpeak = '';
//         setStandbyStatus();
//         input.value = '';
//         // TTS終了後、波形の色をデフォルトに戻す
//         currentWaveColor = WAVE_COLORS.default; // この行はspeakSentenceとの兼ね合いを考慮し維持

//         // TTS終了後、STTが停止していれば自動で再起動を試みる
//         if (recognition && !isRecording) {
//             try {
//                 recognition.start();
//             } catch (e) {
//                 console.warn('Recognition restart failed after TTS:', e);
//             }
//         }
//     };
//     u.onerror = (e) => {
//         console.error('TTS error:', e);
//         isSpeaking = false;
//         currentTextToSpeak = '';
//         setStandbyStatus();
//         input.value = '';
//         currentWaveColor = WAVE_COLORS.default;
//     };

//     synth.speak(u);
// }

// /**
//  * テキスト入力時の即時プレビュー用読み上げ関数
//  * @param {string} text 読み上げるテキスト
//  */
// function speakSentence(text) {
//     if (text.trim() === '' || text === currentTextToSpeak) {
//         return;
//     }

//     if (synth.speaking) {
//         synth.cancel();
//     }

//     currentTextToSpeak = text;

//     const utterance = new SpeechSynthesisUtterance(text);
//     utterance.lang = 'ja-JP';
//     utterance.rate = 1.0;

//     utterance.onstart = () => {
//         isSpeaking = true;
//         const display = text.length > 20 ? text.substring(0, 20) + '...' : text;
//         updateStatus(`文章を読み上げ中: 「${display}」`, '#00ffaa');
//         // プレビュー読み上げ中も波形を動かすため、一時的にcurrentWaveColorをポジティブに設定
//         currentWaveColor = WAVE_COLORS.positive;
//     };

//     utterance.onend = () => {
//         isSpeaking = false;
//         setStandbyStatus();
//         currentWaveColor = WAVE_COLORS.default; // 終了したらデフォルトに戻す
//     };

//     utterance.onerror = (event) => {
//         console.error('Speech Synthesis Error:', event);
//         isSpeaking = false;
//         updateStatus('読み上げエラーが発生しました', '#ff0000');
//         currentWaveColor = WAVE_COLORS.default;
//     };

//     synth.speak(utterance);
// }

// /* ---------- Speech Recognition (Browser STT) & Audio Init ---------- */

// function restartRecognition() {
//     isRecording = false;

//     // TTSが動作中でなければ、待機状態に戻す
//     if (!synth.speaking) {
//         isSpeaking = false;
//         setStandbyStatus();
//     }

//     setTimeout(() => {
//         try {
//             // 既に認識が開始されている場合は何もしない
//             if (!isRecording && !synth.speaking && recognition) recognition.start();
//         } catch (e) {
//             if (e.name !== 'InvalidStateError') {
//                 console.warn('Recognition start failed:', e);
//             }
//         }
//     }, 500);
// }

// function startBrowserRecognition() {
//     if (isRecording) return;

//     if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
//         updateStatus('Error: Speech Recognition not supported in this browser.', '#ff0000');
//         return;
//     }

//     if (recognition) {
//         recognition.stop();
//         recognition = null;
//     }

//     recognition = new (window.webkitSpeechRecognition || window.SpeechRecognition)();
//     recognition.continuous = false;
//     recognition.interimResults = true;
//     recognition.lang = 'ja-JP';

//     recognition.onstart = () => {
//         isRecording = true;
//         isSpeaking = true; // 録音中は波形を動かすために一時的にtrue
//         const standbyMsg = `
//             Listening...
//             話しかけてください...！
//         `;
//         updateStatus(standbyMsg.trim(), '#ffff00');
//         startColorTransition('#ffff00', '#00ffaa', 2000);
//         input.value = '';
//         if (synth.speaking) synth.cancel();
//     };

//     recognition.onresult = (event) => {
//         let interimTranscript = '';
//         let finalTranscript = '';

//         for (let i = event.resultIndex; i < event.results.length; ++i) {
//             if (event.results[i].isFinal) {
//                 finalTranscript += event.results[i][0].transcript;
//             } else {
//                 interimTranscript += event.results[i][0].transcript;
//             }
//         }
//         input.value = finalTranscript || interimTranscript;
//     };
    
//     // 【★ 修正点4: 冗長なrestartRecognitionの定義を削除し、グローバル関数に依存させる ★】
//     recognition.onend = () => {
//         isRecording = false;

//         // TTSが動作していない場合に限り isSpeaking を false に
//         if (!synth.speaking) {
//             isSpeaking = false;
//         }

//         const finalPrompt = input.value.trim();

//         if (finalPrompt && finalPrompt.length > 1 && !finalPrompt.startsWith("話しかけてください") && !finalPrompt.startsWith("イマジナリーナンバー 通称GAIイマさんAI応答:")) {
//             updateStatus('Processing response...', '#00ffaa');
//             // LLM処理中にSTTが自動で再起動しないように、.finallyでrestartRecognitionを呼ぶ
//             processRecognitionResult(finalPrompt).finally(() => {
//                 // TTSが終了した後に再起動させる (speak関数内のonendでも実施されるため冗長ではあるが念のため)
//                 if (!synth.speaking) {
//                     restartRecognition();
//                 }
//             });
//         } else {
//             // 発話がなかったか、短すぎた場合
//             input.value = '';
//             restartRecognition();
//         }
//     };

//     recognition.onerror = (event) => {
//         isRecording = false;
//         // isSpeaking = false;
//         console.error('Speech Recognition Error:', event.error);

//         if (event.error !== 'not-allowed' && event.error !== 'aborted') {
//             restartRecognition();
//         } else if (event.error === 'aborted') {
//             restartRecognition();
//         } else {
//             updateStatus('Error: Microphone permission denied or failed.', '#ff0000');
//         }
//     };

//     try {
//         recognition.start();
//     } catch (e) {
//         console.warn('Initial recognition start failed:', e);
//     }
// }

// async function initAudioAndSTT() {
//     if (analyser) {
//         startBrowserRecognition();
//         return;
//     }
//     updateStatus('Requesting microphone access...');

//     try {
//         audioContext = new (window.AudioContext || window.webkitAudioContext)();
//         analyser = audioContext.createAnalyser();
//         analyser.fftSize = 2048;

//         dataArray = new Uint8Array(analyser.frequencyBinCount);

//         mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
//         const sourceNode = audioContext.createMediaStreamSource(mediaStream);

//         sourceNode.connect(analyser);

//         startBrowserRecognition();

//         updateStatus('Listening...', '#ffff00');
//     } catch (e) {
//         console.error('Audio initialization failed:', e);
//         updateStatus('Error: Microphone access denied or failed to initialize.', '#ff0000');
//     }
// }

// /**
//  * FastAPI/MQTTバックエンドにコマンドを送信する関数
//  */
// async function sendIoTCommand(command) {
//     updateStatus(`Executing IoT command: ${command}...`, '#00ffaa');

//     try {
//         const response = await fetch(MQTT_API_URL, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ command: command })
//         });

//         const data = await response.json();

//         if (response.ok) {
//             const successMsg = `承知しました。${command === 'ON' ? '電気をつけました' : '電気を消しました'}。`;
//             speak(successMsg);
//         } else {
//             const detail = data.detail || "サーバーエラー";
//             const errorMsg = `エラーが発生しました。IoTコマンド '${command}' の実行に失敗しました。詳細: ${detail}`;
//             speak(errorMsg);
//         }
//     } catch (error) {
//         const networkErrorMsg = `🔴 ネットワークエラー: IoTバックエンドサーバーに接続できません (${error.message})`;
//         speak(networkErrorMsg);
//     }
// }

// /* ---------- 統合されたメイン処理関数 (IoT or LLM) ---------- */

// async function processRecognitionResult(finalPrompt) {
//     // 1. IoTコマンドの判定と振り分け
//     const lowerPrompt = finalPrompt.toLowerCase();
//     let iotCommand = null;

//     if ((lowerPrompt.includes('ライト') || lowerPrompt.includes('電気')) && (lowerPrompt.includes('つけ') || lowerPrompt.includes('オン') || lowerPrompt.includes('点け'))) {
//         iotCommand = 'ON';
//     } else if ((lowerPrompt.includes('ライト') || lowerPrompt.includes('電気')) && (lowerPrompt.includes('けし') || lowerPrompt.includes('オフ') || lowerPrompt.includes('消し'))) {
//         iotCommand = 'OFF';
//     }

//     if (iotCommand) {
//         await sendIoTCommand(iotCommand);
//         return;
//     }

//     // 2. LLM応答生成（IoTコマンドでなかった場合）
//     await generateAndSpeakResponse(finalPrompt);
// }

// /* ---------- LLM (Gemini) API & TTS 連携 ---------- */

// async function generateAndSpeakResponse(prompt) {
//     updateStatus('Generating response (via FastAPI)...', '#00ffaa');

//     // ユーザープロンプトから不要なプレフィックスを除去（念のため）
//     const cleanedPrompt = prompt.replace(/^イマジナリーナンバー 通称GAIイマさんAI応答:\s*/, '').trim();
//     if (!cleanedPrompt) {
//         speak("すみません、何も聞こえませんでした。もう一度話しかけてください。");
//         return;
//     }

//     const systemInstruction = "あなたは「イマジナリーナンバー 通称GAIイマさん」という名前のKS-903model8800-a1-90dという音声アシスタントです。ユーザーの質問に日本語で、簡潔かつ丁寧に答えてください。";

//     const payload = {
//         prompt: cleanedPrompt,
//         contents: [{ parts: [{ text: cleanedPrompt }] }],
//         systemInstruction: { parts: [{ text: systemInstruction }] },
//         tools: [{ "google_search": {} }],
//     };

//     const MAX_RETRIES = 3;
//     let responseText = "エラーが発生しました。イマジナリーナンバー 通称GAIイマさんAIのKS-903model8800-a1-90d応答を取得できませんでした。";

//     for (let i = 0; i < MAX_RETRIES; i++) {
//         try {
//             const response = await fetch(LLM_API_URL, {
//                 method: 'POST',
//                 headers: { 'Content-Type': 'application/json' },
//                 body: JSON.stringify(payload)
//             });

//             if (!response.ok) {
//                 const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status} Error.` }));
//                 throw new Error(`FastAPI Error! Status: ${response.status}. Detail: ${errorData.detail}`);
//             }

//             const result = await response.json();

//             if (result && result.text) {
//                 responseText = result.text;
//                 break;
//             } else {
//                 throw new Error("Empty response or invalid JSON structure from FastAPI.");
//             }

//         } catch (e) {
//             console.error(`FastAPI call error on attempt ${i + 1}:`, e);
//             if (i === MAX_RETRIES - 1) {
//                 responseText = "エラーが発生しました。イマジナリーナンバー 通称GAIイマさんAIKS-903model8800-a1-90dの応答を取得できませんでした。Generaltebバックエンドサーバー (ポート8001) の実行状態とAPIキーを確認してください。";
//             } else {
//                 const delay = 2 ** i * 1000 + Math.random() * 500;
//                 await new Promise(resolve => setTimeout(resolve, delay));
//             }
//         }
//     }

//     updateStatus('Speaking response...', '#00ffaa');
//     speak(responseText);

//     return Promise.resolve();
// }

// /* ---------- イベントハンドラの統合と定義 ---------- */

// // テキスト入力欄のイベントを追加 (Enterキーで処理)
// input.addEventListener('keydown', (e) => {
//     if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey) {
//         e.preventDefault();

//         const textPrompt = input.value.trim();

//         if (textPrompt) {
//             // 音声認識が実行中の場合は強制停止
//             if (recognition && isRecording) {
//                 recognition.stop();
//             }
//             // TTSをキャンセル（即時読み上げを停止）
//             if (synth.speaking) synth.cancel();

//             updateStatus('Processing text input...', '#ffff00');

//             // LLM処理を実行
//             processRecognitionResult(textPrompt).catch(error => {
//                 console.error("Text input processing failed:", error);
//             });
//         }
//     }
// });

// // テキスト入力のたびに現在の内容を読み上げる機能の追加 (TTS即時プレビュー)
// input.addEventListener('input', (event) => {
//     const currentText = input.value.trim();

//     // 音声認識が実行中でない、かつ、AIが応答中でない場合にのみ実行
//     // かつ、現在のテキストが読み上げ中のテキストと異なる場合
//     if (!isRecording && !isSpeaking && currentText.length > 0 && currentText !== currentTextToSpeak) {
//         speakSentence(currentText);
//     } else if (currentText.length === 0 && synth.speaking) {
//         // テキストが全て削除され、かつ読み上げ中の場合はキャンセルして待機状態に戻す
//         synth.cancel();
//         isSpeaking = false;
//         setStandbyStatus();
//     }
// });

// // リセットボタンの機能 (STTとTTSの強制停止と再起動)
// sendBtn.addEventListener("click", () => {
//     if (recognition) {
//         recognition.stop();
//         recognition = null;
//         // isRecordingはonendでfalseになるが、即時リセットのため手動でも設定
//         isRecording = false;
//     }
//     if (synth.speaking) synth.cancel();

//     // isSpeakingを強制的にfalseに
//     isSpeaking = false;
//     currentWaveColor = WAVE_COLORS.default;

//     initAudioAndSTT();
//     updateStatus('リセットしました。マイク入力を開始しています...');
// });


// // UI トグル機能 (画面タップ)
// let uiVisible = true;
// tapArea.addEventListener('click', (e) => {
//     // リセットボタンへのタップは無視
//     if (e.target.closest('#input-controls')) {
//         return;
//     }

//     uiVisible = !uiVisible;
//     if (uiVisible) {
//         ui.style.opacity = 1;
//     } else {
//         ui.style.opacity = 0;
//     }
// });
    
// //         /* ---------- Canvasアニメーション関連 ---------- */
// //         const canvas = document.getElementById("waveCanvas");
// //         const ctx = canvas.getContext("2d");
// //         canvas.width = window.innerWidth;
// //         canvas.height = window.innerHeight;

// //         let bars = [];
// //         const barCount = 40;
// //         const barWidth = 8;
// //         // const waveY = canvas.height / 2;　// 初期値の canvas.height を使用
// //         let dataArray; 
// //         
// //         let animationFrameId;
// //         let isSpeaking = false; 
// //         let isRecording = false; 
// //         let rainbowHue = 0; // レインボーアニメーション用の色相を保持　// let rainbowHue = 0;
// // // let bars = [];      // バーのデータを保持する配列
// // // const barCount = 100; // バーの数（任意の数値）
// // // const barWidth = 8; // バーの幅（任意の数値）
// // const WAVE_COLORS = {
// //     // デフォルト: 水色
// //     default: 'rgba(50, 200, 255, 0.7)', 
// //     // ポジティブな回答: 緑
// //     positive: 'rgba(50, 255, 50, 0.7)', 
// //     // お怒り: 赤
// //     anger: 'rgba(255, 50, 50, 0.7)', 
// //     // 本気の怒り/裏切り: 紫
// //     rage: 'rgba(150, 50, 255, 0.7)', 
// //     // ネガティブ (一般的): 明るめのブルー
// //     negative: 'rgba(50, 100, 255, 0.7)', 
// //     // 悲しい/号泣: 濃いブルー
// //     sadness: 'rgba(0, 0, 150, 0.7)'
// // };
// //  // ... (既存の描画ロジック：波形の計算など) ...

// //  function createBars() {
// //             bars = [];
// //             const startX = canvas.width / 2 - (barCount * barWidth) / 2;
// //             for (let i = 0; i < barCount; i++) {
// //                 bars.push({
// //                     x: startX + i * barWidth,
// //                     height: 10,
// //                     color: "#00ffff"
// //                 });
// //             }
// //         }
   
// // /* ----------- スマホ回転時にもCanvasをフィットさせる ----------- */
// // function resizeCanvas() {
// //     const canvas = document.getElementById("waveCanvas");
// //     canvas.width = window.innerWidth;
// //     canvas.height = window.innerHeight;
// //     // 【★ 修正点1: サイズ変更時にバーを再計算 ★】
// //     createBars();
// // }

// // /**
// //  * バーをアニメーションさせて描画する
// //  */
// // function animateBars() {
// //     // Canvasをクリア
// //     ctx.clearRect(0, 0, canvas.width, canvas.height);

// //     // analyserが存在し、音声入力があれば周波数データを取得
// //     if (analyser && dataArray && (isSpeaking || isRecording)) {
// //         analyser.getByteFrequencyData(dataArray);
// //         // dataArrayの中の特定のインデックスの値を、バーの高さにマッピングするロジックを実装
// //         // 今回はシンプルなランダム波形（デモ用）または実際の音声データを使用
// //     }

// //     // 描画色を決定するロジック
// //     let barColor = currentWaveColor;

// //     // レインボーモードの場合、動的に色を計算
// //     if (currentWaveColor === 'rainbow') {
// //         rainbowHue = (rainbowHue + 3) % 360;
// //         barColor = `hsla(${rainbowHue}, 100%, 70%, 0.9)`;
// //     }

// //     ctx.fillStyle = barColor;
// //     const currentWaveY = canvas.height / 2;
// //     const centerOffset = BAR_COUNT / 2;

// //     bars.forEach((bar, i) => {
// //         let height = bar.height;

// //         // 音声入力中またはAI応答中の場合、波形を動かす
// //         if (isRecording && dataArray) {
// //             // 周波数データを単純にマッピング
// //             const dataIndex = Math.floor(i * (dataArray.length / BAR_COUNT));
// //             const rawHeight = dataArray[dataIndex] || 0;
// //             // 0-255を最大高さ（例: 200）にスケール
// //             height = (rawHeight / 255) * 200 + 5; 
// //         } else if (isSpeaking) {
// //             // AI応答中は、シンプルなサイン波で波形を動かす
// //             const waveAmplitude = 100;
// //             const waveFrequency = 0.05;
// //             const waveOffset = Math.sin(Date.now() * 0.005 + i * waveFrequency) * waveAmplitude;
// //             height = 10 + Math.abs(waveOffset);
// //         } else {
// //             // 待機中は最小の高さ
// //             height = 10;
// //         }

// //         // バーの高さと位置を更新
// //         bars[i].height = height;

// //         // 描画
// //         ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
// //     });

// //     animationFrameId = requestAnimationFrame(animateBars);
// // }


// // // 【★ 修正点2: イベントリスナーを統合して重複を解消 ★】
// // window.addEventListener("load", () => {
// //     resizeCanvas();
// //     animateBars();
// //     initAudioAndSTT(); // マイク初期化とSTTを自動で開始
// //     setStandbyStatus();
// //     document.getElementById('ui').style.opacity = 1;
// // });
// // window.addEventListener("resize", resizeCanvas);
// // window.addEventListener("orientationchange", () => {
// //     setTimeout(resizeCanvas, 300);
// // });


// // window.addEventListener("load", resizeCanvas);
// // window.addEventListener("resize", resizeCanvas);
// // window.addEventListener("orientationchange", () => {
// //     setTimeout(resizeCanvas, 300); // 回転後の値が安定してから再計算
// // });


// // /* --------------------------------------------------------------------------------- */
// // /* 2. 感情・色判定ロジック                                                           */
// // /* --------------------------------------------------------------------------------- */
// //  /**
// //  * AIの回答テキストに基づいて波形の色を変更する関数 (柔軟なキーワード＆絵文字対応)
// //  * @param {string} responseText LLMからの回答テキスト
// //  */
// // function setWaveColorBasedOnResponse(responseText) {
// //     const text = responseText.toLowerCase();

// //     // 絵文字抽出関数（グローバルに定義されていることを前提）
// //     const extractEmojis = (t) => {
// //         const emojiRegex = /\p{Emoji_Presentation}|\p{Emoji}\p{Emoji_Modifier}*|\p{Emoji_Component}|\u200d/gu;
// //         const matches = t.match(emojiRegex);
// //         return matches ? matches.join('') : '';
// //         };

// //     // 1. 【本気の怒り・裏切り (紫)】：最も深刻なキーワードを優先 (変更なし)
// //     const rageKeywords = ['裏切り', '許さない', '報復', 'どうしてくれる', '絶交', '失望'];
// //     const rageEmojis = ['😡', '😠', '🤬', '👿', '😾', '💀', '🔪', '💣']; 
// //     if (rageKeywords.some(k => text.includes(k)) || rageEmojis.some(e => text.includes(e))) {
// //         currentWaveColor = WAVE_COLORS.rage; 
// //         console.log("波形の色を【本気の怒り・裏切り】の紫に変更しました。");
// //         return;
// //     }

// //     // 2. 【お怒り (赤)】： (変更なし)
// //     const angerKeywords = ['怒り', 'ふざけるな', 'やめろ', 'だめだ', '不可能だ', '違います', '否定', 'ありえない'];
// //     const angerEmojis = ['😤', '💢', '🔥', '💥', '👹', '😫', '😩']; 
// //     if (angerKeywords.some(k => text.includes(k)) || angerEmojis.some(e => text.includes(e))) {
// //         currentWaveColor = WAVE_COLORS.anger; 
// //         console.log("波形の色を【お怒り】の赤に変更しました。");
// //         return;
// //     }

// //     // 3. 【悲しい・号泣 (濃いブルー)】： (変更なし)
// //     const sadnessKeywords = ['悲しい', '泣く', 'ごめんなさい', 'つらい', '寂しい', '涙', '耐えられない', '最悪', 'しんどい'];
// //     const sadnessEmojis = ['😭', '😢', '🥺', '💧', '😥', '💔', '🌧️', '☔']; 
// //     if (sadnessKeywords.some(k => text.includes(k)) || sadnessEmojis.some(e => text.includes(e))) {
// //         currentWaveColor = WAVE_COLORS.sadness; 
// //         console.log("波形の色を【悲しい・号泣】の濃いブルーに変更しました。");
// //         return;
// //     }

// //     // 4. 【ネガティブ (ブルー)】： (変更なし)
// //     const negativeKeywords = ['エラー', '失敗', 'できません', '警告', '問題', '懸念', '不明', '確認', '無理', '難しい'];
// //     const negativeEmojis = ['😞', '😟', '😨', '🥶', '😰', '😵']; 
// //     if (negativeKeywords.some(k => text.includes(k)) || negativeEmojis.some(e => text.includes(e))) {
// //         currentWaveColor = WAVE_COLORS.negative; 
// //         console.log("波形の色を【ネガティブ】のブルーに変更しました。");
// //         return;
// //     }
    
// //     // 5. 【最高にハッピー (レインボー)】：新しい判定ロジック
// //     const superHappyKeywords = ['最高にハッピー', '神', '究極', 'パーフェクト', '完璧', '奇跡', '感無量', 'レジェンド'];
// //     const superHappyEmojis = ['🤩', '✨', '🥳', '💯', '👑', '🥇', '🚀', '🌈', '🎉🎉🎉']; 
// //     if (superHappyKeywords.some(k => text.includes(k)) || superHappyEmojis.some(e => text.includes(e))) {
// //         currentWaveColor = 'rainbow'; // 描画関数が処理する特別な値
// //         console.log("波形の色を【最高にハッピー】のレインボーに変更しました。");
// //         return;
// //     }

// //     // 6. 【ポジティブ (緑)】：一般的な肯定的 (変更なし)
// //     const positiveKeywords = ['ありがとう', '成功', '完了', '問題ありません', '良い', 'できます', '素晴らしい', '助かる', '了解', 'OK', 'ハッピー'];
// //     const positiveEmojis = ['😄', '😊', '😆', '👍', '👏', '✅', '🌟'];
// //     if (positiveKeywords.some(k => text.includes(k)) || positiveEmojis.some(e => text.includes(e))) {
// //         currentWaveColor = WAVE_COLORS.positive; 
// //         console.log("波形の色を【ポジティブ】の緑に変更しました。");
// //         return;
// //     }

// //     // どの条件にも合致しない場合はデフォルト色に戻す
// //     currentWaveColor = WAVE_COLORS.default; 
// //     console.log("波形の色をデフォルトの水色に戻しました。");
// // }




// //     function drawWave() {
// //     // ... (既存のコード) ...

// //     ctx.clearRect(0, 0, canvas.width, canvas.height); // 波形のクリア
// //     ctx.beginPath();
// //     ctx.lineWidth = 4; // 波の太さ

// //     // 【この部分を修正】 currentWaveColorが'rainbow'かチェックする
// //     if (currentWaveColor === 'rainbow') {
// //         // レインボーアニメーションを有効にする
// //         rainbowHue = (rainbowHue + 3) % 360; // 3度ずつ色相を変化させる (速さは調整可能)
        
// //         // HSL (Hue/色相, Saturation/彩度, Lightness/明度) を使用して色を動的に設定
// //         // 彩度100%、明度70%で鮮やかな色を保ちます
// //         ctx.strokeStyle = `hsla(${rainbowHue}, 100%, 70%, 0.9)`; // 不透明度を少し上げて強調
// //     } else {
// //         // 通常の単色設定
// //         ctx.strokeStyle = currentWaveColor;
// //     }

   


        
// //         function drawBars() {
// //                 // 【★ 修正点3: 描画時に最新の中央位置を計算 ★】
// //                 const currentWaveY = canvas.height / 2;
                
// //                 // 描画色を決定するロジック
// //                 let barColor = currentWaveColor;
            
// //                 // レインボーモードの場合、動的に色を計算
// //                 if (currentWaveColor === 'rainbow') {
// //                     // drawWave()で更新された rainbowHue を使用
// //                     // HSL形式で色相を変化させ、レインボー効果を適用
// //                     const hue = (rainbowHue + 3) % 360; 
// //                     barColor = `hsla(${hue}, 100%, 70%, 0.9)`; 
// //                 }
                
// //                 // 決定した色を塗りつぶし色として設定
// //                 ctx.fillStyle = barColor;
                
// //                 bars.forEach(bar => {
// //                     // ctx.fillStyle = bar.color; // ← この行は削除（またはコメントアウト）
                    
// //                     // 最新の currentWaveY を使用
// //                     // 全てのバーで、動的に設定された同じ barColor が使われます
// //                     ctx.fillRect(bar.x, currentWaveY - bar.height / 2, barWidth - 2, bar.height); 
// //                 });
// //             }
// //             drawBars();
// //             animationFrameId = requestAnimationFrame(animateBars);
// //         }
// //         /* --- 2.機密保持/開発/コードを開く関連 --- */
// // /* --------------------------------------------------------------------------------- */
// // /* 3. 機密保持/開発者ツールの無効化 (重複を解消し整理)                                */
// // /* --------------------------------------------------------------------------------- */
// //     /* ============================================== */
// // /* 開発者ツールと右クリックの無効化        */
// // /* ============================================== */

// // // 1. 右クリック（コンテキストメニュー）を禁止する
// // //    HTMLの <body> タグに oncontextmenu="return false;" を追加するのが最も確実ですが、
// // //    JavaScriptでも document全体と body要素の両方に設定することで、カバー範囲を広げます。

// // document.addEventListener('contextmenu', function(e) {
// //     e.preventDefault();
// //     console.log("右クリックは禁止されています。");
// //     return false;
// // });

// // document.body.addEventListener('contextmenu', function(e) {
// //     e.preventDefault();
// //     return false;
// // });

// // // 2. キーボードショートカットを禁止する
// // // 2. キーボードショートカットを禁止する (ロジックを統合・整理)
// // //    e.keyを小文字に統一し、Windows/Linux (Ctrl) と macOS (Cmd/Option) の両方に対応させます。

// // //         // F12キーやCtrl/Cmd+Uを無効にする
// // document.onkeydown = function(e) {
// //     const key = e.key;
// //     const lowerKey = key.toLowerCase();
    
// //     // F12キー (開発者ツール)
// //     if (key === 'F12' || e.keyCode === 123) { 
// //         e.preventDefault();
// //         return false;
// //     }

// //     // Ctrl/Cmd/Option/Alt キー状態のチェックを容易にする
// //     const isCmdOrCtrl = e.ctrlKey || e.metaKey;
// //     const isShift = e.shiftKey;
// //     const isAltOrOption = e.altKey;

// //     // --- 開発者ツールのショートカット (I, J, C) ---
    
// //     // Ctrl/Cmd + Shift + I/J/C
// //     if (isCmdOrCtrl && isShift && (lowerKey === 'i' || lowerKey === 'j' || lowerKey === 'c')) {
// //         e.preventDefault();
// //         return false;
// //     }
    
// //     // Cmd + Option + I/J/C (macOSの一般的な検証ショートカット)
// //     if (e.metaKey && isAltOrOption && (lowerKey === 'i' || lowerKey === 'j' || lowerKey === 'c')) {
// //         e.preventDefault();
// //         return false;
// //     }

// //     // --- その他、コンテンツ保護のためのショートカット ---

// //     // Ctrl/Cmd + U (ソース表示)
// //     if (isCmdOrCtrl && lowerKey === 'u') {
// //         e.preventDefault();
// //         return false;
// //     }
// //     // その他、コンテンツ保護のためのショートカット (U, S, P)
// //     if (isCmdOrCtrl && (lowerKey === 'u' || lowerKey === 's' || lowerKey === 'p')) {
// //         e.preventDefault();
// //         return false;
// //     }
    
// //     // Ctrl/Cmd + S (保存)
// //     if (isCmdOrCtrl && lowerKey === 's') { 
// //         e.preventDefault();
// //         return false;
// //     }
    
// //     // Ctrl/Cmd + P (印刷)
// //     if (isCmdOrCtrl && lowerKey === 'p') {
// //         e.preventDefault();
// //         return false;
// //     }
// // // };
// // //         
// // //         // F12キーやCtrl/Cmd+Uを無効にする
// // //     document.onkeydown = function(e) {
// // //     const key = e.key;
    
// //     // F12キー (開発者ツール)
// //     if (key === 'F12') {
// //         e.preventDefault();
// //         return false;
// //     }

// //     // Ctrl/Cmd + Shift + I/J/C
// //     if ((e.ctrlKey || e.metaKey) && e.shiftKey && 
// //        (key === 'I' || key === 'i' || key === 'J' || key === 'j' || key === 'C' || key === 'c')) {
// //         e.preventDefault();
// //         return false;
// //     }
    
// //     // Cmd + Option + I/J/C (macOS)
// //     if (e.metaKey && e.altKey && 
// //        (key === 'I' || key === 'i' || key === 'J' || key === 'j' || key === 'C' || key === 'c')) {
// //         e.preventDefault();
// //         return false;
// //     }

// //     // Ctrl/Cmd + U (ソース表示)
// //     if ((e.ctrlKey || e.metaKey) && (key === 'u' || key === 'U')) {
// //         e.preventDefault();
// //         return false;
// //     }
    
// //     // Windows/Linux向けのCtrlキーのみのショートカットをカバー
// //     if (e.ctrlKey && key === 'S') { // Ctrl+S (保存) の無効化なども追加可能
// //         e.preventDefault();
// //         return false;
// //     }
// // };

// //         /* --- 2. 音声読み上げ/認識/API連携関連 --- */
// //         
// //         // DOM要素の取得
// //         const statusArea = document.getElementById("status-area");
// //         const sendBtn = document.getElementById("sendBtn"); 
// //         const input = document.getElementById("messageInput"); 
// //         const transcriptBox = document.getElementById('transcript');
// //         const ui = document.getElementById('ui'); 
// //         const tapArea = document.getElementById('tapArea'); 
// //         
// //         // API設定 (ご自身の環境に合わせて変更してください)
// //         const API_KEY = ""; 
// //         // const LLM_API_URL = "http://127.0.0.1:8001/generate";
// //         // const MQTT_API_URL = "http://127.0.0.1:8000/control"; 
// //     　　const LLM_API_URL = "https://atjmuwnwmtjw-hello.hf.space/llm/generate";
// //         const MQTT_API_URL = "https://atjmuwnwmtjw-hello.hf.space/iot/control"; 

// //         // 状態管理変数
// //         const synth = window.speechSynthesis;
// //         let audioContext, analyser, mediaStream;
// //         let recognition = null; 
// //         let currentTextToSpeak = ''; 
// //         
// //         // --- ヘルパー関数 (色の補間) ---
// //         function hexToRgb(hex) {
// //             const bigint = parseInt(hex.slice(1), 16);
// //             const r = (bigint >> 16) & 255;
// //             const g = (bigint >> 8) & 255;
// //             const b = bigint & 255;
// //             return [r, g, b];
// //         }

// //         function rgbToHex(r, g, b) {
// //             const toHex = (c) => ('0' + Math.max(0, Math.min(255, c)).toString(16)).slice(-2);
// //             return '#' + toHex(Math.round(r)) + toHex(Math.round(g)) + toHex(Math.round(b));
// //         }
// //         
// //         function startColorTransition(startColor, endColor, duration = 2000) {
// //             // 既存の実装を維持
// //             const startTime = performance.now();
// //             const startRgb = hexToRgb(startColor);
// //             const endRgb = hexToRgb(endColor);

// //             function interpolate(currentTime) {
// //                 const elapsed = currentTime - startTime;
// //                 const progress = Math.min(1, elapsed / duration);
// //                 
// //                 const r = startRgb[0] + (endRgb[0] - startRgb[0]) * progress;
// //                 const g = startRgb[1] + (endRgb[1] - startRgb[1]) * progress;
// //                 const b = startRgb[2] + (endRgb[2] - startRgb[2]) * progress; 
// //                 
// //                 const currentColor = rgbToHex(r, g, b);
// //                 
// //                 statusArea.style.color = currentColor;
// //                 statusArea.style.boxShadow = `0 0 20px ${currentColor}80`;

// //                 if (progress < 1) {
// //                     requestAnimationFrame(interpolate);
// //                 }
// //             }
// //             
// //             requestAnimationFrame(interpolate);
// //         }
// //         
// //         /* ---------- UI helpers ---------- */

// //         function updateStatus(message, color = '#00ffff') {
// //             statusArea.innerHTML = message; 
// //             statusArea.style.color = color;
// //             statusArea.style.boxShadow = `0 0 20px ${color}80`;
// //         }

// //         function setStandbyStatus() {
// //             const standbyMsg = `
// //             イマジナリーナンバー
// //             通称GAIイマさんAI
// //             AIアシスタント待機中...
// //             `;
// //             updateStatus(standbyMsg.trim(), '#00ffff');
// //         }
// //         
// //         /* ---------- TTS (Speech Synthesis) ---------- */

// //         /**
// //          * LLM応答など、AIからの正式な応答を読み上げ、終了後にSTTを再起動する
// //          */

// //          // 【外部定義】絵文字抽出関数 (これをどこかグローバルな場所に置いてください)
// //         const extractEmojis = (text) => {
// //             // 最新のブラウザ・環境向け
// //             const emojiRegex = /\p{Emoji_Presentation}|\p{Emoji}\p{Emoji_Modifier}*|\p{Emoji_Component}|\u200d/gu;
// //             const matches = text.match(emojiRegex);
// //             return matches ? matches.join('') : '';
// //         };


// //         function speak(text){ 
// //             if(!text) return; 
// //             
// //             currentTextToSpeak = text; 
// //             
// //             if(synth.speaking) synth.cancel(); 
// //             
// //             isSpeaking = true; 

// //             // LLM応答に基づいて波形の色を設定
// //             setWaveColorBasedOnResponse(text);

// //             const u = new SpeechSynthesisUtterance(text); 
// //             u.lang='ja-JP'; 
// //             u.rate=1.0; 

// //             // ⭐ u.onstart イベントハンドラを修正 ⭐
// //             u.onstart=()=>{ 
// //                 // 1. 表示用のテキストを切り出す (既存のロジック)
// //                 const display = text.length > 20 ? text.substring(0, 20) + '...' : text;

// //                 // 2. 絵文字のみを抽出する (追加)
// //                 // const onlyEmojis = extractEmojis(responseText);
// //                 // 【約821行目】
// // // ...

// //                 // 2. 絵文字のみを抽出する (修正: text を使用)
// //                 const onlyEmojis = extractEmojis(text); // ★★★ 修正箇所: text に変更 ★★★
// // // ...

// //                 // 3. formattedStatusに絵文字を含めて表示する (修正)
// //                 const formattedStatus = `
// //             　　---==(_____[　イマジナリーナンバー通称GAIイマさんAI応答:  ?&!! ${onlyEmojis}　]_____)==--- __(V._.V)__
// //                                              「${display}」
// //             `;
// //             // 絵文字を ${onlyEmojis} の位置に挿入しました
// //                 updateStatus(formattedStatus.trim(), '#00ffaa');
// //             }; 
// //             u.onend=()=>{ 
// //                 isSpeaking = false; 
// //                 currentTextToSpeak = ''; 
// //                 setStandbyStatus();
// //                 input.value = '';

// //                 // // TTS終了後、波形の色をデフォルトに戻す
// //                 //         currentWaveColor = WAVE_COLORS.default;

// //                 // TTS終了後、STTが停止していれば自動で再起動を試みる
// //                 if (recognition && !isRecording) {
// //                     try {
// //                         recognition.start();
// //                     } catch(e) {
// //                         console.warn('Recognition restart failed after TTS:', e);
// //                     }
// //                 }
// //             }; 
// //             u.onerror = (e) => {
// //                 console.error('TTS error:', e);
// //                 isSpeaking = false;
// //                 currentTextToSpeak = '';
// //                 setStandbyStatus();
// //                 input.value = '';
// // 　　　　　　　　  currentWaveColor = WAVE_COLORS.default;
// //             };


// //             // スピーチをキューに追加
// //             synth.speak(u); 
// //         }

// //         /**
// //          * テキスト入力時の即時プレビュー用読み上げ関数（グローバルスコープに移動）
// //          */
// //         /**
// //          * テキスト入力時の即時プレビュー用読み上げ関数
// //          * @param {string} text 読み上げるテキスト
// //          */
// //         function speakSentence(text) {
// //             // テキストが空か、既に同じテキストの読み上げが開始されている場合は何もしない
// //             if (text.trim() === '' || text === currentTextToSpeak) {
// //                 return;
// //             }

// //             // 新しい読み上げが開始されるので、現在の読み上げをキャンセル
// //             if (synth.speaking) {
// //                 synth.cancel();
// //             }
// //             
// //             currentTextToSpeak = text; // 新しい文章を記憶

// //             const utterance = new SpeechSynthesisUtterance(text); // const/let を使用
// //             utterance.lang = 'ja-JP'; // 日本語を設定
// //             utterance.rate = 1.0; 

// //             utterance.onstart = () => {
// //                 isSpeaking = true;
// //                 // 読み上げ中の文章を一部表示
// //                 const display = text.length > 20 ? text.substring(0, 20) + '...' : text;
// //                 updateStatus(`文章を読み上げ中: 「${display}」`, '#00ffaa');
// //                 // プレビュー読み上げ中も波形を動かすため、一時的にcurrentWaveColorをポジティブに設定
// //                  currentWaveColor = WAVE_COLORS.positive;
// //                 };
// //             
// //             utterance.onend = () => {
// //                 isSpeaking = false;
// //                 // 即時プレビューが終わっても、待機中のステータスに戻すだけ
// //                 setStandbyStatus(); 
// //                 currentWaveColor = WAVE_COLORS.default; // 終了したらデフォルトに戻す
// //             };

// //             utterance.onerror = (event) => {
// //                 console.error('Speech Synthesis Error:', event);
// //                 isSpeaking = false;
// //                 updateStatus('読み上げエラーが発生しました', '#ff0000');
// //                 currentWaveColor = WAVE_COLORS.default;  
// //             };

// //             synth.speak(utterance);
// //         }

// //         /* ---------- Speech Recognition (Browser STT) & Audio Init ---------- */

// //         function restartRecognition() {
// //             isRecording = false;

// //             // TTSが動作中でなければ、待機状態に戻す
// //             if (!synth.speaking) {
// //                 isSpeaking = false;
// //                 setStandbyStatus();
// //             }

// //             setTimeout(() => {
// //                 try {
// //                     // 既に認識が開始されている場合は何もしない
// //                     if (!isRecording && !synth.speaking && recognition) recognition.start();
// //                 } catch (e) {
// //                     if (e.name !== 'InvalidStateError') {
// //                         console.warn('Recognition start failed:', e);
// //                     }
// //                 }
// //             }, 500);
// //         }

// //         /* ---------- Speech Recognition (Browser STT) & Audio Init ---------- */

// //         function startBrowserRecognition() {
// //             if (isRecording) return;
// //             
// //             if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
// //                 updateStatus('Error: Speech Recognition not supported in this browser.', '#ff0000');
// //                 return;
// //             }

// //             if (recognition) {
// //                 recognition.stop();
// //                 recognition = null;
// //             }

// //             recognition = new (window.webkitSpeechRecognition || window.SpeechRecognition)();
// //             recognition.continuous = false; 
// //             recognition.interimResults = true; 
// //             recognition.lang = 'ja-JP';

// //             recognition.onstart = () => {
// //                 isRecording = true;
// //                 isSpeaking = true; 
// //                 const standbyMsg = `
// //                 Listening...
// //                 話しかけてください...！
// //                 `;
// //                 updateStatus(standbyMsg.trim(), '#ffff00');
// //                 startColorTransition('#ffff00', '#00ffaa', 2000); 
// //                 
// //                 // 画面中央のトランスクリプトは非表示のため、処理はコメントアウト
// //                 // transcriptBox.textContent = '話しかけてください...';
// //                 input.value = ''; 
// //                 if (synth.speaking) synth.cancel(); 
// //             };

// //             recognition.onresult = (event) => {
// //                 let interimTranscript = '';
// //                 let finalTranscript = '';

// //                 for (let i = event.resultIndex; i < event.results.length; ++i) {
// //                     if (event.results[i].isFinal) {
// //                         finalTranscript += event.results[i][0].transcript;
// //                     } else {
// //                         interimTranscript += event.results[i][0].transcript;
// //                     }
// //                 }
// //                 // 画面中央のトランスクリプト表示を非表示にするため、コメントアウト
// //                 // transcriptBox.textContent = finalTranscript || interimTranscript; 
// //                 input.value = finalTranscript || interimTranscript; // 入力欄には反映
// //             };

// //             // 発話終了またはエラー時の自動再スタートロジック
// //             const restartRecognition = () => {
// //                 isRecording = false;
// //                 
// //                 // TTSが動作中でなければ、待機状態に戻す
// //                 if (!synth.speaking) {
// //                     isSpeaking = false; 
// //                     setStandbyStatus();
// //                 }
// //                 
// //                 setTimeout(() => {
// //                     try {
// //                         // 既に認識が開始されている場合は何もしない
// //                         if (!isRecording && !synth.speaking) recognition.start(); 
// //                     } catch (e) {
// //                         if (e.name !== 'InvalidStateError') {
// //                             console.warn('Recognition start failed:', e);
// //                         }
// //                     }
// //                 }, 500); 
// //             };
// //             
// //             recognition.onend = () => {
// //                 isRecording = false;
// //                 
// //                 // TTSが動作していない場合に限り isSpeaking を false に
// //                 if (!synth.speaking) {
// //                     isSpeaking = false; 
// //                 }
// //                 
// //                 const finalPrompt = input.value.trim(); // transcriptBoxの代わりにinput.valueを使う
// //                 
// //                 // 認識結果が空でない、またはデフォルトメッセージでない場合のみ処理
// //                 if (finalPrompt && finalPrompt.length > 1 && !finalPrompt.startsWith("話しかけてください") && !finalPrompt.startsWith("イマジナリーナンバー 通称GAIイマさんAI応答:")) {
// //                     updateStatus('Processing response...', '#00ffaa');
// //                     
// //                     // LLM処理中にSTTが自動で再起動しないように、.finallyでrestartRecognitionを呼ぶ
// //                     processRecognitionResult(finalPrompt).finally(() => {
// //                         // TTSが終了した後に再起動させる (speak関数内のonendでも実施されるため冗長ではあるが念のため)
// //                         if (!synth.speaking) {
// //                             restartRecognition(); 
// //                         }
// //                     });
// //                 } else {
// //                     // input.value = ''; // onresultでクリアされるため不要
// //                     restartRecognition();
// //                 }
// //             };

// //             recognition.onerror = (event) => {
// //                 isRecording = false;
// //                 isSpeaking = false;
// //                 console.error('Speech Recognition Error:', event.error);
// //                 
// //                 if (event.error !== 'not-allowed' && event.error !== 'aborted') {
// //                     restartRecognition();
// //                 } else if (event.error === 'aborted') {
// //                     // 意図的な停止（stop()呼び出し）の場合もあるため、再起動
// //                     restartRecognition(); 
// //                 } else {
// //                     updateStatus('Error: Microphone permission denied or failed.', '#ff0000');
// //                 }
// //             };

// //             try {
// //                 recognition.start();
// //             } catch (e) {
// //                 console.warn('Initial recognition start failed:', e);
// //             }
// //         }

// //         async function initAudioAndSTT(){
// //             if(analyser) {
// //                 startBrowserRecognition();
// //                 return;
// //             }
// //             updateStatus('Requesting microphone access...');

// //             try {
// //                 audioContext = new (window.AudioContext || window.webkitAudioContext)();
// //                 analyser = audioContext.createAnalyser();
// //                 analyser.fftSize = 2048;
// //                 
// //                 dataArray = new Uint8Array(analyser.frequencyBinCount);
// //                 
// //                 mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
// //                 const sourceNode = audioContext.createMediaStreamSource(mediaStream);
// //                 
// //                 sourceNode.connect(analyser);

// //                 startBrowserRecognition();

// //                 updateStatus('Listening...', '#ffff00');
// //             } catch (e) {
// //                 console.error('Audio initialization failed:', e);
// //                 updateStatus('Error: Microphone access denied or failed to initialize.', '#ff0000');
// //             }
// //         }

// //         /**
// //          * FastAPI/MQTTバックエンドにコマンドを送信する関数
// //          */
// //         async function sendIoTCommand(command) {
// //             updateStatus(`Executing IoT command: ${command}...`, '#00ffaa');
// //             // 画面中央のトランスクリプト表示を非表示にするため、コメントアウト
// //             // transcriptBox.textContent = `IoTコマンド: ${command} を実行中...`;
// //             
// //             try {
// //                 const response = await fetch(MQTT_API_URL, {
// //                     method: 'POST',
// //                     headers: { 'Content-Type': 'application/json' },
// //                     body: JSON.stringify({ command: command })
// //                 });

// //                 const data = await response.json();

// //                 if (response.ok) {
// //                     const successMsg = `承知しました。${command === 'ON' ? '電気をつけました' : '電気を消しました'}。`;
// //                     speak(successMsg);
// //                 } else {
// //                     const detail = data.detail || "サーバーエラー";
// //                     const errorMsg = `エラーが発生しました。IoTコマンド '${command}' の実行に失敗しました。詳細: ${detail}`;
// //                     speak(errorMsg);
// //                 }
// //             } catch (error) {
// //                 const networkErrorMsg = `🔴 ネットワークエラー: IoTバックエンドサーバーに接続できません (${error.message})`;
// //                 speak(networkErrorMsg);
// //             }
// //         }


// //         /* ---------- 統合されたメイン処理関数 (IoT or LLM) ---------- */

// //         async function processRecognitionResult(finalPrompt) {
// //             // 1. IoTコマンドの判定と振り分け
// //             const lowerPrompt = finalPrompt.toLowerCase();
// //             let iotCommand = null;

// //             if ((lowerPrompt.includes('ライト') || lowerPrompt.includes('電気')) && (lowerPrompt.includes('つけ') || lowerPrompt.includes('オン') || lowerPrompt.includes('点け'))) {
// //                 iotCommand = 'ON';
// //             } else if ((lowerPrompt.includes('ライト') || lowerPrompt.includes('電気')) && (lowerPrompt.includes('けし') || lowerPrompt.includes('オフ') || lowerPrompt.includes('消し'))) {
// //                 iotCommand = 'OFF';
// //             }

// //             if (iotCommand) {
// //                 await sendIoTCommand(iotCommand);
// //                 return; 
// //             }
// //             
// //             // 2. LLM応答生成（IoTコマンドでなかった場合）
// //             await generateAndSpeakResponse(finalPrompt);

// //             // // 【新規追加または修正】 processRecognitionResult関数の実装を推定
// //             // async function processRecognitionResult(finalPrompt) {
// //             //     try {
// //             //         // LLMへの応答生成を試みる
// //             //         await generateAndSpeakResponse(finalPrompt);
            
// //             //     } catch (error) {
// //             //         // generateAndSpeakResponse内でエラーハンドリングが実施されるため、
// //             //         // ここでは特別な処理は不要ですが、念のためログを残します。
// //             //         console.error('Overall LLM processing failed:', error);
// //             //     }
// //             // }
                        
// //         }


// //         /* ---------- LLM (Gemini) API & TTS 連携 ---------- */
// //          async function generateAndSpeakResponse(prompt) {
// //              updateStatus('Generating response (via FastAPI)...', '#00ffaa');
// //             
// //              const cleanedPrompt = prompt.replace(/^イマジナリーナンバー 通称GAIイマさんAI応答:\s*/, '').trim();
// //              if (!cleanedPrompt) {
// //     　　　　　　　speak("すみません、何も聞こえませんでした。もう一度話しかけてください。");
// //                  return; 
// //              }


// //             const systemInstruction = "あなたは「イマジナリーナンバー 通称GAIイマさん」という名前のKS-903model8800-a1-90dという音声アシスタントです。ユーザーの質問に日本語で、簡潔かつ丁寧に答えてください。";

// //             const payload = {
// //                 prompt: cleanedPrompt,
// //                 contents: [{ parts: [{ text: cleanedPrompt }] }],
// //                 systemInstruction: { parts: [{ text: systemInstruction }] },
// //                 tools: [{ "google_search": {} }], 
// //             };
// //             
// //             const MAX_RETRIES = 3;
// //             let responseText = "エラーが発生しました。イマジナリーナンバー 通称GAIイマさんAIのKS-903model8800-a1-90d応答を取得できませんでした。";

// //             for (let i = 0; i < MAX_RETRIES; i++) {
// //                 try {
// //                     const response = await fetch(LLM_API_URL, {
// //                         method: 'POST',
// //                         headers: { 'Content-Type': 'application/json' },
// //                         body: JSON.stringify(payload)
// //                     });
// //                     
// //                     if (!response.ok) {
// //                         const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status} Error.` }));
// //                         throw new Error(`FastAPI Error! Status: ${response.status}. Detail: ${errorData.detail}`);
// //                     }
// //                     
// //                     const result = await response.json();
// //                     
// //                     if (result && result.text) {
// //                         responseText = result.text;
// //                         break; 
// //                     } else {
// //                          throw new Error("Empty response or invalid JSON structure from FastAPI.");
// //                     }

// //                 } catch (e) {
// //                     console.error(`FastAPI call error on attempt ${i + 1}:`, e);
// //                     if (i === MAX_RETRIES - 1) {
// //                         responseText = "エラーが発生しました。イマジナリーナンバー 通称GAIイマさんAIKS-903model8800-a1-90dの応答を取得できませんでした。Generaltebバックエンドサーバー (ポート8001) の実行状態とAPIキーを確認してください。";
// //                     } else {
// //                         const delay = 2 ** i * 1000 + Math.random() * 500;
// //                         await new Promise(resolve => setTimeout(resolve, delay));
// //                     }
// //                 }
// //             }

// //             updateStatus('Speaking response...', '#00ffaa');
// //             speak(responseText); 
// //             
// //             return Promise.resolve();
// //         }

// //         /* ---------- イベントハンドラの統合と定義 ---------- */

// //         // テキスト入力欄のイベントを追加 (Enterキーで処理)
// //         input.addEventListener('keydown', (e) => {
// //             // Enterキーが押された場合（改行を防ぎ、処理を開始）
// //             if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey) {
// //                 e.preventDefault(); 
// //                 
// //                 const textPrompt = input.value.trim();
// //                 
// //                 if (textPrompt) {
// //                     // 音声認識が実行中の場合は強制停止
// //                     if (recognition && isRecording) {
// //                         recognition.stop(); 
// //                     }
// //                     // TTSをキャンセル（即時読み上げを停止）
// //                     if(synth.speaking) synth.cancel(); 
// //                     
// //                     // 処理を優先
// //                     updateStatus('Processing text input...', '#ffff00');
// //                     // 画面中央のトランスクリプト表示を非表示にするため、コメントアウト
// //                     // transcriptBox.textContent = textPrompt; 
// //                     
// //                     // LLM処理を実行
// //                     processRecognitionResult(textPrompt).catch(error => {
// //                         console.error("Text input processing failed:", error);
// //                     }).finally(() => {
// //                         // input.valueはspeakのonendでクリアされるため、ここでは何もしない
// //                     });
// //                 }
// //             }
// //         });

// //         // テキスト入力のたびに現在の内容を読み上げる機能の追加 (TTS即時プレビュー)
// //         input.addEventListener('input', (event) => {
// //             const currentText = input.value.trim();
// //             
// //             // 音声認識が実行中でない、かつ、AIが応答中でない場合にのみ実行
// //             // かつ、現在のテキストが読み上げ中のテキストと異なる場合
// //             if (!isRecording && !isSpeaking && currentText.length > 0 && currentText !== currentTextToSpeak) {
// //                 // ★★★ ここを speakSentence に変更 ★★★
// //                 speakSentence(currentText); 
// //             } else if (currentText.length === 0 && synth.speaking) {
// //                 // テキストが全て削除され、かつ読み上げ中の場合はキャンセルして待機状態に戻す
// //                 synth.cancel();
// //                 isSpeaking = false;
// //                 setStandbyStatus();
// //             }
// //         });

// //         // リセットボタンの機能 (STTとTTSの強制停止と再起動)
// //         sendBtn.addEventListener("click", () => {
// //             if (recognition) {
// //                 recognition.stop();
// //                 recognition = null;
// //                 // ... (既存のコード) ...
// //                 // 【追記箇所】色をデフォルトに戻す
// //                 currentWaveColor = WAVE_COLORS.default; 
// //                 // ... (既存のコード) ...
// //                 // isRecordingはonendでfalseになるが、即時リセットのため手動でも設定
// //                 isRecording = false;
// //             }
// //             if(synth.speaking) synth.cancel(); 

// //             // 画面中央のトランスクリプト表示を非表示にするため、コメントアウト
// //             // transcriptBox.textContent='リセット中...'; 
// //             
// //             // isSpeakingとisRecordingを強制的にfalseに
// //             isSpeaking = false;
// // //             isRecording = false
// // 　　　　　　　currentWaveColor = WAVE_COLORS.default;

// //             initAudioAndSTT();
// //             updateStatus('リセットしました。マイク入力を開始しています...'); 
// //         });


// //         // UI トグル機能 (画面タップ) 
// //         let uiVisible = true; 
// //         tapArea.addEventListener('click', (e) => {
// //             // リセットボタンへのタップは無視
// //             if (e.target.closest('#input-controls')) {
// //                 return;
// //             }

// //             uiVisible = !uiVisible;
// //             if (uiVisible) {
// //                 ui.style.opacity = 1; 
// //             } else {
// //                 ui.style.opacity = 0; 
// //             }
// //         });

// //         /* ---------- Start-up ---------- */
// //         window.onload = function() {
// //             // createBars();
// //             animateBars();
// //             initAudioAndSTT(); // マイク初期化とSTTを自動で開始
// //             setStandbyStatus();
// //             
// //             // UIをデフォルトで表示状態にする
// //             ui.style.opacity = 1; 
// //             uiVisible = true;
// //         }