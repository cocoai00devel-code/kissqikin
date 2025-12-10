/// imakunAI.js #

/* --------------------------------------------------------------------------------- */
/* 1. Canvasアニメーションとレスポンシブ対応                                          */
/* --------------------------------------------------------------------------------- */

const canvas = document.getElementById("waveCanvas");
const ctx = canvas.getContext("2d");

let bars = [];
const BAR_COUNT = 40;
const BAR_WIDTH = 8;
let dataArray;

let animationFrameId;
let transitionFrameId; // 連続色遷移用IDを追加
let isSpeaking = false;
let isRecording = false;
let isTtsSpeaking = false; // ★追加★ TTSがアクティブかどうかを判定するフラグ
let currentWaveColor = 'rgba(50, 200, 255, 0.7)'; // 初期色を定義
let rainbowHue = 0; // レインボーアニメーション用の色相を保持


const WAVE_COLORS = {
    default: 'rgba(50, 200, 255, 0.7)',
    positive: 'rgba(50, 255, 50, 0.7)',
    anger: 'rgba(255, 50, 50, 0.7)',
    rage: 'rgba(150, 50, 255, 0.7)',
    negative: 'rgba(50, 100, 255, 0.7)',
    sadness: 'rgba(0, 0, 150, 0.7)'
};

const STATUS_TRANSITION_COLORS = [
    '#32CD32', '#ADFF2F', '#FFA500', '#FF4500', 
    '#8A2BE2', '#00008B', '#00FFFF', '#FFFF00'
];
// 緑 → ライムグリーン → オレンジ → 赤オレンジ → 青紫 → 濃い青 → シアン → 黄色





/**
 * ステータスエリアを現在の感情色で点滅アニメーションさせる
 */
/* --------------------------------------------------------------------------------- */
/* 4. 音声読み上げ/認識/API連携関連 に追加 (グローバルに追加) */
/* --------------------------------------------------------------------------------- */

let blinkFrameId; // 点滅アニメーション用のフレームID

/**
 * ステータスエリアを現在の感情色で点滅アニメーションさせる
 */
// STATUS_TRANSITION_COLORS がグローバルで定義されている必要があります。
// 例: const STATUS_TRANSITION_COLORS = ['#32CD32', '#ADFF2F', ...];

/**
 * 感情色でステータスエリアを点滅アニメーションさせる
 * 感情が 'rainbow' の場合は、連続色遷移に切り替える
 */
function startStatusBlink() { // 👈 startSequentialColorTransitionの外に移動
    // ----------------------------------------------------
    // ★修正点: レインボー判定と分岐処理★
    if (currentWaveColor === 'rainbow') {
        // レインボーの場合は点滅を停止し、連続色遷移を開始
        // 既存のblinkアニメーションを停止（もし動いていれば）
        stopStatusBlink(); 
        
        // 連続色遷移関数を呼び出し、通常の処理を終了
        // TTS開始時なので segmentDuration は 500ms を使用
        startSequentialColorTransition(STATUS_TRANSITION_COLORS, 500); 
        console.log("感情がレインボーのため、連続色遷移アニメーションを開始しました。");
        return; // 点滅ロジックには進まない
    }
    // ----------------------------------------------------
    
    // 既存のアニメーションを確実に停止 (非レインボーの場合の点滅準備)
    // 既存のアニメーションを確実に停止
    stopSequentialColorTransition(); 
    stopStatusRainbow();
    if (blinkFrameId) cancelAnimationFrame(blinkFrameId);
    
    // 現在の感情色を取得 (例: 'rgba(50, 255, 50, 0.7)' のような形式)
    const baseColor = currentWaveColor.replace(/,\s*0\.\d+\)/, ', 1)'); // 透明度を1.0に強制
    
    const startTime = performance.now();

    function animate(currentTime) {
        // AI応答中（isSpeakingまたはisTtsSpeaking）の間だけアニメーションを継続// AI応答中（isSpeakingがtrueの間）だけアニメーションを継続
        if (!isSpeaking) { 
            stopStatusBlink();
            return;
        }

        const elapsed = currentTime - startTime;
        const blinkSpeed = 0.005; // 点滅速度 (値を大きくすると速くなる)
        const intensity = 0.65 + Math.sin(elapsed * blinkSpeed) * 0.35; 

        const shadowColor = baseColor.replace(/1\)/, `${intensity.toFixed(2)})`);

        // 文字色をベースカラーに固定
        statusArea.style.color = baseColor;
        // シャドウで明滅感を出す
        statusArea.style.boxShadow = `0 0 20px ${baseColor.replace(/1\)/, '0.8)')}, 0 0 50px ${shadowColor}`;

        blinkFrameId = requestAnimationFrame(animate);
    }
    blinkFrameId = requestAnimationFrame(animate);
}

/**
 * 点滅アニメーションを停止
 */
function stopStatusBlink() {
    if (blinkFrameId) {
        cancelAnimationFrame(blinkFrameId);
        blinkFrameId = null;
    }
}
/**
 * 連続色遷移アニメーションを開始 (ステータスエリア用)
 */
function startSequentialColorTransition(colors, segmentDuration = 500) {
    // ★追加★ 既存のレインボーアニメーションを確実に停止させる
    stopStatusRainbow();
    stopStatusBlink();
    if (transitionFrameId) cancelAnimationFrame(transitionFrameId);
    
    const startTime = performance.now();
    const numSegments = colors.length;

    

    function animate(currentTime) {
        // isSpeakingがtrueの間だけアニメーションを継続
        if (!isSpeaking && !synth.speaking) {
            stopSequentialColorTransition();
            return;
        }
        // ... (レインボーの色相計算と適用) ...
        // statusRainbowFrameId = requestAnimationFrame(animate);
        /*
        // ★削除するべき行★ startStatusRainbowから誤ってコピーされたロジック
        // ... (レインボーの色相計算と適用) ...
        // statusRainbowFrameId = requestAnimationFrame(animate); 
        */

        const elapsed = currentTime - startTime;
        const totalDuration = numSegments * segmentDuration;
        const progress = (elapsed % totalDuration) / totalDuration;
        const currentSegmentIndex = Math.floor(progress * numSegments);
        const nextSegmentIndex = (currentSegmentIndex + 1) % numSegments;
        const segmentProgress = (elapsed % segmentDuration) / segmentDuration;

        const startRgb = hexToRgb(colors[currentSegmentIndex]);
        const endRgb = hexToRgb(colors[nextSegmentIndex]);

        // 補間
        const r = startRgb[0] + (endRgb[0] - startRgb[0]) * segmentProgress;
        const g = startRgb[1] + (endRgb[1] - startRgb[1]) * segmentProgress;
        const b = startRgb[2] + (endRgb[2] - startRgb[2]) * segmentProgress;

        const currentColor = rgbToHex(r, g, b);

        statusArea.style.color = currentColor;
        statusArea.style.boxShadow = `0 0 20px ${currentColor}80`;

        transitionFrameId = requestAnimationFrame(animate);
    }
    transitionFrameId = requestAnimationFrame(animate);
}

/**
 * 連続色遷移アニメーションを停止
 */
function stopSequentialColorTransition() {
    if (transitionFrameId) {
        cancelAnimationFrame(transitionFrameId);
        transitionFrameId = null;
    }
}


/**
 * バーのデータを再計算する
 */
function createBars() {
    bars = [];
    // 中央揃えでバーを配置
    const startX = canvas.width / 2 - (BAR_COUNT * BAR_WIDTH) / 2;
    for (let i = 0; i < BAR_COUNT; i++) {
        bars.push({
            x: startX + i * BAR_WIDTH,
            height: 10,
            color: "#00ffff" // 初期色は使用されないが、初期化
        });
    }
}

/**
 * Canvasサイズをウィンドウにフィットさせ、バーを再計算する
 */
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    // サイズ変更時にバーを再計算
    createBars();
}

/**
 * バーをアニメーションさせて描画する
 */
// function animateBars() {
//     // Canvasをクリア
//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // analyserが存在し、音声入力があれば周波数データを取得
//     if (analyser && dataArray && (isSpeaking || isRecording)) {
//         analyser.getByteFrequencyData(dataArray);
//     }
//     /* ★修正点: 冗長で不正確な height = 10; のロジックを削除 */

//     // 描画色を決定するロジック
//     let barColor = currentWaveColor;

//     // レインボーモードの場合、動的に色を計算
//     if (currentWaveColor === 'rainbow') {
//         rainbowHue = (rainbowHue + 3) % 360;
//         barColor = `hsla(${rainbowHue}, 100%, 70%, 0.9)`;
//     // ★修正点2: 待機中の色遷移ロジックを追加★
//     } else if (　currentWaveColor === 'standby_transition') {
//         const segmentDuration = 4000; // 4秒かけて一周する
//         const totalDuration = STATUS_TRANSITION_COLORS.length * segmentDuration;
        
//         // 時間経過で色を滑らかに遷移させる（performance.now() を使用）
//         const elapsed = performance.now() % totalDuration;
//         const numSegments = STATUS_TRANSITION_COLORS.length;
//         const progress = elapsed / totalDuration;
        
//         const currentSegmentIndex = Math.floor(progress * numSegments);
//         const nextSegmentIndex = (currentSegmentIndex + 1) % numSegments;
        
//         const segmentProgress = (elapsed % segmentDuration) / segmentDuration;

//         const startRgb = hexToRgb(STATUS_TRANSITION_COLORS[currentSegmentIndex]);
//         const endRgb = hexToRgb(STATUS_TRANSITION_COLORS[nextSegmentIndex]);

//         // 補間
//         const r = startRgb[0] + (endRgb[0] - startRgb[0]) * segmentProgress;
//         const g = startRgb[1] + (endRgb[1] - startRgb[1]) * segmentProgress;
//         const b = startRgb[2] + (endRgb[2] - startRgb[2]) * segmentProgress;
        
//         barColor = rgbToHex(r, g, b) + 'b3'; // 透明度 0.7 (b3) を追加
//     /* ★修正点: ループ外の高さ計算ロジック（iに依存するもの）を削除 */
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
//             // Date.now()で動き、iでオフセット
//             const waveOffset = Math.sin(Date.now() * 0.005 + i * waveFrequency) * waveAmplitude;
//             height = 10 + Math.abs(waveOffset);
//         } else {
//             // 待機中は最小の高さ
//             height = 10;
//         }

//         // バーの高さと位置を更新
//         bars[i].height = height;

//         // 描画
//         ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
//     });

//     animationFrameId = requestAnimationFrame(animateBars);
// }

// // 【★ 修正点1: window.addEventListenerの重複を解消し、一つに統合済み ★】
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

/**
 * バーをアニメーションさせて描画する (isSpeaking時の波形アニメーションを停止)
//  */
// function animateBars() {
//     // Canvasをクリア
//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // analyserが存在し、音声入力があれば周波数データを取得
//     if (analyser && dataArray && (isSpeaking || isRecording)) {
//         analyser.getByteFrequencyData(dataArray);
//     }
    
//     // 描画色を決定するロジック (変更なし: 色は 'rainbow' や 'standby_transition' に応じて変化)
//     let barColor = currentWaveColor;
//     // ... (色遷移ロジックは省略) ...
//     if (currentWaveColor === 'standby_transition') {
//         // const segmentDuration = 4000;
//         const segmentDuration = 890; // 🔴 ここを 4000 から 1000 に変更 (1秒で次の色へ遷移)
//         const totalDuration = STATUS_TRANSITION_COLORS.length * segmentDuration;
//         const elapsed = performance.now() % totalDuration;
//         const numSegments = STATUS_TRANSITION_COLORS.length;
//         const progress = elapsed / totalDuration;
//         const currentSegmentIndex = Math.floor(progress * numSegments);
//         const nextSegmentIndex = (currentSegmentIndex + 1) % numSegments;
//         const segmentProgress = (elapsed % segmentDuration) / segmentDuration;
//         const startRgb = hexToRgb(STATUS_TRANSITION_COLORS[currentSegmentIndex]);
//         const endRgb = hexToRgb(STATUS_TRANSITION_COLORS[nextSegmentIndex]);
//         const r = startRgb[0] + (endRgb[0] - startRgb[0]) * segmentProgress;
//         const g = startRgb[1] + (endRgb[1] - startRgb[1]) * segmentProgress;
//         const b = startRgb[2] + (endRgb[2] - startRgb[2]) * segmentProgress;
//         barColor = rgbToHex(r, g, b) + 'b3';
//     } else if (currentWaveColor === 'rainbow') {
//         rainbowHue = (rainbowHue + 3) % 360;
//         barColor = `hsla(${rainbowHue}, 100%, 70%, 0.9)`;
//     } 
    
//     // 決定した色を塗りつぶし色として設定
//     ctx.fillStyle = barColor;
//     const currentWaveY = canvas.height / 2;

//     bars.forEach((bar, i) => {
//         let height = bar.height;

//         // 音声入力中のみ、波形を動かす
//         if (isRecording && dataArray) {
//             // 周波数データを単純にマッピング
//             const dataIndex = Math.floor(i * (dataArray.length / BAR_COUNT));
//             const rawHeight = dataArray[dataIndex] || 0;
//             // 0-255を最大高さ（例: 200）にスケール
//             height = (rawHeight / 255) * 200 + 5;
//         } else if (isTtsSpeaking) { // ★修正点4: isSpeaking ではなく isTtsSpeaking の時のみ波形を動かす
//             // TTS応答中は、シンプルなサイン波で波形を動かす
//             const waveAmplitude = 100;
//             const waveFrequency = 0.05;
//             // Date.now()で動き、iでオフセット
//             const waveOffset = Math.sin(Date.now() * 0.005 + i * waveFrequency) * waveAmplitude;
//             height = 10 + Math.abs(waveOffset);
//         } else {
//             // ★修正点★ 待機中 (isRecording=false かつ isSpeaking=false)
//             // または AI応答中 (isSpeaking=true) の場合、高さを 10 に固定する
//             height = 10; 
//         }

//         // バーの高さと位置を更新
//         bars[i].height = height;

//         // 描画
//         ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
//     });

//     animationFrameId = requestAnimationFrame(animateBars);
// }

// 注意: このコードは animateBars 関数のみを含みます。
// barCount, BAR_WIDTH, ctx, canvas, analyser, dataArray,
// isRecording, isTtsSpeaking, currentWaveColor などの変数が
// グローバルスコープまたは上位スコープで正しく定義されていることが前提です。

function animateBars() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // --- 色決定ロジック ---
    let barColor = currentWaveColor;
    if (currentWaveColor === 'standby_transition') {
        const segmentDuration = 890;
        const totalDuration = STATUS_TRANSITION_COLORS.length * segmentDuration;
        const elapsed = performance.now() % totalDuration;
        const numSegments = STATUS_TRANSITION_COLORS.length;
        const progress = elapsed / totalDuration;
        const currentSegmentIndex = Math.floor(progress * numSegments);
        const nextSegmentIndex = (currentSegmentIndex + 1) % numSegments;
        const segmentProgress = (elapsed % segmentDuration) / segmentDuration;
        const startRgb = hexToRgb(STATUS_TRANSITION_COLORS[currentSegmentIndex]);
        const endRgb = hexToRgb(STATUS_TRANSITION_COLORS[nextSegmentIndex]);
        const r = startRgb[0] + (endRgb[0] - startRgb[0]) * segmentProgress;
        const g = startRgb[1] + (endRgb[1] - startRgb[1]) * segmentProgress;
        const b = startRgb[2] + (endRgb[2] - startRgb[2]) * segmentProgress;
        barColor = rgbToHex(r, g, b) + 'b3';
    } else if (currentWaveColor === 'rainbow') {
        // rainbowHueが未定義の場合は適宜初期化（例: let rainbowHue = 0;）が必要です
        rainbowHue = (rainbowHue + 3) % 360;
        barColor = `hsla(${rainbowHue}, 100%, 70%, 0.9)`;
    }

    ctx.fillStyle = barColor;
    const currentWaveY = canvas.height / 2;


    /* ========================================================
       ① 音声認識中の「リアル波形」: bar.height の計算
    ======================================================== */
    if (isRecording && analyser && audioContext.state === 'running' && dataArray) {
       // ★1: 周波数データを取得
       analyser.getByteFrequencyData(dataArray);

       // ★2: step を計算
       const step = Math.floor(dataArray.length / barCount);

       bars.forEach((bar, i) => {
           // 取得した周波数データに基づいて高さを計算
           const volume = dataArray[i * step] / 255;
           let height = volume * 180 + 20;

           bars[i].height = height;
       });
    }


    /* ========================================================
       ② TTS応答中（isTtsSpeaking）: bar.height の計算
    ======================================================== */
    else if (isTtsSpeaking) {
      bars.forEach((bar, i) => {
          // サイン波に基づいたアニメーションの高さを計算
          const waveAmplitude = 100;
          const waveFrequency = 0.05;
          const waveOffset = Math.sin(Date.now() * 0.005 + i * waveFrequency) * waveAmplitude;
          let height = 10 + Math.abs(waveOffset);

          bars[i].height = height;
      });
    }


    /* ========================================================
       ③ 待機中・AI応答中（波形は固定）: bar.height の計算
    ======================================================== */
    else {
        bars.forEach((bar, i) => {
            // 固定値の高さを設定
            let height = 10;  
            bars[i].height = height;
        });
    }

    /* ========================================================
       ④ 描画（一元化）
    ======================================================== */
    bars.forEach(bar => {
        // 計算された bars[i].height を使ってバーを描画
        ctx.fillRect(bar.x, currentWaveY - bar.height / 2, BAR_WIDTH - 2, bar.height);
    });

    animationFrameId = requestAnimationFrame(animateBars);
}

// function animateBars() {
//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // 色決定ロジック（あなたのまま）
//     let barColor = currentWaveColor;
//     if (currentWaveColor === 'standby_transition') {
//         const segmentDuration = 890;
//         const totalDuration = STATUS_TRANSITION_COLORS.length * segmentDuration;
//         const elapsed = performance.now() % totalDuration;
//         const numSegments = STATUS_TRANSITION_COLORS.length;
//         const progress = elapsed / totalDuration;
//         const currentSegmentIndex = Math.floor(progress * numSegments);
//         const nextSegmentIndex = (currentSegmentIndex + 1) % numSegments;
//         const segmentProgress = (elapsed % segmentDuration) / segmentDuration;
//         const startRgb = hexToRgb(STATUS_TRANSITION_COLORS[currentSegmentIndex]);
//         const endRgb = hexToRgb(STATUS_TRANSITION_COLORS[nextSegmentIndex]);
//         const r = startRgb[0] + (endRgb[0] - startRgb[0]) * segmentProgress;
//         const g = startRgb[1] + (endRgb[1] - startRgb[1]) * segmentProgress;
//         const b = startRgb[2] + (endRgb[2] - startRgb[2]) * segmentProgress;
//         barColor = rgbToHex(r, g, b) + 'b3';
//     } else if (currentWaveColor === 'rainbow') {
//         rainbowHue = (rainbowHue + 3) % 360;
//         barColor = `hsla(${rainbowHue}, 100%, 70%, 0.9)`;
//     }

//     ctx.fillStyle = barColor;
//     const currentWaveY = canvas.height / 2;


//     /* ========================================================
//        ① 音声認識中の「リアル波形」 ← step の位置はここ
//     ======================================================== */
//     if (isRecording && analyser && audioContext.state === 'running' && dataArray) {
//        // ★1: 周波数データを取得
//     analyser.getByteFrequencyData(dataArray);

//     // ★2: step を計算
//     const step = Math.floor(dataArray.length / barCount);

//     bars.forEach((bar, i) => {
//         const volume = dataArray[i * step] / 255;
//         let height = volume * 180 + 20;

//         bars[i].height = height;
//         ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
//     });
//     }


//     /* ========================================================
//        ② TTS応答中（isTtsSpeaking） → サイン波アニメ
//     ======================================================== */
//     else if (isTtsSpeaking) {
//       bars.forEach((bar, i) => {
//         const waveAmplitude = 100;
//         const waveFrequency = 0.05;
//         const waveOffset = Math.sin(Date.now() * 0.005 + i * waveFrequency) * waveAmplitude;
//         let height = 10 + Math.abs(waveOffset);

//         bars[i].height = height;
//         ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
//       });
//     }


// //     /* ========================================================
// //        ③ 待機中・AI応答中（波形は固定）
// //     ======================================================== */
// //     else {
// //          // ★あなたの希望どおりの固定10
// //         bars.forEach((bar, i) => {
// //             let height = 10;
// //             bars[i].height = height;
// //             ctx.fillRect(bar.x, currentWaveY - height / 2, BAR_WIDTH - 2, height);
// //         });
// //     }


// //     /* ========================================================
// //        ④ 描画
// //     ======================================================== */
// //     bars.forEach(bar => {
// //         ctx.fillRect(bar.x, currentWaveY - bar.height / 2, barWidth - 2, bar.height);
// //     });

// //     animationFrameId = requestAnimationFrame(animateBars);
// // }

// /* ========================================================
//    ③ 待機中・AI応答中（波形は固定）
// ======================================================== */
// else {

//     bars.forEach((bar, i) => {
//         let height = 10;  // 固定値
//         bars[i].height = height;
//     });

// }

// /* ========================================================
//    ④ 描画（★ここだけで描く。二重描画禁止）
// ======================================================== */
// bars.forEach(bar => {
//     ctx.fillRect(bar.x, currentWaveY - bar.height / 2, barWidth - 2, bar.height);
// });

// animationFrameId = requestAnimationFrame(animateBars);



// 【★ 修正点1: window.addEventListenerの重複を解消し、一つに統合済み ★】
window.addEventListener("load", () => {
    resizeCanvas();
    animateBars();
    initAudioAndSTT(); // マイク初期化とSTTを自動で開始
    setStandbyStatus();
    document.getElementById('ui').style.opacity = 1;
});
window.addEventListener("resize", resizeCanvas);
window.addEventListener("orientationchange", () => {
    // 回転後の値が安定してから再計算
    setTimeout(resizeCanvas, 300);
});



/* --------------------------------------------------------------------------------- */
/* 2. 感情・色判定ロジック                                                            */
/* --------------------------------------------------------------------------------- */


// グローバル変数 (実装済みのものと仮定)
// let isMusicPlayerEnabled = true;
let isMusicPlayerEnabled = true; // 音楽再生機能を初期状態でONとする
// ...




/**
 * AIの感情に対応した音楽をYouTube Musicで再生する
 * @param {string} emotion 感情の種別 ('Rage', 'SuperHappy' など)
 * @param {string} text AIの応答テキスト
 * 
 * 
 */
function playEmotionMusic(emotion, text) {
    if (!isMusicPlayerEnabled) {
        console.log("ミュージックプレイヤー機能が無効のため、再生をスキップします。");
        // 音楽機能を無効にした場合に限り、再生を停止
        stopEmotionMusic();
        return;
    }

    let query = '';
    
    switch (emotion) {
        case 'Rage':
            query = `本気の怒りや絶望のロック`;
            break;
        case 'SuperHappy':
            query = `最高にハッピーなポップヒット`;
            break;
        case 'Anger':
            query = `激しいロックや怒りを鎮めるクラシック`;
            break;
        case 'Sadness':
            query = `心が癒されるバラード`;
            break;
        case 'Negative':
            query = `落ち着くアンビエント`;
            break;
        case 'Positive':
            query = `元気が出るアップテンポ`;
            break;
        default:
            query = `穏やかなリラックスミュージック`;
            break;
    }
    
    // 音楽再生を開始
    console.log(`[${emotion}] の感情に基づいて、YouTube Musicプレイヤーサーバーへの「${query}」の再生をリクエストをロードします。`);
    
    // 📢 【修正箇所】: 独自のサーバーAPIを呼び出す
    // fetch('/api/play-music', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json'
    //     },
    //     body: JSON.stringify({ 
    //         musicQuery: query, // サーバーが理解できるキー名に変更
    //         intentType: 'CONVERSATIONAL_RADIO'
    //     })
    // })
    // .then(response => {
    //     if (response.ok) {
    //         console.log("音楽再生リクエストをサーバーに送信しました。");
    //     } else {
    //         console.error(`音楽再生リクエスト失敗: サーバー応答コード ${response.status}`);
    //     }
    // })
    // .catch(error => {
    //     console.error("ネットワークエラー: サーバーへの接続に失敗しました。", error);
    // });

    // 元の youtube_music:play は削除またはコメントアウト
    // youtube_music:play({ 
    //     query: query,
    //     media_intent_type: 'CONVERSATIONAL_RADIO' 
    // });
// }
// function playEmotionMusic(emotion, text) {
//     if (!isMusicPlayerEnabled) {
//         console.log("ミュージックプレイヤー機能が無効のため、再生をスキップします。");
//         return;
//     }

//     let query = '';
    
//     switch (emotion) {
//         case 'Rage':
//             query = `本気の怒りや絶望のロック`;
//             break;
//         case 'SuperHappy':
//             query = `最高にハッピーなポップヒット`;
//             break;
//         case 'Anger':
//             query = `激しいロックや怒りを鎮めるクラシック`;
//             break;
//         case 'Sadness':
//             query = `心が癒されるバラード`;
//             break;
//         case 'Negative':
//             query = `落ち着くアンビエント`;
//             break;
//         case 'Positive':
//             query = `元気が出るアップテンポ`;
//             break;
//         default:
//             query = `穏やかなリラックスミュージック`;
//             break;
//     }
    
//     // 音楽再生を開始
//     console.log(`[${emotion}] の感情に基づいて、YouTube Musicで「${query}」を再生します。`);
    
//     // 📢 YouTube Music API呼び出し 
//     // ※このコード自体は実行環境に依存するため、ここではツール呼び出しの形式で提示
//     youtube_music:play({
//         query: query,
//         media_intent_type: 'CONVERSATIONAL_RADIO' // 感情やムードはラジオ/ミックスで再生するのが最適
//     });
// }

/**
 * 音楽を停止する (YouTube Musicには直接的な停止APIがないため、ここでは一時的に空の関数とします)
 * 連続再生を避けるためには、新しい感情が検出されるたびに新しいプレイリスト/ラジオに上書きするのが一般的です。
 */
// 📢 【修正箇所】: 独自のloadYouTubePlayer関数を呼び出す
    loadYouTubePlayer(query);
}

/**
 * 音楽を停止する (埋め込みプレイヤーを空にする)
 */
function stopEmotionMusic() {
    const container = document.getElementById("musicPlayerContainer");
    if (container) {
        container.innerHTML = '';
        container.style.opacity = 0; // プレイヤーを非表示に戻す
        console.log("YouTube埋め込みプレイヤーを停止（クリア）しました。");
        console.log("現在、YouTube Musicの再生を明示的に停止するAPIはありません。新しい感情の曲が、前の曲を上書き再生します。");
    }
}
// function stopEmotionMusic() {
//     console.log("現在、YouTube Musicの再生を明示的に停止するAPIはありません。新しい感情の曲が、前の曲を上書き再生します。");
// }
/**
 * AIの回答テキストに基づいて波形の色を変更する関数
 * @param {string} responseText LLMからの回答テキスト
 */
function setWaveColorBasedOnResponse(responseText) {
    const text = responseText.toLowerCase();

    // 絵文字抽出関数（ローカルまたはグローバルで定義されている前提）
    const extractEmojis = (t) => {
        const emojiRegex = /\p{Emoji_Presentation}|\p{Emoji}\p{Emoji_Modifier}*|\p{Emoji_Component}|\u200d/gu;
        const matches = t.match(emojiRegex);
        return matches ? matches.join('') : '';
    };

    // 判定ロジックは提示された内容を維持（優先順位順）

    /* setWaveColorBasedOnResponse 関数内の該当部分を修正 */

    // ... (前略) ...
    // 1. 【本気の怒り・裏切り (紫)】
    const rageKeywords = ['裏切り', '許さない', '報復', 'どうしてくれる', '絶交', '失望'];
    const rageEmojis = ['😡', '😠', '🤬', '👿', '😾', '💀', '🔪', '💣'];
    if (rageKeywords.some(k => text.includes(k)) || rageEmojis.some(e => text.includes(e))) {
        currentWaveColor = WAVE_COLORS.rage;
        console.log("波形の色を【本気の怒り・裏切り】の紫に変更しました。");
            if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ 本気の怒り・裏切りの曲を再生
            playEmotionMusic('Rage', responseText); // ★追加★// playEmotionMusic('Rage'); 
        }
        return;
    }

    // 5. 【最高にハッピー (レインボー)】
    const superHappyKeywords = ['最高にハッピー', '神', '究極', 'パーフェクト', '完璧', '奇跡', '感無量', 'レジェンド'];
    const superHappyEmojis = ['🤩', '✨', '🥳', '💯', '👑', '🥇', '🚀', '🌈', '🎉🎉🎉'];
    if (superHappyKeywords.some(k => text.includes(k)) || superHappyEmojis.some(e => text.includes(e))) {
        currentWaveColor = 'rainbow';
        console.log("波形の色を【最高にハッピー】のレインボーに変更しました。");
        if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ 最高のハッピーな曲を再生
            playEmotionMusic('SuperHappy', responseText); // ★追加★  // playEmotionMusic('SuperHappy');
        }
        return;
    }

    // 2. 【お怒り (赤)】
    const angerKeywords = ['怒り', 'ふざけるな', 'やめろ', 'だめだ', '不可能だ', '違います', '否定', 'ありえない'];
    const angerEmojis = ['😤', '💢', '🔥', '💥', '👹', '😫', '😩'];
    if (angerKeywords.some(k => text.includes(k)) || angerEmojis.some(e => text.includes(e))) {
        currentWaveColor = WAVE_COLORS.anger;
        console.log("波形の色を【お怒り】の赤に変更しました。");
        if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ お怒りな曲を再生
            playEmotionMusic('Anger', responseText); // ★追加★
        }
        return;
    }

    // 3. 【悲しい・号泣 (濃いブルー)】
    const sadnessKeywords = ['悲しい', '泣く', 'ごめんなさい', 'つらい', '寂しい', '涙', '耐えられない', '最悪', 'しんどい'];
    const sadnessEmojis = ['😭', '😢', '🥺', '💧', '😥', '💔', '🌧️', '☔'];
    if (sadnessKeywords.some(k => text.includes(k)) || sadnessEmojis.some(e => text.includes(e))) {
        currentWaveColor = WAVE_COLORS.sadness;
        console.log("波形の色を【悲しい・号泣】の濃いブルーに変更しました。");
        if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ 悲しい・号泣な曲を再生
            playEmotionMusic('Sadness', responseText); // ★追加★
        }
        return;
    }

    // 4. 【ネガティブ (ブルー)】
    const negativeKeywords = ['エラー', '失敗', 'できません', '警告', '問題', '懸念', '不明', '確認', '無理', '難しい'];
    const negativeEmojis = ['😞', '😟', '😨', '🥶', '😰', '😵'];
    if (negativeKeywords.some(k => text.includes(k)) || negativeEmojis.some(e => text.includes(e))) {
        currentWaveColor = WAVE_COLORS.negative;
        console.log("波形の色を【ネガティブ】のブルーに変更しました。");
        if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ ネガティブ的なブルーになる曲を再生
           playEmotionMusic('Negative', responseText); // ★追加★
        }
        return;
    }

    // 6. 【ポジティブ (緑)】
    const positiveKeywords = ['ありがとう', '成功', '完了', '問題ありません', '良い', 'できます', '素晴らしい', '助かる', '了解', 'OK', 'ハッピー'];
    const positiveEmojis = ['😄', '😊', '😆', '👍', '👏', '✅', '🌟'];
    if (positiveKeywords.some(k => text.includes(k)) || positiveEmojis.some(e => text.includes(e))) {
        currentWaveColor = WAVE_COLORS.positive;
        console.log("波形の色を【ポジティブ】の緑に変更しました。");
        if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ ポジティブな曲を再生
            playEmotionMusic('Positive', responseText); // ★追加★
        }
        return;
    }

    // どの条件にも合致しない場合はデフォルト色に戻す
    currentWaveColor = WAVE_COLORS.default;
    console.log("波形の色をデフォルトの水色に戻しました。");
    if (isMusicPlayerEnabled) {
            // ★音楽再生ロジック★ 平常心および無心的通常モードの曲を再生
            playEmotionMusic('Default', responseText); // ★追加★
    }
}

/* JavaScriptファイル内の適切な位置に追加 */

document.addEventListener('DOMContentLoaded', () => {
    const musicToggle = document.getElementById('music-toggle-checkbox');
    
    // スイッチの初期状態を設定 (グローバル変数と同期)
    musicToggle.checked = isMusicPlayerEnabled; 

    // スイッチの変更イベントを監視
    musicToggle.addEventListener('change', () => {
        // トグルの状態が変わるたびに機能を切り替える
        toggleMusicPlayer();
    });
});

/**
 * 音楽再生機能のON/OFFを切り替える (既存の関数)
 */
function toggleMusicPlayer() {
    // グローバル変数 isMusicPlayerEnabled を反転
    isMusicPlayerEnabled = !isMusicPlayerEnabled; 
    
    // UIのフィードバック (必要に応じて)
    if (isMusicPlayerEnabled) {
        updateStatus('ミュージックプレイヤー: ON 🎶', WAVE_COLORS.positive);
    } else {
        stopEmotionMusic(); 
        updateStatus('ミュージックプレイヤー: OFF 🔇', WAVE_COLORS.negative);
    }
}
// /**
//  * 音楽再生機能のON/OFFを切り替える
//  */
// function toggleMusicPlayer() {
//     isMusicPlayerEnabled = !isMusicPlayerEnabled; // ON/OFFを反転
    
//     if (isMusicPlayerEnabled) {
//         // ONになった場合は特に何もしない（次の応答から再生開始）
//         updateStatus('ミュージックプレイヤー: ON 🎶', WAVE_COLORS.positive);
//     } else {
//         // OFFになった場合は、現在再生中の音楽を停止
//         stopEmotionMusic(); 
//         updateStatus('ミュージックプレイヤー: OFF 🔇', WAVE_COLORS.negative);
//     }
// }
/* --------------------------------------------------------------------------------- */
/* 3. 機密保持/開発者ツールの無効化 (重複を解消し整理済み)                                 */
/* --------------------------------------------------------------------------------- */

// 1. 右クリック（コンテキストメニュー）を禁止する
document.addEventListener('contextmenu', function (e) {
    e.preventDefault();
    console.log("右クリックは禁止されています。");
    return false;
});
document.body.addEventListener('contextmenu', function (e) {
    e.preventDefault();
    return false;
});

// 2. キーボードショートカットを禁止する
document.onkeydown = function (e) {
    const key = e.key;
    const lowerKey = key.toLowerCase();

    // F12キー (開発者ツール)
    if (key === 'F12' || e.keyCode === 123) {
        e.preventDefault();
        return false;
    }

    // Ctrl/Cmd/Option/Alt キー状態のチェックを容易にする
    const isCmdOrCtrl = e.ctrlKey || e.metaKey;
    const isShift = e.shiftKey;
    const isAltOrOption = e.altKey;

    // 開発者ツールのショートカット (I, J, C)
    if (
        (isCmdOrCtrl && isShift && (lowerKey === 'i' || lowerKey === 'j' || lowerKey === 'c')) || // Ctrl/Cmd + Shift + I/J/C
        (e.metaKey && isAltOrOption && (lowerKey === 'i' || lowerKey === 'j' || lowerKey === 'c')) // Cmd + Option + I/J/C (macOS)
    ) {
        e.preventDefault();
        return false;
    }

    // その他、コンテンツ保護のためのショートカット (U, S, P)
    if (isCmdOrCtrl && (lowerKey === 'u' || lowerKey === 's' || lowerKey === 'p')) {
        e.preventDefault();
        return false;
    }
};
/* --------------------------------------------------------------------------------- */
/* 4. 音声読み上げ/認識/API連携関連                                                   */
/* --------------------------------------------------------------------------------- */


/* --------------------------------------------------------------------------------- */
/* 4. 音声読み上げ/認識/API連携関連 に追加 */
/* --------------------------------------------------------------------------------- */

// ... (hexToRgb, rgbToHex, startStatusRainbow, stopStatusRainbow が定義されている前提) ...



/* --------------------------------------------------------------------------------- */
/* 4. 音声読み上げ/認識/API連携関連 に追加 (ここに追加)                                */
/* --------------------------------------------------------------------------------- */

let statusRainbowFrameId; // ステータスエリア専用のフレームID

/**
 * ステータスエリアを連続色相変化（レインボー）でアニメーションさせる
 */
function startStatusRainbow() {
    stopSequentialColorTransition(); 
    if (statusRainbowFrameId) cancelAnimationFrame(statusRainbowFrameId);
    
    function animate() {
        if (!isSpeaking) {
            stopStatusRainbow();
            return;
        }

        rainbowHue = (rainbowHue + 3) % 360; 
        const currentColor = `hsl(${rainbowHue}, 100%, 50%)`;
        
        statusArea.style.color = currentColor;
        statusArea.style.boxShadow = `0 0 20px ${currentColor}80`;

        statusRainbowFrameId = requestAnimationFrame(animate);
    }
    statusRainbowFrameId = requestAnimationFrame(animate);
}

/**
 * ステータスエリアのレインボーアニメーションを停止
 */
function stopStatusRainbow() {
    if (statusRainbowFrameId) {
        cancelAnimationFrame(statusRainbowFrameId);
        statusRainbowFrameId = null;
    }
    // 停止後にデフォルトの色を適用する（setStandbyStatusで処理されるため、ここでは不要）
}

/**
 * 静的な緑色から動的なレインボーアニメーションへ滑らかに遷移させる
 * ... (以下、startGreenToRainbowTransition関数の定義が続く) ...
/**
 * 静的な緑色から動的なレインボーアニメーションへ滑らかに遷移させる
 * @param {number} duration 遷移にかける時間 (ms)
 */
function startGreenToRainbowTransition(duration = 750) {
    // 既存のアニメーションを全て停止
    stopSequentialColorTransition(); 
    stopStatusRainbow();
    
    // 遷移開始色：緑
    const startColor = '#00ffaa'; 
    // 遷移終了色：レインボーの開始色 (hue 0 = 赤)
    const endColor = '#FF0000'; 
    
    const startTime = performance.now();
    const startRgb = hexToRgb(startColor);
    const endRgb = hexToRgb(endColor);

    function interpolate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(1, elapsed / duration);

        // RGBを補間
        const r = startRgb[0] + (endRgb[0] - startRgb[0]) * progress;
        const g = startRgb[1] + (endRgb[1] - startRgb[1]) * progress;
        const b = startRgb[2] + (endRgb[2] - startRgb[2]) * progress;

        const currentColor = rgbToHex(r, g, b);

        statusArea.style.color = currentColor;
        statusArea.style.boxShadow = `0 0 20px ${currentColor}80`;

        if (progress < 1) {
            requestAnimationFrame(interpolate);
        } else {
            // トランジション完了後、連続レインボーをシームレスに開始
            startStatusRainbow();
        }
    }
    requestAnimationFrame(interpolate);
}


// DOM要素の取得
const statusArea = document.getElementById("status-area");
const sendBtn = document.getElementById("sendBtn");
const input = document.getElementById("messageInput");
const ui = document.getElementById('ui');
const tapArea = document.getElementById('tapArea');

// API設定
const LLM_API_URL = "https://atjmuwnwmtjw-hello.hf.space/llm/generate";
const MQTT_API_URL = "https://atjmuwnwmtjw-hello.hf.space/iot/control";

// 状態管理変数
const synth = window.speechSynthesis;
let audioContext, analyser, mediaStream;
let recognition = null;
let currentTextToSpeak = '';

// --- ヘルパー関数 (色の補間) ---
function hexToRgb(hex) {
    const bigint = parseInt(hex.slice(1), 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return [r, g, b];
}

function rgbToHex(r, g, b) {
    const toHex = (c) => ('0' + Math.max(0, Math.min(255, c)).toString(16)).slice(-2);
    return '#' + toHex(Math.round(r)) + toHex(Math.round(g)) + toHex(Math.round(b));
}

function startColorTransition(startColor, endColor, duration = 2000) {
    const startTime = performance.now();
    const startRgb = hexToRgb(startColor);
    const endRgb = hexToRgb(endColor);

    function interpolate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(1, elapsed / duration);

        const r = startRgb[0] + (endRgb[0] - startRgb[0]) * progress;
        const g = startRgb[1] + (endRgb[1] - startRgb[1]) * progress;
        const b = startRgb[2] + (endRgb[2] - startRgb[2]) * progress;

        const currentColor = rgbToHex(r, g, b);

        statusArea.style.color = currentColor;
        statusArea.style.boxShadow = `0 0 20px ${currentColor}80`;

        if (progress < 1) {
            requestAnimationFrame(interpolate);
        }
    }
    requestAnimationFrame(interpolate);
}

/* ---------- UI helpers ---------- */

function updateStatus(message, color = '#00ffff') {
    statusArea.innerHTML = message;
    statusArea.style.color = color;
    statusArea.style.boxShadow = `0 0 20px ${color}80`;
}

function setStandbyStatus() {
    stopSequentialColorTransition(); // 待機時はアニメーション停止
    const standbyMsg = `
    
        イマジナリーナンバー
        通称GAIイマさんAI
        AIアシスタント待機中...
    `;
    updateStatus(standbyMsg.trim(), '#00ffff');
}

/* ---------- TTS (Speech Synthesis) ---------- */

// 【外部定義】絵文字抽出関数
const extractEmojis = (text) => {
    const emojiRegex = /\p{Emoji_Presentation}|\p{Emoji}\p{Emoji_Modifier}*|\p{Emoji_Component}|\u200d/gu;
    const matches = text.match(emojiRegex);
    return matches ? matches.join('') : '';
};


/**
 * LLM応答など、AIからの正式な応答を読み上げ、終了後にSTTを再起動する
 * @param {string} text 読み上げるテキスト
 */
function speak(text) {
    if (!text) return;

    currentTextToSpeak = text;

    if (synth.speaking) synth.cancel();

    isSpeaking = true;

    // LLM応答に基づいて波形の色を設定
    // setWaveColorBasedOnResponse(text);
    // ★修正点1 (最重要)★: TTS開始より前に、感情に基づいて音楽再生と波形の色を設定
    // これにより、音楽のロード/再生開始がTTSの直前に行われ、体験が向上する
    setWaveColorBasedOnResponse(text); // この中で playEmotionMusic() が呼ばれる

    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'ja-JP';
    u.rate = 1.0;

    u.onstart = () => {
        isTtsSpeaking = true; // ★修正点1: TTS開始時に波形アニメーションをONにする
        // startSequentialColorTransition() に分岐するロジックを信用し、startSequentialColorTransition(STATUS_TRANSITION_COLORS, 500); // ステータス色アニメーション開始
        // ここからは startStatusBlink() のみを呼び出す
        startStatusBlink();
        const display = text.length > 20 ? text.substring(0, 20) + '...' : text;
        const onlyEmojis = extractEmojis(text); // 絵文字抽出
        
        const formattedStatus = `
        ---==(_____[　イマジナリーナンバー通称GAIイマさんAI応答:  ?&!! ${onlyEmojis}　]_____)==--- __(V._.V)__
                      「${display}」
        `;
        statusArea.innerHTML = formattedStatus.trim();
    };
    
    u.onend = () => {
        isTtsSpeaking = false; // ★修正点2: TTS終了時に波形アニメーションをOFFにする
        isSpeaking = false;
        // TTS終了時に連続色遷移アニメーションを停止し、待機状態に戻す
        stopSequentialColorTransition();
        stopStatusRainbow(); // ★追加★
        setStandbyStatus();
        input.value = '';
        // TTS終了後、波形の色をデフォルトに戻す
        currentWaveColor = WAVE_COLORS.default;

        // TTS終了後、STTが停止していれば自動で再起動を試みる
        if (recognition && !isRecording) {
            try {
                recognition.start();
            } catch (e) {
                console.warn('Recognition restart failed after TTS:', e);
            }
        }
    };
    u.onerror = (e) => {
        isTtsSpeaking = false; // ★修正点3: エラー発生時も波形アニメーションをOFFにする
        console.error('TTS error:', e);
        isSpeaking = false;
        currentTextToSpeak = '';
        setStandbyStatus();
        stopStatusRainbow(); // ★追加★
        input.value = '';
        currentWaveColor = WAVE_COLORS.default;
    };

    synth.speak(u);
}

/**
 * テキスト入力時の即時プレビュー用読み上げ関数
 * @param {string} text 読み上げるテキスト
 */
function speakSentence(text) {
    if (text.trim() === '' || text === currentTextToSpeak) {
        return;
    }

    if (synth.speaking) {
        synth.cancel();
    }

    currentTextToSpeak = text;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ja-JP';
    utterance.rate = 1.0;

    utterance.onstart = () => {
        isSpeaking = true;
        const display = text.length > 20 ? text.substring(0, 20) + '...' : text;
        updateStatus(`文章を読み上げ中: 「${display}」`, '#00ffaa');
        // プレビュー読み上げ中も波形を動かすため、一時的にcurrentWaveColorをポジティブに設定
        currentWaveColor = WAVE_COLORS.positive;
    };

    utterance.onend = () => {
        isSpeaking = false;
        setStandbyStatus();
        currentWaveColor = WAVE_COLORS.default; // 終了したらデフォルトに戻す
    };

    utterance.onerror = (event) => {
        console.error('Speech Synthesis Error:', event);
        isSpeaking = false;
        updateStatus('読み上げエラーが発生しました', '#ff0000');
        currentWaveColor = WAVE_COLORS.default;
    };

    synth.speak(utterance);
}

/* ---------- Speech Recognition (Browser STT) & Audio Init ---------- */

function restartRecognition() {
    isRecording = false;

    // TTSが動作中でなければ、待機状態に戻す
    if (!synth.speaking) {
        isSpeaking = false;
        setStandbyStatus();
    }

    setTimeout(() => {
        try {
            // 既に認識が開始されている場合は何もしない
            if (!isRecording && !synth.speaking && recognition) recognition.start();
        } catch (e) {
            if (e.name !== 'InvalidStateError') {
                console.warn('Recognition start failed:', e);
            }
        }
    }, 500);
}

function startBrowserRecognition() {
    if (isRecording) return;

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        updateStatus('Error: Speech Recognition not supported in this browser.', '#ff0000');
        return;
    }

    if (recognition) {
        recognition.stop();
        recognition = null;
    }

    recognition = new (window.webkitSpeechRecognition || window.SpeechRecognition)();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'ja-JP';

    recognition.onstart = () => {
        isRecording = true;
        isSpeaking = true; // 録音中は波形を動かすために一時的にtrue
        const standbyMsg = `
            Listening...
            話しかけてください...！
        `;
        updateStatus(standbyMsg.trim(), '#ffff00');
        startColorTransition('#ffff00', '#00ffaa', 2000);
        input.value = '';
        if (synth.speaking) synth.cancel();
    };

    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        input.value = finalTranscript || interimTranscript;
    };
    
    // 【★ 修正点4: 冗長なrestartRecognitionの定義を削除し、グローバル関数に依存させる済み ★】
    recognition.onend = () => {
        isRecording = false;

        // TTSが動作していない場合に限り isSpeaking を false に
        if (!synth.speaking) {
            isSpeaking = false;
        }

        const finalPrompt = input.value.trim();

        if (finalPrompt && finalPrompt.length > 1 && !finalPrompt.startsWith("話しかけてください") && !finalPrompt.startsWith("イマジナリーナンバー 通称GAIイマさんAI応答:")) {
            updateStatus('Processing response...', '#00ffaa');
            // LLM処理中にSTTが自動で再起動しないように、.finallyでrestartRecognitionを呼ぶ
            processRecognitionResult(finalPrompt).finally(() => {
                // TTSが終了した後に再起動させる (speak関数内のonendでも実施されるため冗長ではあるが念のため)
                if (!synth.speaking) {
                    restartRecognition();
                }
            });
        } else {
            // 発話がなかったか、短すぎた場合
            input.value = '';
            restartRecognition();
        }
    };

    recognition.onerror = (event) => {
        isRecording = false;
        console.error('Speech Recognition Error:', event.error);

        if (event.error !== 'not-allowed' && event.error !== 'aborted') {
            restartRecognition();
        } else if (event.error === 'aborted') {
            restartRecognition();
        } else {
            updateStatus('Error: Microphone permission denied or failed.', '#ff0000');
        }
    };

    try {
        recognition.start();
    } catch (e) {
        console.warn('Initial recognition start failed:', e);
    }
}

async function initAudioAndSTT() {
    if (analyser) {
        startBrowserRecognition();
        return;
    }
    updateStatus('Requesting microphone access...');

    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;

        dataArray = new Uint8Array(analyser.frequencyBinCount);

        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const sourceNode = audioContext.createMediaStreamSource(mediaStream);

        sourceNode.connect(analyser);

        startBrowserRecognition();

        updateStatus('Listening...', '#ffff00');
    } catch (e) {
        console.error('Audio initialization failed:', e);
        updateStatus('Error: Microphone access denied or failed to initialize.', '#ff0000');
    }
}

/**
 * FastAPI/MQTTバックエンドにコマンドを送信する関数
 */
async function sendIoTCommand(command) {
    updateStatus(`Executing IoT command: ${command}...`, '#00ffaa');

    try {
        const response = await fetch(MQTT_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: command })
        });

        const data = await response.json();

        if (response.ok) {
            const successMsg = `承知しました。${command === 'ON' ? '電気をつけました' : '電気を消しました'}。`;
            speak(successMsg);
        } else {
            const detail = data.detail || "サーバーエラー";
            const errorMsg = `エラーが発生しました。IoTコマンド '${command}' の実行に失敗しました。詳細: ${detail}`;
            speak(errorMsg);
        }
    } catch (error) {
        const networkErrorMsg = `🔴 ネットワークエラー: IoTバックエンドサーバーに接続できません (${error.message})`;
        speak(networkErrorMsg);
    }
}

/* ---------- 統合されたメイン処理関数 (IoT or LLM) ---------- */

async function processRecognitionResult(finalPrompt) {
    // 1. IoTコマンドの判定と振り分け
    const lowerPrompt = finalPrompt.toLowerCase();
    let iotCommand = null;

    if ((lowerPrompt.includes('ライト') || lowerPrompt.includes('電気')) && (lowerPrompt.includes('つけ') || lowerPrompt.includes('オン') || lowerPrompt.includes('点け'))) {
        iotCommand = 'ON';
    } else if ((lowerPrompt.includes('ライト') || lowerPrompt.includes('電気')) && (lowerPrompt.includes('けし') || lowerPrompt.includes('オフ') || lowerPrompt.includes('消し'))) {
        iotCommand = 'OFF';
    }

    if (iotCommand) {
        await sendIoTCommand(iotCommand);
        return;
    }

    // 2. LLM応答生成（IoTコマンドでなかった場合）
    await generateAndSpeakResponse(finalPrompt);
}

/* ---------- LLM (Gemini) API & TTS 連携 ---------- */

/**
 * LLM APIにリクエストを送信し、応答を読み上げる
 * @param {string} prompt ユーザーのプロンプト
 */
async function generateAndSpeakResponse(prompt) {
    
    // ★修正点1: 波形の色を「待機中色遷移」モードに設定★
    currentWaveColor = 'standby_transition';
    /* ★決定的なバグ修正: 関数定義の途中で閉じられていた括弧を削除し、ロジック全体を関数内に収める */
    // ★修正★ LLM生成中はアニメーションを維持するため isSpeaking を true に設定
    isSpeaking = true;
    // 状態を「生成中」に設定し、色を #00ffaa に設定
    // 状態を「生成中」に設定し、色を #00ffaa (緑) に設定
    updateStatus('Generating response (via FastAPI)...', '#00ffaa'); // ★開始色を緑に設定★

    rainbowHue = 0; // 色相をリセットして、常に緑から開始するようにする
    
    // ★修正点: 緑からレインボーへ自動で切り替わるトランジションを開始★
    startGreenToRainbowTransition(750); 
    
    // ユーザープロンプトから不要なプレフィックスを除去（念のため）
// ... (以降のコードは変更なし) ...
    // ユーザープロンプトから不要なプレフィックスを除去（念のため）
    const cleanedPrompt = prompt.replace(/^イマジナリーナンバー 通称GAIイマさんAI応答:\s*/, '').trim();
    if (!cleanedPrompt) {
        isSpeaking = false; 
        // ★修正★ エラー終了時に波形色もリセット
        currentWaveColor = WAVE_COLORS.default;
        speak("すみません、何も聞こえませんでした。もう一度話しかけてください。");
        return;
    }

    const systemInstruction = "あなたは「イマジナリーナンバー 通称GAIイマさん」という名前のKS-903model8800-a1-90dという音声アシスタントです。ユーザーの質問に日本語で、簡潔かつ丁寧に答えてください。";

    const payload = {
        prompt: cleanedPrompt,
        contents: [{ parts: [{ text: cleanedPrompt }] }],
        systemInstruction: { parts: [{ text: systemInstruction }] },
        tools: [{ "google_search": {} }],
    };

    const MAX_RETRIES = 3;
    let responseText = "エラーが発生しました。イマジナリーナンバー 通称GAIイマさんAIのKS-903model8800-a1-90d応答を取得できませんでした。";

    for (let i = 0; i < MAX_RETRIES; i++) {
        try {
            const response = await fetch(LLM_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status} Error.` }));
                throw new Error(`FastAPI Error! Status: ${response.status}. Detail: ${errorData.detail}`);
            }

            const result = await response.json();

            if (result && result.text) {
                responseText = result.text;
                break;
            } else {
                throw new Error("Empty response or invalid JSON structure from FastAPI.");
            }

        } catch (e) {
            console.error(`FastAPI call error on attempt ${i + 1}:`, e);
            if (i === MAX_RETRIES - 1) {
                responseText = "エラーが発生しました。イマジナリーナンバー 通称GAIイマさんAIKS-903model8800-a1-90dの応答を取得できませんでした。Generaltebバックエンドサーバー (ポート8001) の実行状態とAPIキーを確認してください。";
            } else {
                const delay = 2 ** i * 1000 + Math.random() * 500;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    // LLM応答取得後: 応答の感情に基づいて波形の色を再設定する
    // ★追加★ TTS開始前に、波形の色を感情に基づいて判定し直す
    setWaveColorBasedOnResponse(responseText);

    updateStatus('Speaking response...', '#ffd000ff');
    // isSpeakingはそのままtrueを維持し、speak()内のonendでfalseになる
    speak(responseText);

    return Promise.resolve();
}

/* ---------- イベントハンドラの統合と定義 ---------- */

// テキスト入力欄のイベントを追加 (Enterキーで処理)
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();

        const textPrompt = input.value.trim();

        if (textPrompt) {
            // 音声認識が実行中の場合は強制停止
            if (recognition && isRecording) {
                recognition.stop();
            }
            // TTSをキャンセル（即時読み上げを停止）
            if (synth.speaking) synth.cancel();

            updateStatus('Processing text input...', '#ffff00');

            // LLM処理を実行
            processRecognitionResult(textPrompt).catch(error => {
                console.error("Text input processing failed:", error);
            });
        }
    }
});

// テキスト入力のたびに現在の内容を読み上げる機能の追加 (TTS即時プレビュー)
input.addEventListener('input', (event) => {
    const currentText = input.value.trim();

    // 音声認識が実行中でない、かつ、AIが応答中でない場合にのみ実行
    // かつ、現在のテキストが読み上げ中のテキストと異なる場合
    if (!isRecording && !isSpeaking && currentText.length > 0 && currentText !== currentTextToSpeak) {
        speakSentence(currentText);
    } else if (currentText.length === 0 && synth.speaking) {
        // テキストが全て削除され、かつ読み上げ中の場合はキャンセルして待機状態に戻す
        synth.cancel();
        isSpeaking = false;
        setStandbyStatus();
    }
});

// リセットボタンの機能 (STTとTTSの強制停止と再起動)
sendBtn.addEventListener("click", () => {
    if (recognition) {
        recognition.stop();
        recognition = null;
        // isRecordingはonendでfalseになるが、即時リセットのため手動でも設定
        isRecording = false;
    }
    if (synth.speaking) synth.cancel();

    // isSpeakingを強制的にfalseに
    isSpeaking = false;
    currentWaveColor = WAVE_COLORS.default;
    stopSequentialColorTransition(); // 念のため色遷移も停止

    initAudioAndSTT();
    updateStatus('リセットしました。マイク入力を開始しています...');
});


// UI トグル機能 (画面タップ)
let uiVisible = true;
tapArea.addEventListener('click', (e) => {
    // リセットボタンへのタップは無視
    if (e.target.closest('#input-controls')) {
        return;
    }

    uiVisible = !uiVisible;
    if (uiVisible) {
        ui.style.opacity = 1;
    } else {
        ui.style.opacity = 0;
    }
});