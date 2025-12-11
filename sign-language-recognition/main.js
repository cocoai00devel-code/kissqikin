// main.js

// グローバル変数
const VIDEO = document.getElementById('webcam');
const CANVAS = document.getElementById('output_canvas');
const CTX = CANVAS.getContext('2d');
const INPUT_LETTERS_DIV = document.getElementById('input-letters');
const FPS_DIV = document.getElementById('fps-info');
const MODE_DIV = document.getElementById('mode-info');

const IMG_SIZE = { width: 960, height: 540 };
const OUT_NO = 87;
const v = new Variables(OUT_NO); // hand-logic.js から Variables クラスをロード

let mode = 0;
let number = -1;
let keypointClassifier = null; // TFLiteモデルの読み込み後に格納
let keypointClassifierLabels = []; // ラベルデータ
const cvFpsCalc = new FpsCalc(10); // hand-logic.js から FpsCalc クラスをロード

// MediaPipe Hands と FaceMesh の初期化
const mpHands = window.Hands;
const mpFaceMesh = window.FaceMesh; // MediaPipe FaceMesh for JS は別途インポートが必要 (またはHandsで顔を代用)

// ここでは Hand/FaceMesh を使用しますが、Webでの性能と互換性のためにHandsのみを使用する方が簡単かもしれません。

const hands = new mpHands.Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469640/${file}`;
    }
});
hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5
});
hands.onResults(onResults); // 結果処理関数を登録

// =================================================================
// TFLiteモデルのロード (keypoint_classifier.py の移植)
// =================================================================

async function loadTFLiteModel() {
    // TFLiteモデルをWeb互換のTensorFlow.jsモデルに変換・ロード
    // 変換されたモデルファイル (model.json, weight files) がWebサーバー上に必要です。
    // 仮に 'model/keypoint_classifier_js/model.json' に配置されていると仮定します。
    try {
        keypointClassifier = await tf.loadGraphModel('./model/keypoint_classifier_js/model.json');
        console.log("TensorFlow.js Model loaded successfully.");
    } catch (e) {
        console.error("Failed to load TensorFlow.js model. Ensure model files are converted and path is correct.", e);
    }
    
    // ラベルのロード (PythonのCSV読み込みの移植)
    try {
        const response = await fetch('./model/keypoint_classifier/hand_keypoint_classifier_label.csv');
        const text = await response.text();
        keypointClassifierLabels = text.trim().split('\n').map(row => row.split(',')[0]);
        console.log("Labels loaded successfully.");
    } catch (e) {
        console.error("Failed to load labels CSV.", e);
    }
}

// =================================================================
// メイン処理ループ (face_hand.py の while True ループの移植)
// =================================================================

function onResults(results) {
    CTX.save();
    CTX.clearRect(0, 0, CANVAS.width, CANVAS.height);

    // 1. カメラ画像の描画 (MediaPipeはRGBA画像で結果を返すため、そのまま描画)
    CTX.drawImage(results.image, 0, 0, CANVAS.width, CANVAS.height);

    const fps = cvFpsCalc.get();
    FPS_DIV.textContent = `FPS: ${fps}`;
    
    // 2. キー入力処理 (Web版ではキーイベントで代替)
    // ブラウザでは `cv.waitKey(5)` のような処理はできません。
    // `mode`と`number`の変更は、キーボードイベントリスナーで別途処理する必要があります。

    // 3. 手の処理
    if (results.multiHandLandmarks && mode !== 9) {
        let left_fg = 0;
        let right_fg = 0;

        for (const [i, handLandmarks] of results.multiHandLandmarks.entries()) {
            const handedness = results.multiHandedness[i].label[0];
            const isRightHand = handedness === 'Right';
            
            // 外接矩形の計算
            const brect = calcBoundingRect(handLandmarks, CANVAS.width, CANVAS.height);
            
            // ランドマークリストの計算 (X, Y, Z)
            const landmarkList = calcLandmarkList(handLandmarks, CANVAS.width, CANVAS.height);
            
            // 前処理と正規化
            const preProcessedLandmarkList = preProcessLandmark(landmarkList);
            
            let handSignId = OUT_NO;

            if (handedness === "Left") {
                left_fg = 1;
                handSignId = OUT_NO; // 左手は削除フラグ
            } else { // Right Hand
                right_fg = 1;

                // TFLite推論 (keypoint_classifier.py の __call__ に相当)
                if (keypointClassifier) {
                    const inputTensor = tf.tensor2d([preProcessedLandmarkList], [1, preProcessedLandmarkList.length], 'float32');
                    const output = keypointClassifier.predict(inputTensor);
                    const resultIndex = output.dataSync()[0]; // 単純な分類モデルと仮定
                    handSignId = resultIndex;
                    output.dispose(); 
                }
                
                v.hand_sign_id = handSignId;

                // パターンマッチのためのランドマークチップ取得
                const landmarkTips = getLandmarkTips(handLandmarks);
                
                // 手型判定 (hand.py の get_hand_sign_id の移植)
                // v.get_hand_sign_id(landmarkTips, handLandmarks.landmark);

                // 動き判定のための平均座標計算
                const landmarksAve = calcLandmarksAve(handLandmarks);
                v.get_move_id(landmarksAve); // hand-logic.jsの移植版

                // 手型＋動きで判定 (hand.py の fix_hand_sign_id の移植)
                v.fix_hand_sign_id();
                
                // 入力判定 (hand.py の get_input_letters の移植)
                if (left_fg === 0) {
                    v.get_input_letters(keypointClassifierLabels);
                }
            }

            // 描画 (draw.py の移植)
            // drawLandmarks(CTX, landmarkList);
            drawBoundingRect(CTX, brect);
            drawInfoBrect(CTX, brect, keypointClassifierLabels[v.hand_sign_id]);
        }
        
        // 両手検出時の削除処理 (face_hand.py の移植)
        if (left_fg === 1 && right_fg === 1) {
            v.hand_sign_id = OUT_NO - 1; // 削除を示すID
            const currentTm = v.getTmInt();
            if (currentTm - v.del_lock_tm > 100 && v.input_letters.length >= 2) {
                // Pythonの v.input_letters = v.input_letters[:-2] + v.input_letters[-1] を再現
                // (例: 'あいう' -> 'あい' + 'う'の最後 'う' -> 'あいう' - これは削除ロジックではない)
                // 削除ロジックは v.input_letters = v.input_letters.slice(0, -1) が一般的
                v.input_letters = v.input_letters.slice(0, -1); 
                v.del_lock_tm = currentTm;
            }
        }
    } 
    // 手を認識していないときの全削除
    else {
        const currentTm = v.getTmInt();
        if (currentTm - v.del_lock_tm > 300) {
            v.input_letters = "";
            v.del_lock_tm = currentTm;
        }
    }

    // FPS, MODE, 入力文字の描画/更新
    INPUT_LETTERS_DIV.textContent = v.input_letters;
    MODE_DIV.textContent = `MODE: ${mode} NUM: ${number}`;

    CTX.restore();
    
    // 次のフレームを要求
    requestAnimationFrame(function() {
        if (VIDEO.readyState === VIDEO.HAVE_ENOUGH_DATA) {
            hands.send({ image: VIDEO });
        }
    });
}

// =================================================================
// ユーティリティ関数の移植 (face_hand.py から)
// =================================================================

// Pythonの select_mode の代わり (キーイベントリスナーで処理)
document.addEventListener('keydown', (event) => {
    number = -1;
    const key = event.key;
    const keyCode = event.keyCode;
    
    if (key >= '0' && key <= '9') {
        number = parseInt(key);
    } else if (key === 'n') { // n (110)
        mode = 0;
    } else if (key === 'k') { // k (107)
        mode = 1;
    } else if (key === 's') { // s (ord("s"))
        mode = 9;
    } else if (keyCode === 27) { // ESC (27)
        // ブラウザのWebカメラ処理を停止するロジックをここに記述
        console.log("ESC pressed. Stopping video.");
    }
    
    // コンソールに出力 (デバッグ用)
    console.log(`Key: ${key}, Mode: ${mode}, Number: ${number}`);
});

// calc_bounding_rect の移植
function calcBoundingRect(landmarks, width, height) {
    let minX = width, minY = height, maxX = 0, maxY = 0;
    
    for (const landmark of landmarks) {
        const x = landmark.x * width;
        const y = landmark.y * height;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }
    // Pythonの [x, y, x + w, y + h] 形式を返す
    return [Math.floor(minX), Math.floor(minY), Math.ceil(maxX), Math.ceil(maxY)];
}

// calc_landmark_list の移植 (x, y, z のリストを返す)
function calcLandmarkList(landmarks, width, height) {
    const landmarkPoint = [];
    for (const landmark of landmarks) {
        const x = Math.min(Math.floor(landmark.x * width), width - 1);
        const y = Math.min(Math.floor(landmark.y * height), height - 1);
        const z = landmark.z;
        landmarkPoint.push([x, y, z]);
    }
    return landmarkPoint;
}

// calc_landmarks_ave の移植 (正規化された座標の平均を返す)
function calcLandmarksAve(landmarks) {
    let sumX = 0, sumY = 0, sumZ = 0;
    for (const landmark of landmarks) {
        sumX += landmark.x;
        sumY += landmark.y;
        sumZ += landmark.z;
    }
    const count = landmarks.length;
    return [sumX / count, sumY / count, sumZ / count]; // [x_ave, y_ave, z_ave]
}

// pre_process_landmark の移植
function preProcessLandmark(landmark_list) {
    const temp_landmark_list = landmark_list.map(p => [...p]); // deep copy
    
    // 相対座標に変換
    const base_x = temp_landmark_list[0][0];
    const base_y = temp_landmark_list[0][1];
    const base_z = temp_landmark_list[0][2];
    
    for (let i = 0; i < temp_landmark_list.length; i++) {
        temp_landmark_list[i][0] = temp_landmark_list[i][0] - base_x;
        temp_landmark_list[i][1] = temp_landmark_list[i][1] - base_y;
        temp_landmark_list[i][2] = (temp_landmark_list[i][2] - base_z) * 200;
    }

    // 1次元リストに変換 (itertools.chain.from_iterable の代わり)
    const flattenedList = temp_landmark_list.flat();

    // 正規化
    const max_value = Math.max(...flattenedList.map(Math.abs));
    const normalizedList = flattenedList.map(n => n / max_value);

    return normalizedList;
}

// get_landmark_tips の移植 (正規化された座標から取得)
function getLandmarkTips(landmarks) {
    let upNum = 0, downNum = 0, leftNum = 0, rightNum = 0;
    let upBase = 1.0, downBase = 0.0, leftBase = 1.0, rightBase = 0.0;
    
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        if (upBase >= landmark.y) { upNum = i; upBase = landmark.y; }
        if (downBase <= landmark.y) { downNum = i; downBase = landmark.y; }
        if (leftBase >= landmark.x) { leftNum = i; leftBase = landmark.x; }
        if (rightBase <= landmark.x) { rightNum = i; rightBase = landmark.x; }
    }
    // [上端のランドマークID, 下端のランドマークID, 左端のランドマークID, 右端のランドマークID]
    return [upNum, downNum, leftNum, rightNum];
}


// =================================================================
// 描画関数の移植 (draw.py から)
// =================================================================

// draw_landmarks の移植 (CV2の代わりにCanvas APIを使用)
function drawLandmarks(ctx, landmarkList) {
    // MediaPipeの描画ユーティリティを使用するか、Canvas APIで手動で描画するのが一般的
    // 手動描画の例 (ここでは MediaPipeのユーティリティ関数を使用することを推奨します)
    /*
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    // 親指の接続線の例: 2-3, 3-4
    ctx.beginPath();
    ctx.moveTo(landmarkList[2][0], landmarkList[2][1]);
    ctx.lineTo(landmarkList[3][0], landmarkList[3][1]);
    ctx.lineTo(landmarkList[4][0], landmarkList[4][1]);
    ctx.stroke();
    // ... 他の指もすべて同様に描画 ...
    
    // キーポイントの描画
    for (let i = 0; i < landmarkList.length; i++) {
        ctx.fillStyle = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20) ? 'red' : 'white';
        ctx.beginPath();
        ctx.arc(landmarkList[i][0], landmarkList[i][1], (i === 4 || i === 8 || i === 12 || i === 16 || i === 20) ? 8 : 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    */
}

// draw_bounding_rect の移植
function drawBoundingRect(ctx, brect) {
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.strokeRect(brect[0], brect[1], brect[2] - brect[0], brect[3] - brect[1]);
}

// draw_jp_brect の移植 (認識された文字の表示 - draw.pyのPIL/ImageDrawロジックをCanvasに移植)
function drawInfoBrect(ctx, brect, handSignText) {
    if (handSignText === "") return;
    
    const infoText = handSignText;
    const fontSize = 30;
    const pos = { 
        x: Math.round((brect[0] + brect[2]) / 2), 
        y: brect[1] - 50 
    };

    ctx.font = `${fontSize}px sans-serif`; // Webで利用可能なフォント
    ctx.textAlign = 'center';
    
    // 文字の輪郭 (黒)
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.strokeText(infoText, pos.x, pos.y);

    // 文字の色 (白)
    ctx.fillStyle = 'white';
    ctx.fillText(infoText, pos.x, pos.y);
}


// =================================================================
// カメラの起動と初期化
// =================================================================

async function initCamera() {
    // TFLiteモデルとラベルをロード
    await loadTFLiteModel();
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            'video': {
                facingMode: 'user',
                width: IMG_SIZE.width,
                height: IMG_SIZE.height
            },
        });
        VIDEO.srcObject = stream;
        
        VIDEO.onloadedmetadata = () => {
            VIDEO.play();
            // MediaPipe Handsの処理を開始
            hands.send({ image: VIDEO });
        };
    } catch (e) {
        console.error("Could not access the camera: ", e);
        alert("カメラへのアクセスが必要です。許可してください。");
    }
}

// アプリケーションのエントリーポイント
initCamera();