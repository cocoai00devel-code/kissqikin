// app.js

import { 
    Variables, getTmInt, FpsCalculator, 
    preProcessLandmark, calcLandmarkList, calcBoundingRect, 
    drawLandmarks, drawJpBrect, drawInfo, overlayFace, loadAmaImage, 
    calcLandmarksAve, getLandmarkTips // hand.pyのロジックで利用するユーティリティ
} from './HandRecognitionLogic.js';

// --- 定数と変数 (face_hand.py main() の冒頭) ---
const WIDTH = 960;
const HEIGHT = 540;
const outNo = 87; // '対象外'のインデックス
const video = document.getElementById('video-feed');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const inputTextElement = document.getElementById('input-text-area');

let mode = 0; // 0: 実行モード
let v = new Variables(outNo);
let keypointClassifierLabels = [];
let keypointClassifierModel = null;
let faceLandmarker = null;
let handLandmarker = null;
let fpsCalculator = new FpsCalculator(10); 
let lastVideoTime = -1;

// --- 初期化処理 ---
async function initialize() {
    const { HandLandmarker, FilesetResolver, FaceLandmarker } = window.vision;

    // 1. ラベルのロード (Python: CSV読み込み)
    await loadLabels();

    // 2. TFLiteモデルのロード (Python: KeyPointClassifier)
    try {
        const modelPath = 'web_model/keypoint_classifier/model.json'; // 事前変換したモデル
        keypointClassifierModel = await tf.loadLayersModel(modelPath);
        // ダミー推論で高速化
        const dummyInput = tf.zeros([1, 63]); 
        keypointClassifierModel.predict(dummyInput).dispose();
        dummyInput.dispose();
    } catch (e) {
        console.error("TF.jsモデルのロードに失敗しました。ファイルパスを確認してください:", e);
        return; 
    }

    // 3. MediaPipe HandLandmarker の初期化 (Python: mp.solutions.hands)
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.create(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
    });
    
    // 4. MediaPipe FaceLandmarker の初期化 (Python: mp.solutions.face_mesh)
    faceLandmarker = await FaceLandmarker.create(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_face_geometry/float16/1/face_landmarker_v2_with_face_geometry.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        outputFaceBlendshapes: true,
        numFaces: 1
    });

    // 5. 顔隠し画像のロード
    await loadAmaImage(); 

    // 6. カメラの起動
    await setupCamera();
}

// ラベルCSVの読み込み
async function loadLabels() {
    try {
        const response = await fetch('hand_keypoint_classifier_label.csv');
        const text = await response.text();
        keypointClassifierLabels = text.split('\n')
                                       .map(label => label.trim())
                                       .filter(label => label.length > 0);
    } catch (e) {
        console.error("ラベルファイルのロードに失敗しました。ファイルパスを確認してください:", e);
    }
}

// カメラの設定 (Python: cap = cv.VideoCapture(0))
async function setupCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: WIDTH }, 
                    height: { ideal: HEIGHT },
                    facingMode: 'user'
                } 
            });
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                // 映像準備完了後にループ開始
                window.requestAnimationFrame(detectLoop);
            };
        } catch (error) {
            console.error('カメラの起動に失敗しました:', error);
        }
    }
}

// --- メインループ (Python: while True:) ---
async function detectLoop() {
    const currentTm = getTmInt();
    
    // Canvasをクリア (Python: flipと新しい画像の取得に相当)
    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    if (video.currentTime !== lastVideoTime) {
        
        // 1. FPSの計算と表示
        const fps = fpsCalculator.get();
        drawInfo(ctx, fps, mode);

        // 2. 検出処理
        const handResults = handLandmarker.detectForVideo(video, currentTm);
        const faceResults = faceLandmarker.detectForVideo(video, currentTm);
        
        // 3. 顔隠し処理
        overlayFace(ctx, faceResults, WIDTH, HEIGHT);

        let leftFg = 0; // 左手フラグ
        let rightFg = 0; // 右手フラグ

        // 4. 手のランドマーク処理
        if (handResults.landmarks.length > 0 && mode !== 9) {

            for (let i = 0; i < handResults.landmarks.length; i++) {
                const landmarks = handResults.landmarks[i];
                const handedness = handResults.handedness[i][0].categoryName;
                
                // 座標変換 (正規化座標 -> ピクセル座標)
                const landmarkList = calcLandmarkList(landmarks, WIDTH, HEIGHT);
                
                // 外接矩形の計算
                const brect = calcBoundingRect(landmarkList);

                if (handedness === "Left") {
                    leftFg = 1;
                    v.hand_sign_id = outNo; // 左手は分類しない
                } else { // Right
                    rightFg = 1;
                    
                    // --- 前処理と分類 (Python: pre_process_landmark と keypoint_classifier) ---
                    const preProcessed = preProcessLandmark(landmarks.map(l => [l.x, l.y, l.z])); // MediaPipeの座標を利用
                    
                    // 推論の実行
                    const inputTensor = tf.tensor2d([preProcessed]);
                    const outputTensor = keypointClassifierModel.predict(inputTensor);
                    const resultIndex = outputTensor.argMax(-1).dataSync()[0];
                    inputTensor.dispose();
                    outputTensor.dispose();

                    // --- 判定ロジックの実行 (hand.pyの各種メソッド) ---
                    v.hand_sign_id = resultIndex; // 最初の分類結果をセット
                    
                    const landmarkTips = getLandmarkTips(landmarks);
                    v.getHandSignId(landmarkTips, landmarks); // 座標ベースの判定を適用
                    
                    const landmarksAve = calcLandmarksAve(landmarks);
                    v.getMoveId(landmarks, landmarksAve); // 移動判定
                    v.fixHandSignId(); // IDの確定処理
                    
                    if (leftFg === 0) { // 右手のみの場合に入力文字を更新
                        v.getInputLetters(keypointClassifierLabels); 
                    }
                    
                    // 描画
                    drawLandmarks(ctx, landmarkList);
                    drawJpBrect(ctx, brect, keypointClassifierLabels[v.hand_sign_id]);
                }
            }

            // 両手検出時の削除処理 (Python: if left_fg == 1 and right_fg == 1:)
            if (leftFg === 1 && rightFg === 1) {
                if (getTmInt() - v.del_lock_tm > 300) {
                    v.input_letters = v.input_letters.slice(0, -1);
                    v.del_lock_tm = getTmInt();
                }
            }
        } 
        // 手を認識していないときの削除処理 (Python: else: ...)
        else {
            if (getTmInt() - v.del_lock_tm > 3000) {
                v.input_letters = "";
                v.del_lock_tm = getTmInt();
            }
        }
        
        // 入力文字の表示更新
        inputTextElement.innerText = v.input_letters;

        lastVideoTime = video.currentTime;
    }
    
    // ループ継続
    window.requestAnimationFrame(detectLoop);
}

// アプリケーション開始
initialize();