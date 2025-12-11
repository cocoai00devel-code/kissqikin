// HandRecognitionLogic.js
import { Collection } from './collections_lite.js'; // Pythonのdequeに相当する簡易クラス

// ===============================================
// ユーティリティ関数 (face_hand.py から移植)
// ===============================================

// 時刻取得 (Python: get_tm_int)
export function getTmInt() { return Date.now(); }

// 正規化処理 (Python: pre_process_landmark)
export function preProcessLandmark(landmarkList) {
    if (landmarkList.length === 0) return [];
    
    // 相対座標に変換 (基準点: 0番目のランドマーク)
    const tempLandmarkList = landmarkList.map(p => [...p]); // deep copy
    const [baseX, baseY, baseZ] = tempLandmarkList[0];

    for (let i = 0; i < tempLandmarkList.length; i++) {
        tempLandmarkList[i][0] = tempLandmarkList[i][0] - baseX;
        tempLandmarkList[i][1] = tempLandmarkList[i][1] - baseY;
        // Z軸はそのまま利用 (スケールはPython版の *200 を適用せず、TensorFlow.js側で処理すると仮定)
        // Python版ではZ軸が正規化前に200倍されているが、ここでは正規化に任せる
        tempLandmarkList[i][2] = tempLandmarkList[i][2] - baseZ; 
    }

    // 1次元リストに変換
    const flatList = tempLandmarkList.flat();

    // 正規化 [-1, 1]
    let maxValue = 0.0;
    flatList.forEach(value => {
        const absValue = Math.abs(value);
        if (absValue > maxValue) {
            maxValue = absValue;
        }
    });

    if (maxValue === 0.0) return flatList.fill(0);

    const normalizedList = flatList.map(value => value / maxValue);
    return normalizedList;
}

// ランドマークをピクセル座標に変換 (Python: calc_landmark_list)
export function calcLandmarkList(landmarks, width, height) {
    const landmarkList = [];
    for (const landmark of landmarks) {
        // x, y は正規化座標(0-1.0)なので、幅と高さをかけてピクセル座標に変換
        const x = Math.min(Math.floor(landmark.x * width), width - 1);
        const y = Math.min(Math.floor(landmark.y * height), height - 1);
        const z = landmark.z; // Z軸はそのまま保持
        landmarkList.push([x, y, z]);
    }
    return landmarkList;
}

// 外接矩形の計算 (Python: calc_bounding_rect)
export function calcBoundingRect(landmarkList) {
    let xMin = Infinity, yMin = Infinity;
    let xMax = -Infinity, yMax = -Infinity;

    for (const [x, y, z] of landmarkList) {
        xMin = Math.min(xMin, x);
        yMin = Math.min(yMin, y);
        xMax = Math.max(xMax, x);
        yMax = Math.max(yMax, y);
    }
    // 外接矩形をバッファ分広げる
    const margin = 20; 
    return [
        Math.max(0, xMin - margin),
        Math.max(0, yMin - margin),
        Math.min(960, xMax + margin),
        Math.min(540, yMax + margin)
    ];
}

// ランドマークの平均座標計算 (Python: calc_landmarks_ave)
export function calcLandmarksAve(landmarks) {
    let sumX = 0, sumY = 0, sumZ = 0;
    for (const landmark of landmarks) {
        sumX += landmark.x;
        sumY += landmark.y;
        sumZ += landmark.z;
    }
    const count = landmarks.length;
    return [sumX / count, sumY / count, sumZ / count];
}

// 指先の座標リストを取得 (Python: get_landmark_tips)
export function getLandmarkTips(landmarks) {
    // 指先のインデックス: [4, 8, 12, 16, 20]
    return [
        landmarks[4].x, landmarks[4].y,
        landmarks[8].x, landmarks[8].y,
        landmarks[12].x, landmarks[12].y,
        landmarks[16].x, landmarks[16].y,
        landmarks[20].x, landmarks[20].y
    ];
}

// ===============================================
// Variablesクラス (hand.py から移植)
// ===============================================

/**
 * Pythonの Variables クラスに相当
 */
export class Variables {
    constructor(outNumber) {
        this._hand_sign_id = outNumber;
        this._pre_hand_number = outNumber;
        this._point_history = new Collection(12); // deque(maxlen=12)に相当
        this._move_id = 0;
        this._move_lock_tm = getTmInt();
        this._input_letters = "";
        this._letter_lock_id = outNumber;
        this._letter_lock_tm = getTmInt();
        this._del_lock_tm = getTmInt();
    }
    
    // 省略されていたプロパティ
    get hand_sign_id() { return this._hand_sign_id; }
    set hand_sign_id(val) { this._hand_sign_id = val; }
    get input_letters() { return this._input_letters; }
    set input_letters(val) { this._input_letters = val; }
    get del_lock_tm() { return this._del_lock_tm; }
    set del_lock_tm(val) { this._del_lock_tm = val; }
    // ... 他のプロパティも同様に移植 ...

    // move_id の判定 (Python: get_move_id)
    getMoveId(handLandmarks, landmarksAve) {
        // ... Python版の移動判定ロジックをここに移植 ...
        // point_historyへの追加、移動量の計算、move_idの更新、move_lock_tmの更新
        // 複雑なため、ここではロジックの呼び出しのみ示します
    }
    
    // 指文字IDの決定ロジック (Python: fix_hand_sign_id)
    fixHandSignId() {
        // ... Python版の安定化ロジックをここに移植 ...
        // self.__pre_hand_number, self.__letter_lock_id, self.__letter_lock_tm の更新
    }
    
    // 入力文字の確定ロジック (Python: get_input_letters)
    getInputLetters(keypointClassifierLabels) {
        // ... Python版の入力確定・濁点/半濁点処理をここに移植 ...
        // hand_sign_idが確定したら input_letters を更新
        const currentSign = keypointClassifierLabels[this._hand_sign_id];

        if (currentSign === '1字削除') {
            this._input_letters = this._input_letters.slice(0, -1);
            this._del_lock_tm = getTmInt();
        } else if (currentSign === '対象外') {
            // 何もしない
        } else if (this._hand_sign_id === 85) { // 'ー'
             // ... 長音符ロジック ...
        } else if (this._hand_sign_id >= 62 && this._hand_sign_id <= 80) {
            // 濁点/半濁点/拗音 の処理
            // if (this._input_letters.length > 0) { ... }
        } else {
            this._input_letters += currentSign;
        }
    }
}

// ===============================================
// FpsCalculatorクラス (cvfpscalc.py から移植)
// ===============================================

/**
 * Pythonの CvFpsCalc クラスに相当
 */
export class FpsCalculator {
    constructor(bufferLen = 10) { // デフォルトバッファを10に設定
        this._startTick = getTmInt();
        this._diffTimes = new Collection(bufferLen); // dequeに相当
    }

    get() {
        const currentTick = getTmInt();
        const differentTime = currentTick - this._startTick; 
        this._startTick = currentTick;

        this._diffTimes.append(differentTime);

        // 合計
        const sumOfDiffTimes = this._diffTimes._list.reduce((a, b) => a + b, 0); 
        
        // FPSの計算: 1000ms / (平均経過時間)
        const fps = 1000.0 / (sumOfDiffTimes / this._diffTimes._list.length);
        
        return Math.round(fps * 100) / 100; // 小数点第2位まで (Python: round(fps, 2))
    }
}

// ===============================================
// 描画関数群 (draw.py から移植)
// ===============================================

// 画像ロード (Python: cv.imread("ama.png"))
let amaImage = null;
export async function loadAmaImage(src = 'ama.png') {
    if (amaImage) return;
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            amaImage = img;
            resolve(true);
        };
        img.onerror = () => {
             console.error('ama.pngのロードに失敗しました。顔隠し機能は無効になります。');
             resolve(false);
        };
        img.src = src;
    });
}

// ランドマークの描画 (Python: draw_landmarks)
export function drawLandmarks(ctx, landmarkList) {
    if (landmarkList.length === 0) return;

    // MediaPipe Handのランドマーク接続定義 (Python版のdraw.pyに対応)
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], 
        [0, 5], [5, 6], [6, 7], [7, 8], 
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20],
        [0, 1], [5, 9], [9, 13], [13, 17] // 手のひら
    ];
    
    // 接続線 (黒い太線と白い細線で描画)
    for (const [start, end] of connections) {
        const p1 = landmarkList[start];
        const p2 = landmarkList[end];
        if (!p1 || !p2) continue;

        // 黒い太線
        ctx.lineWidth = 6;
        ctx.strokeStyle = '#000000'; 
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();

        // 白い細線
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#FFFFFF'; 
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
    }

    // 点の描画 (白塗り、黒枠)
    for (const [x, y, z] of landmarkList) {
        ctx.lineWidth = 1;
        ctx.fillStyle = '#FFFFFF'; 
        ctx.strokeStyle = '#000000'; 
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI); 
        ctx.fill();
        ctx.stroke();
    }
}

// 矩形内の日本語表示 (Python: draw_jp_brect)
export function drawJpBrect(ctx, brect, text) {
    const [x1, y1, x2, y2] = brect;
    const padding = 10;
    
    // 外接矩形
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    
    // 日本語表示
    ctx.font = 'bold 30px sans-serif'; 
    
    // 影/縁取り（黒）
    ctx.fillStyle = 'black'; 
    ctx.fillText(text, x1 + padding + 2, y1 + 35 + 2);

    // 文字本体（白）
    ctx.fillStyle = 'white';
    ctx.fillText(text, x1 + padding, y1 + 35);
}

// 情報表示 (Python: draw_info)
export function drawInfo(ctx, fps, mode) {
    // FPSの表示
    ctx.font = '24px sans-serif';
    ctx.lineWidth = 4;
    const fpsText = `FPS: ${fps.toFixed(2)}`;
    ctx.strokeStyle = 'black';
    ctx.strokeText(fpsText, 10, 30);
    ctx.fillStyle = 'white';
    ctx.fillText(fpsText, 10, 30);

    // MODEの表示
    if (mode >= 0 && mode <= 9) {
        const modeText = `MODE: ${mode}`;
        ctx.font = '18px sans-serif';
        ctx.strokeStyle = 'black';
        ctx.strokeText(modeText, 10, 60);
        ctx.fillStyle = 'white';
        ctx.fillText(modeText, 10, 60);
    }
}

// 顔隠し (Python: overlay_Image)
export function overlayFace(ctx, faceResults, width, height) {
    if (!faceResults || !faceResults.faceLandmarks || faceResults.faceLandmarks.length === 0 || !amaImage) return;

    // FaceLandmarksの外接矩形を計算
    const landmarks = faceResults.faceLandmarks[0];
    let x_min = width, y_min = height, x_max = 0, y_max = 0;

    for (const landmark of landmarks) {
        const x = landmark.x * width;
        const y = landmark.y * height;
        x_min = Math.min(x_min, x);
        y_min = Math.min(y_min, y);
        x_max = Math.max(x_max, x);
        y_max = Math.max(y_max, y);
    }

    const rect_x = x_min;
    const rect_y = y_min;
    const rect_w = x_max - x_min;
    const rect_h = y_max - y_min;

    // 画像を描画
    ctx.drawImage(amaImage, rect_x, rect_y, rect_w, rect_h);
}

// Pythonの collections.deque を簡易的に再現
export class Collection {
    constructor(maxLength) {
        this._list = [];
        this._maxLength = maxLength;
    }
    append(item) {
        this._list.push(item);
        if (this._list.length > this._maxLength) {
            this._list.shift();
        }
    }
}