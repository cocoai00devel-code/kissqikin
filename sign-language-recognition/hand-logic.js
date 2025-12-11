// hand-logic.js

// Pythonのcollections.dequeの代わり
class Deque {
    constructor(maxLength) {
        this.maxLength = maxLength;
        this.data = [];
    }
    push(item) {
        this.data.push(item);
        if (this.data.length > this.maxLength) {
            this.data.shift();
        }
    }
    // Pythonのリストスライスの代わり (e.g., list(self.point_history)[0:3])
    slice(start, end) {
        return this.data.slice(start, end);
    }
    get length() {
        return this.data.length;
    }
    // Pythonのnumpy.mean(axis=0) の代わり
    static mean(points) {
        if (points.length === 0) return [0, 0, 0];
        let sumX = 0, sumY = 0, sumZ = 0;
        for (const p of points) {
            sumX += p[0];
            sumY += p[1];
            sumZ += p[2];
        }
        const count = points.length;
        return [sumX / count, sumY / count, sumZ / count];
    }
}

// =================================================================
// CvFpsCalc (cvfpscalc.py) の移植
// =================================================================

class FpsCalc {
    constructor(bufferLen = 1) {
        this._difftimes = new Deque(bufferLen);
        this._lastTime = performance.now();
    }

    get() {
        const currentTime = performance.now();
        const differentTime = currentTime - this._lastTime; // ミリ秒
        this._lastTime = currentTime;

        this._difftimes.push(differentTime);

        const sumDiffTimes = this._difftimes.data.reduce((a, b) => a + b, 0);
        const avgDiffTime = sumDiffTimes / this._difftimes.length;
        
        // FPS = 1000.0 / 平均差分時間
        const fps = 1000.0 / avgDiffTime;
        return fps.toFixed(2);
    }
}

// =================================================================
// Variables (hand.py) の移植
// =================================================================

class Variables {
    constructor(out_number) {
        this._out_number = out_number;
        this._hand_sign_id = out_number;
        this._pre_hand_number = out_number;
        this._point_history = new Deque(12);
        this._move_id = 0;
        this._move_lock_tm = this.getTmInt();
        this._input_letters = "";
        this._input_letters_position_x = 0;
        this._letter_lock_id = out_number;
        this._letter_lock_tm = this.getTmInt();
        this._del_lock_tm = this.getTmInt();
    }

    // プロパティ (Pythonの@propertyと.setterに相当)
    get hand_sign_id() { return this._hand_sign_id; }
    set hand_sign_id(val) { this._hand_sign_id = val; }
    get pre_hand_number() { return this._pre_hand_number; }
    set pre_hand_number(val) { this._pre_hand_number = val; }
    get point_history() { return this._point_history; }
    set point_history(val) { this._point_history = val; }
    get move_id() { return this._move_id; }
    set move_id(val) { this._move_id = val; }
    get move_lock_tm() { return this._move_lock_tm; }
    set move_lock_tm(val) { this._move_lock_tm = val; }
    get input_letters() { return this._input_letters; }
    set input_letters(val) { this._input_letters = val; }
    get input_letters_position_x() { return this._input_letters_position_x; }
    set input_letters_position_x(val) { this._input_letters_position_x = val; }
    get letter_lock_id() { return this._letter_lock_id; }
    set letter_lock_id(val) { this._letter_lock_id = val; }
    get letter_lock_tm() { return this._letter_lock_tm; }
    set letter_lock_tm(val) { this._letter_lock_tm = val; }
    get del_lock_tm() { return this._del_lock_tm; }
    set del_lock_tm(val) { this._del_lock_tm = val; }

    // Pythonのget_tm_int()の移植 (精度は異なりますが、相対的な時間差に使用)
    getTmInt() {
        return Date.now(); // JavaScriptではミリ秒単位のタイムスタンプを使用
    }

    // get_move_id (hand.py) の移植: 動きによる濁音・半濁音の判定ロジック
    get_move_id(landmarks_ave) {
        const currentTm = this.getTmInt();
        
        if (this.pre_hand_number !== this.hand_sign_id) {
            if (currentTm - this.move_lock_tm > 100) {
                // 初期化: 過去のポイント履歴を現在の平均点で埋める
                for (let i = 0; i < 12; i++) {
                    this.point_history.push(landmarks_ave);
                }
                this.pre_hand_number = this.hand_sign_id;
                this.move_lock_tm = currentTm;
                this.move_id = 0;
            }
        } else if (this.move_id === 0) {
            const t_v1 = 1.03, t_v2 = 1.05;
            this.point_history.push(landmarks_ave);

            // 平均座標を計算 (Deque.mean を使用)
            const sum_p1 = Deque.mean(this.point_history.slice(0, 3));
            const sum_p2 = Deque.mean(this.point_history.slice(4, 7));
            const sum_p3 = Deque.mean(this.point_history.slice(8, 11));

            // Pythonロジックの移植: X軸 (横方向) の動き
            if (sum_p1[0] < sum_p2[0] && sum_p2[0] * t_v1 < sum_p3[0] && sum_p1[0] * t_v2 < sum_p3[0]) {
                this.move_id = 2; // 濁音 (右に移動: x1 < x2 < x3)
            } 
            // Y軸 (下方向) の動き
            else if (sum_p1[1] < sum_p2[1] && sum_p2[1] * t_v1 < sum_p3[1] && sum_p1[1] * t_v2 < sum_p3[1]) {
                this.move_id = 1; // を、小文字 (下に移動: y1 < y2 < y3)
            } 
            // Y軸 (上方向) の動き
            else if (sum_p3[1] < sum_p2[1] && sum_p2[1] * t_v1 < sum_p1[1] && sum_p3[1] * t_v2 < sum_p1[1]) {
                this.move_id = 3; // 半濁音 (上に移動: y3 < y2 < y1)
            }
            
            if (this.move_id !== 0) {
                console.log("move_id : " + this.move_id);
                this.move_unlock_tm = currentTm;
            }
        } else if (currentTm - this.move_unlock_tm > 100) {
            // 動きのロックを解除
            this.move_id = 0;
            for (let i = 0; i < 12; i++) {
                this.point_history.push(landmarks_ave);
            }
        }
    }

    // get_hand_sign_id (hand.py) の移植: 分類後のパターンマッチによる微調整ロジック
    // このロジックは非常に複雑で、ここでは簡易的な構造のみ示します。
    // 実際の実装では、Pythonコードの全ての if-elif-else の条件を完全に移植する必要があります。
    get_hand_sign_id(landmark_tips, hand_landmarks) {
        // 注: hand_landmarks は MediaPipeの正規化されたランドマークオブジェクトであると仮定
        
        // 複雑なロジックを全て移植するのは非現実的であるため、ここでは最初の分岐のみを例示します。
        // Pythonコード (hand.py) の全ロジックをここに移植してください。

        /*
        if (this.hand_sign_id === 0) { // いきちつめ
            if (landmark_tips[0] === 8) {
                this.hand_sign_id = 16; // き
            } else if (hand_landmarks[4].y <= hand_landmarks[15].y) { 
                // JSではランドマークアクセスが results.multiHandLandmarks[i].landmark[4].y のようになる
                // このロジックを正しく動作させるには、hand_landmarksがPython版と同じ構造を持つ必要があります。
                this.hand_sign_id = 11; // い
            } 
            // ... 他の分岐もすべて移植 ...
        }
        */
        
        // 開発環境では、この関数内でPythonコードを参考に全ロジックを移植する必要があります。
    }


    // fix_hand_sign_id (hand.py) の移植: 動きと基本手型による最終判定ロジック
    fix_hand_sign_id() {
        if (this.move_id === 0) return;
        
        // Pythonコードの全分岐 (濁音・半濁音・小文字) をここに移植します。
        // 例:
        if (this.hand_sign_id === 14 && this.move_id === 1) { this.hand_sign_id = 54; } // お -> を
        else if (this.hand_sign_id === 15 && this.move_id === 2) { this.hand_sign_id = 56; } // か -> が
        // ... (他のすべて: 57～85)
    }

    // get_input_letters (hand.py) の移植: 文字入力と連結ロジック
    get_input_letters(keypoint_classifier_labels) {
        const currentTm = this.getTmInt();

        if (this.letter_lock_id === this.hand_sign_id) {
            if (currentTm - this.letter_lock_tm > 400) {
                // 長押しで新しい文字として追加
                this.input_letters += keypoint_classifier_labels[this.hand_sign_id];
                this.letter_lock_tm = currentTm;
            } else if (currentTm - this.letter_lock_tm > 20) {
                // 短時間で同一または関連する手型の場合、変換または追加
                const lastLetter = this.input_letters.slice(-1);
                
                // 連結・変換ロジック (Pythonコードの全分岐をここに移植)
                if (this.input_letters.length === 0) {
                    this.input_letters += keypoint_classifier_labels[this.hand_sign_id];
                } else if (lastLetter === "お" && this.hand_sign_id === 54) { // 例: お + 'をのID' -> を
                    this.input_letters = this.input_letters.slice(0, -1) + "を";
                } 
                // ... (他の濁音・半濁音・小文字の変換ロジックをすべて移植) ...
                
                else if (lastLetter !== keypoint_classifier_labels[this.hand_sign_id].slice(-1)) {
                    // 異なる文字は新しい文字として追加
                    this.input_letters += keypoint_classifier_labels[this.hand_sign_id];
                }
            }
        } else {
            this.letter_lock_id = this.hand_sign_id;
            this.letter_lock_tm = currentTm;
        }
    }
}