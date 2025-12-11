import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

# --- MediaPipeと設定の初期化 ---
mp_object_detection = mp.solutions.object_detection

# MediaPipe Object Detection モデルのクラスラベル (COCOデータセットに基づく)
OBJECT_LABELS = [
    'Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 
    'Truck', 'Boat', 'Traffic light', 'Fire hydrant', 'Stop sign', 'Parking meter', 
    'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 
    'Zebra', 'Giraffe', 'Backpack', 'Umbrella', 'Handbag', 'Tie', 'Suitcase', 
    'Frisbee', 'Skis', 'Snowboard', 'Sports ball', 'Kite', 'Baseball bat', 
    'Baseball glove', 'Skateboard', 'Surfboard', 'Tennis racket', 'Bottle', 
    'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple', 
    'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut', 
    'Cake', 'Chair', 'Couch', 'Potted plant', 'Bed', 'Dining table', 'Toilet', 
    'TV', 'Laptop', 'Mouse', 'Remote', 'Keyboard', 'Cell phone', 'Microwave', 
    'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book', 'Clock', 'Vase', 'Scissors', 
    'Teddy bear', 'Hair drier', 'Toothbrush'
]

# グローバル変数としてモデルとカメラを定義
object_detector = None
cap = None
templates = Jinja2Templates(directory="templates")

# --- アプリケーションのライフサイクル管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPIの起動時と終了時にリソースの初期化と解放を行う
    """
    global object_detector, cap
    
    print("アプリケーション起動: モデルとカメラを初期化中...")
    
    # 1. モデルの初期化
    object_detector = mp_object_detection.ObjectDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    )
    
    # 2. カメラの初期化
    cap = cv.VideoCapture(0)
    # パフォーマンスとストリームサイズを考慮した解像度設定
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
    
    yield
    
    # シャットダウン処理
    print("アプリケーション終了: リソースを解放中...")
    if cap is not None:
        cap.release()
    if object_detector is not None:
        object_detector.close()
    print("リソース解放完了。")

app = FastAPI(lifespan=lifespan)


# --- 映像ストリーム処理の Generator 関数 ---
def video_feed_generator():
    """
    Webカメラからフレームを取得し、物体検出を実行して、
    JPEG形式のバイト列としてストリームを生成するジェネレータ関数。
    """
    global object_detector, cap
    
    if cap is None or object_detector is None:
        return

    while True:
        success, image = cap.read()
        if not success:
            break

        # 映像の前処理: カメラウィンドウは開かない
        debug_image = image.copy()
        image = cv.flip(image, 1) # ミラー表示
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # 1. 物体検出の実行
        results = object_detector.process(image_rgb)
        
        image_rgb.flags.writeable = True
        
        # 2. 物体検出結果の描画
        if results.detection is not None:
            ih, iw, _ = debug_image.shape
            for detection in results.detection:
                bbox_c = detection.location_data.relative_bounding_box
                
                # 正規化された座標をピクセル値に変換
                x = int(bbox_c.xmin * iw)
                y = int(bbox_c.ymin * ih)
                w = int(bbox_c.width * iw)
                h = int(bbox_c.height * ih)
                
                class_id = detection.label_id
                score = detection.score[0]
                
                # 検出結果の描画 (OpenCVの基本機能を使用)
                label = OBJECT_LABELS[class_id] if class_id < len(OBJECT_LABELS) else 'Unknown'
                text = f'{label}: {score:.2f}'
                
                # Bounding Box
                cv.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Label & Score
                cv.putText(debug_image, text, (x, y - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Webストリーム用にJPEGにエンコード
        ret, buffer = cv.imencode('.jpg', debug_image)
        frame = buffer.tobytes()

        # 4. M-JPEG形式でクライアントに送信
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # 簡易的なフレームレート制御
        time.sleep(0.005)

# --- FastAPI ルート定義 ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    ストリーミング映像を表示するためのHTMLページを返す (templates/index.htmlを参照)
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    """
    物体検出結果を含むM-JPEGストリームを返すエンドポイント
    """
    return StreamingResponse(
        video_feed_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# --- 実行方法 ---
# 1. 'templates'フォルダ内に 'index.html' を配置
# 2. ターミナルで実行: uvicorn main:app --reload
# 3. ブラウザで http://127.0.0.1:8000 にアクセス