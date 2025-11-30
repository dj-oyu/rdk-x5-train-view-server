from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import cv2
import os
import sys
import numpy as np
import json
import ctypes
import time

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

# Classes from coco_classes.names
with open('coco_classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# YOLOv5 model path
model_path = '/app/pydev_demo/models/yolov5s_672x672_nv12.bin'  # Adjust path if needed

# Load model
models = dnn.load(model_path)

# Define ctypes structures (from test_yolov5.py)
class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_double),
        ("virAddr", ctypes.c_void_p),
        ("memSize", ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen", ctypes.c_int),
        ("shiftData", ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen", ctypes.c_int),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int),
        ("zeroPointData", ctypes.c_char_p)
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize", ctypes.c_int * 8),
        ("numDimensions", ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape", hbDNNTensorShape_t),
        ("alignedShape", hbDNNTensorShape_t),
        ("tensorLayout", ctypes.c_int),
        ("tensorType", ctypes.c_int),
        ("shift", hbDNNQuantiShift_yt),
        ("scale", hbDNNQuantiScale_t),
        ("quantiType", ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize", ctypes.c_int),
        ("stride", ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem", hbSysMem_t * 4),
        ("properties", hbDNNTensorProperties_t)
    ]

class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("ori_height", ctypes.c_int),
        ("ori_width", ctypes.c_int),
        ("score_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("nms_top_k", ctypes.c_int),
        ("is_pad_resize", ctypes.c_int)
    ]

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

get_Postprocess_result = libpostprocess.Yolov5PostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)

def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

def capture_image():
    video_device = find_first_usb_camera()
    if video_device is None:
        raise Exception("No USB camera found.")
    
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        raise Exception(f"Failed to open video device: {video_device}")
    
    # Set camera properties
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise Exception("Failed to capture image from USB camera")
    return frame

def detect_objects(image):
    """物体検出を実行し、検出結果を返す"""
    h, w = get_hw(models[0].inputs[0].properties)
    des_dim = (w, h)
    resized_data = cv2.resize(image, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)
    
    outputs = models[0].forward(nv12_data)
    
    # Postprocess
    yolov5_postprocess_info = Yolov5PostProcessInfo_t()
    yolov5_postprocess_info.height = h
    yolov5_postprocess_info.width = w
    org_height, org_width = image.shape[0:2]
    yolov5_postprocess_info.ori_height = org_height
    yolov5_postprocess_info.ori_width = org_width
    yolov5_postprocess_info.score_threshold = 0.4
    yolov5_postprocess_info.nms_threshold = 0.45
    yolov5_postprocess_info.nms_top_k = 50
    yolov5_postprocess_info.is_pad_resize = 0
    
    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(outputs[i].properties.layout)
        if len(outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:
            output_tensors[i].properties.quantiType = 2
            output_tensors[i].properties.scale.scaleData = outputs[i].properties.scale_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            
        for j in range(len(outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = outputs[i].properties.shape[j]
        
        libpostprocess.Yolov5doProcess(output_tensors[i], ctypes.pointer(yolov5_postprocess_info), i)
    
    result_str = get_Postprocess_result(ctypes.pointer(yolov5_postprocess_info))
    result_str = result_str.decode('utf-8')
    
    # Parse JSON - the result is in format '"yolov5_result": [...]'
    # We need to wrap it in braces to make valid JSON
    if result_str.startswith('"yolov5_result"'):
        result_str = '{' + result_str + '}'
        data = json.loads(result_str)
        data = data.get('yolov5_result', [])
    else:
        data = json.loads(result_str)
    
    return data, org_height, org_width


def calculate_bbox_area(bbox):
    """バウンディングボックスの面積を計算する"""
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def calculate_congestion_rate(image, use_camera=True, image_path=None):
    """
    混雑率を計算する
    - personのバウンディングボックスの面積を計算
    - ソース画像との面積比を計算して混雑率として返す
    """
    # 物体検出を実行
    detections, img_height, img_width = detect_objects(image)
    
    # 画像全体の面積
    total_image_area = img_height * img_width
    
    # personのバウンディングボックスのみを抽出
    person_detections = [d for d in detections if d['name'] == 'person']
    
    # 各personのバウンディングボックス面積を計算
    person_areas = []
    for person in person_detections:
        bbox = person['bbox']
        area = calculate_bbox_area(bbox)
        person_areas.append({
            'bbox': bbox,
            'area': area,
            'score': person['score'],
            'name': person['name']
        })
    
    # personの総面積（重複を考慮しない単純合計）
    total_person_area = sum(p['area'] for p in person_areas)
    
    # 混雑率 = personの総面積 / 画像全体の面積 * 100
    congestion_rate = (total_person_area / total_image_area) * 100 if total_image_area > 0 else 0
    
    return {
        'congestion_rate': round(congestion_rate, 2),
        'person_count': len(person_detections),
        'total_person_area': total_person_area,
        'total_image_area': total_image_area,
        'image_size': {'width': img_width, 'height': img_height},
        'person_details': person_areas
    }


def load_image_from_file(image_path: str):
    """画像ファイルを読み込む"""
    if not os.path.exists(image_path):
        raise Exception(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image: {image_path}")
    return image


@app.get("/")
def root():
    """APIのルートエンドポイント"""
    return {
        "message": "混雑率検出API",
        "endpoints": {
            "/congestion": "カメラから画像を取得して混雑率を計算",
            "/congestion?image_path=/path/to/image.jpg": "指定した画像ファイルから混雑率を計算",
            "/health": "ヘルスチェック"
        }
    }


@app.get("/health")
def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "model_loaded": models is not None}


@app.get("/congestion")
def get_congestion(image_path: Optional[str] = None):
    """
    混雑率を取得するエンドポイント
    
    Parameters:
    - image_path: 画像ファイルのパス（省略時はカメラから取得）
    
    Returns:
    - congestion_rate: 混雑率（%）
    - person_count: 検出された人数
    - total_person_area: personの総面積（ピクセル）
    - total_image_area: 画像全体の面積（ピクセル）
    - image_size: 画像サイズ
    - person_details: 各personの詳細情報
    """
    try:
        if image_path:
            # 画像ファイルから読み込み
            image = load_image_from_file(image_path)
            source = f"file: {image_path}"
        else:
            # カメラから画像を取得
            image = capture_image()
            source = "camera"
        
        # 最後に使用した画像を保存
        cv2.imwrite(LAST_PHOTO_PATH, image)
        
        result = calculate_congestion_rate(image)
        
        # 検出結果をJSONファイルに保存
        with open(LAST_DETECTION_PATH, 'w') as f:
            json.dump(result, f)
        
        result['source'] = source
        result['last_photo'] = LAST_PHOTO_PATH
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/count_people")
def count_people(image_path: Optional[str] = None):
    """人数カウントエンドポイント（後方互換性のため維持）"""
    try:
        if image_path:
            image = load_image_from_file(image_path)
        else:
            image = capture_image()
        
        detections, _, _ = detect_objects(image)
        person_count = sum(1 for d in detections if d['name'] == 'person')
        return {"people_count": person_count}
    except Exception as e:
        return {"error": str(e)}


# テスト画像のディレクトリ
TEST_IMAGES_DIR = os.path.dirname(os.path.abspath(__file__))

# 最後に撮影した画像のパス
LAST_PHOTO_PATH = os.path.join(TEST_IMAGES_DIR, "last_photo.jpg")
# 最後の検出結果を保存するパス
LAST_DETECTION_PATH = os.path.join(TEST_IMAGES_DIR, "last_detection.json")

def get_test_images():
    """テスト画像ディレクトリ内の画像ファイルを自動検出（自動生成ファイルは除外）"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    exclude_files = {'last_photo', 'last_photo_with_boxes'}  # 自動生成ファイルは除外
    images = {}
    for filename in os.listdir(TEST_IMAGES_DIR):
        ext = os.path.splitext(filename)[1].lower()
        name = os.path.splitext(filename)[0]
        if ext in image_extensions and name not in exclude_files:
            images[name] = os.path.join(TEST_IMAGES_DIR, filename)
    return images


@app.get("/test")
def list_test_images():
    """利用可能なテスト画像一覧を取得"""
    test_images = get_test_images()
    available = {}
    for name, path in test_images.items():
        available[name] = {
            "path": path,
            "exists": os.path.exists(path)
        }
    return {
        "message": "テスト画像一覧",
        "images": available,
        "usage": "/test/{image_name} でテストを実行（拡張子付きでも可）"
    }


@app.get("/test/last_photo", response_class=HTMLResponse)
def get_last_photo_html():
    """
    最後に/congestionで使用した画像をHTMLで表示
    """
    if not os.path.exists(LAST_PHOTO_PATH):
        return HTMLResponse(
            content="<html><body><h1>No photo available</h1><p>Call /congestion first.</p></body></html>",
            status_code=404
        )
    
    # タイムスタンプをキャッシュ回避用に追加
    timestamp = int(os.path.getmtime(LAST_PHOTO_PATH) * 1000)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Last Photo - 混雑率検出</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
            h1 {{ color: #333; }}
            img {{ max-width: 100%; height: auto; border: 2px solid #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            .info {{ margin-top: 10px; color: #666; }}
            a {{ color: #0066cc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📷 Last Photo</h1>
            <img src="/test/last_photo.jpg?t={timestamp}" alt="Last captured photo">
            <div class="info">
                <p>この画像は最後に /congestion エンドポイントで使用された画像です。</p>
                <p><a href="/congestion">→ /congestion を実行して新しい画像を取得</a></p>
                <p><a href="/test">→ テスト画像一覧に戻る</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


def draw_bounding_boxes(image, detections):
    """画像にバウンディングボックスを描画する"""
    img_with_boxes = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        score = det['score']
        name = det.get('name', 'unknown')
        
        # バウンディングボックスを描画
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # personは緑、それ以外は青
        color = (0, 255, 0) if name == 'person' else (255, 0, 0)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # ラベルを描画
        label = f'{name} {score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # ラベル背景
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img_with_boxes


@app.get("/test/last_photo.jpg")
def get_last_photo_image():
    """
    最後に/congestionで使用した画像にバウンディングボックスを描画して返す
    """
    if not os.path.exists(LAST_PHOTO_PATH):
        return {"error": "No photo available. Call /congestion first."}
    
    # 元画像を読み込み
    image = cv2.imread(LAST_PHOTO_PATH)
    
    # 検出結果があればバウンディングボックスを描画
    if os.path.exists(LAST_DETECTION_PATH):
        with open(LAST_DETECTION_PATH, 'r') as f:
            detection_result = json.load(f)
        
        person_details = detection_result.get('person_details', [])
        if person_details:
            image = draw_bounding_boxes(image, person_details)
    
    # 一時ファイルに保存して返す
    temp_path = os.path.join(TEST_IMAGES_DIR, "last_photo_with_boxes.jpg")
    cv2.imwrite(temp_path, image)
    
    return FileResponse(
        temp_path,
        media_type="image/jpeg",
        filename="last_photo.jpg"
    )


@app.get("/test/{image_name}")
def test_congestion(image_name: str):
    """
    テスト画像で混雑率を検証するエンドポイント
    
    Parameters:
    - image_name: テスト画像名（拡張子付きでも可）
    
    Returns:
    - 混雑率検出結果 + 検証情報
    """
    # 拡張子を除去して正規化
    normalized_name = os.path.splitext(image_name)[0]
    
    # 動的にテスト画像を取得
    test_images = get_test_images()
    
    if normalized_name not in test_images:
        return {
            "error": f"Unknown test image: {image_name}",
            "available_images": list(test_images.keys())
        }
    
    image_path = test_images[normalized_name]
    
    if not os.path.exists(image_path):
        return {"error": f"Test image not found: {image_path}"}
    
    try:
        image = load_image_from_file(image_path)
        
        # 最後に使用した画像を保存
        cv2.imwrite(LAST_PHOTO_PATH, image)
        
        result = calculate_congestion_rate(image)
        
        # 検出結果をJSONファイルに保存
        with open(LAST_DETECTION_PATH, 'w') as f:
            json.dump(result, f)
        
        # テスト検証情報を追加
        result['test_info'] = {
            'image_name': normalized_name,
            'image_path': image_path,
            'verification': {
                'is_crowded': result['congestion_rate'] > 10,  # 10%以上で混雑
                'crowd_level': get_crowd_level(result['congestion_rate'])
            }
        }
        return result
    except Exception as e:
        return {"error": str(e)}


def get_crowd_level(rate: float) -> str:
    """混雑率から混雑レベルを判定"""
    if rate < 5:
        return "空いている"
    elif rate < 15:
        return "やや混雑"
    elif rate < 30:
        return "混雑"
    else:
        return "非常に混雑"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)