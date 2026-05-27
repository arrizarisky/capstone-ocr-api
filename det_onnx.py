"""
PP-OCRv4 Detection (DBNet++) via ONNXRuntime.

Input : gambar BGR full-size (np.ndarray)
Output: list polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] per text region

Referensi arsitektur:
  - DBNet++: Real-time Scene Text Detection with Differentiable Binarization
    https://arxiv.org/abs/2202.10304
  - PaddleOCR det postprocess:
    https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/db_postprocess.py
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List

# ── Konstanta preprocessing (sama dengan PaddleOCR det pipeline) ──────────────
DET_LIMIT_SIDE_LEN = 960          # resize sisi terpanjang ke 960px
DET_LIMIT_TYPE = "max"             # resize berdasarkan sisi terpanjang
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DB_THRESH        = 0.3             # threshold binarisasi probability map
DB_BOX_THRESH    = 0.6             # threshold confidence per box
DB_UNCLIP_RATIO  = 1.5            # expand box sedikit agar tidak kepotong
DB_MAX_CANDIDATES = 1000


def resize_image(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Resize agar sisi terpanjang <= DET_LIMIT_SIDE_LEN, lebar/tinggi kelipatan 32."""
    h, w = img.shape[:2]
    if max(h, w) > DET_LIMIT_SIDE_LEN:
        if h > w:
            ratio = DET_LIMIT_SIDE_LEN / h
        else:
            ratio = DET_LIMIT_SIDE_LEN / w
    else:
        ratio = 1.0

    new_h = max(32, int(round(h * ratio / 32) * 32))
    new_w = max(32, int(round(w * ratio / 32) * 32))
    resized = cv2.resize(img, (new_w, new_h))

    ratio_h = new_h / h
    ratio_w = new_w / w
    return resized, ratio_h, ratio_w


def preprocess_det(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """BGR → normalized float32 tensor [1, 3, H, W]."""
    resized, ratio_h, ratio_w = resize_image(img)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - MEAN) / STD
    tensor = img_norm.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, H, W]
    return tensor.astype(np.float32), ratio_h, ratio_w


def boxes_from_bitmap(pred: np.ndarray, orig_h: int, orig_w: int,
                      ratio_h: float, ratio_w: float) -> List[np.ndarray]:
    """
    Konversi probability map → list bounding box (quad polygon) di koordinat
    gambar asli.

    Menggunakan DB postprocess: threshold → contours → minAreaRect → unclip.
    """
    import cv2
    from shapely.geometry import Polygon
    import numpy as np

    bitmap = (pred > DB_THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours[:DB_MAX_CANDIDATES]:
        # Hitung bounding polygon
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if points.shape[0] < 4:
            continue

        # Score box: rata-rata nilai pred di dalam polygon
        poly = Polygon(points)
        if not poly.is_valid or poly.area < 1:
            continue

        # Unclip: perbesar polygon agar teks tidak kepotong
        distance = poly.area * DB_UNCLIP_RATIO / poly.length
        try:
            from shapely.geometry import JOIN_STYLE
            expanded = poly.buffer(distance, join_style=JOIN_STYLE.mitre)
        except Exception:
            expanded = poly.buffer(distance)

        if expanded.is_empty:
            continue

        # Ambil 4 titik dari convex hull
        hull = np.array(expanded.convex_hull.exterior.coords, dtype=np.float32)
        rect = cv2.minAreaRect(hull)
        box_pts = cv2.boxPoints(rect).astype(np.float32)

        # Score filter berdasarkan rata-rata probabilitas di dalam box
        score = box_score(pred, points)
        if score < DB_BOX_THRESH:
            continue

        # Kembalikan ke koordinat gambar asli
        box_pts[:, 0] = np.clip(box_pts[:, 0] / ratio_w, 0, orig_w - 1)
        box_pts[:, 1] = np.clip(box_pts[:, 1] / ratio_h, 0, orig_h - 1)

        boxes.append(box_pts.astype(np.int32))

    return boxes


def box_score(pred: np.ndarray, box: np.ndarray) -> float:
    """Rata-rata nilai pred di dalam bounding polygon."""
    h, w = pred.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(box).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return float(pred[mask == 1].mean()) if mask.any() else 0.0


def sort_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Urutkan boxes dari atas-kiri ke bawah-kanan:
      1. Sort by Y (baris)
      2. Dalam satu baris (Y delta < threshold), sort by X
    """
    if not boxes:
        return boxes

    # Hitung titik tengah setiap box
    centers = [(box.mean(axis=0), i) for i, box in enumerate(boxes)]
    centers.sort(key=lambda x: (x[0][1], x[0][0]))  # sort by cy, cx
    return [boxes[i] for _, i in centers]


def crop_text_region(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Crop dan warp perspective sebuah text region dari gambar.
    Input box: 4 titik polygon [[x,y], ...]
    Output: gambar rectangular siap masuk rec model
    """
    # Urutkan pojok: top-left, top-right, bottom-right, bottom-left
    rect = order_points(box.astype(np.float32))
    (tl, tr, br, bl) = rect

    width  = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    if width <= 0 or height <= 0:
        return None

    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """Urutkan 4 titik: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: x+y terkecil
    rect[2] = pts[np.argmax(s)]   # bottom-right: x+y terbesar
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: y-x terkecil
    rect[3] = pts[np.argmax(diff)]  # bottom-left: y-x terbesar
    return rect


class DetectionModel:
    """Wrapper ONNXRuntime untuk PP-OCRv4 det model."""

    def __init__(self, model_path: str = "./models/det_model.onnx"):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[DET] Model loaded: {model_path}")

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Deteksi semua text region dalam gambar.

        Args:
            img: BGR image (np.ndarray)
        Returns:
            List polygon boxes diurutkan dari atas ke bawah
        """
        orig_h, orig_w = img.shape[:2]
        tensor, ratio_h, ratio_w = preprocess_det(img)

        output = self.session.run(None, {self.input_name: tensor})
        # Output shape: [1, 1, H, W] - probability map
        pred = output[0][0, 0]

        boxes = boxes_from_bitmap(pred, orig_h, orig_w, ratio_h, ratio_w)
        boxes = sort_boxes(boxes)
        return boxes
