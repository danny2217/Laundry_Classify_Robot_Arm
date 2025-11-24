import cv2
import numpy as np
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# 카테고리 정의
CATEGORIES = [
    "white cotton t-shirt", "white cotton shirt", "cotton t-shirt", "cotton shirt", "cotton fabric",
    "white linen shirt", "linen shirt", "linen pants", "blue denim jacket", "denim jeans", "polyester shirt",
    "nylon jacket", "tencel dress", "rayon blouse",
    "acrylic sweater", "fleece jacket", "wool sweater", "cashmere sweater",
    "angora sweater", "alpaca coat", "mohair sweater", "leather jacket",
    "suede coat", "tweed jacket", "silk blouse", "fur coat",
    "other clothes"
]

# 세탁법 분류
WASHABLE = [
    "cotton", "cotton t-shirt", "cotton shirt", "white cotton t-shirt", "white cotton shirt",
    "polyester shirt", "nylon jacket", "linen shirt", "linen pants",
    "tencel dress", "rayon blouse", "acrylic sweater"
]
DRY_CLEAN = [
    "denim jeans", "blue denim jacket", "fleece jacket", "wool sweater", "cashmere sweater",
    "angora sweater", "alpaca coat", "mohair sweater", "leather jacket",
    "suede coat", "tweed jacket", "silk blouse", "fur coat"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def is_white_hybrid(img_bgr, morph_kernel_size=15, mask_thresh=0.18, v_thresh=158):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    h, w = img_bgr.shape[:2]
    mask_ratio = cv2.countNonZero(mask_clean) / (h * w)
    v_mean = np.mean(hsv[:, :, 2])
    is_white = (mask_ratio > mask_thresh) and (v_mean > v_thresh)
    return is_white, mask_ratio, v_mean

def classify_clip(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inputs = clip_processor(
        text=CATEGORIES,
        images=pil_img,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = CATEGORIES[idx]
        conf = float(probs[idx])
    return label, conf

def get_laundry_type(label):
    for washable in WASHABLE:
        if washable in label:
            return "Normal_Washing"
    for dry in DRY_CLEAN:
        if dry in label:
            return "Dry_Cleaning"
    return "etc"

# ===== 라벨에서 '소재'만 추출 (오버레이용) =====
_FABRIC_PATTERNS = [
    ("cashmere", "cashmere"),
    ("angora", "angora"),
    ("alpaca", "alpaca"),
    ("mohair", "mohair"),
    ("wool", "wool"),
    ("denim", "denim"),
    ("cotton", "cotton"),
    ("linen", "linen"),
    ("polyester", "polyester"),
    ("nylon", "nylon"),
    ("tencel", "tencel"),
    ("rayon", "rayon"),
    ("acrylic", "acrylic"),
    ("fleece", "fleece"),
    ("leather", "leather"),
    ("suede", "suede"),
    ("tweed", "tweed"),
    ("silk", "silk"),
    ("fur", "fur"),
    ("other clothes", "other"),
    ("other", "other"),
]

def extract_fabric(label: str) -> str:
    low = label.lower()
    for kw, fabric in _FABRIC_PATTERNS:
        # 단어 경계 매칭 (예: 'line' ≠ 'linen')
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return fabric
    return "other"

def main():
    cam_index = find_working_camera()
    if cam_index is None:
        print("사용 가능한 카메라를 찾지 못했습니다.")
        return

    cap = cv2.VideoCapture(cam_index)
    print(f"🎥 카메라로 분류 시작 (인덱스 {cam_index}) — q 누르면 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임 읽기 실패")
                break

            # CLIP 분류
            label, conf = classify_clip(frame)

            # 화이트/컬러 판단
            is_white, mask_ratio, v_mean = is_white_hybrid(frame)
            color_str = "White" if is_white else "Color"

            # 세탁법
            laundry = get_laundry_type(label)  # Normal_Washing / Dry_Cleaning / etc

            # 화면에는 "소재 | 색상 | 세탁법"
            fabric = extract_fabric(label).title()  # 예: Cotton, Wool, Denim
            display = f"{fabric} | {color_str} | {laundry}"

            cv2.putText(frame, display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Laundry Camera Classifier", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()