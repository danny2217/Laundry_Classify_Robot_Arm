import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import serial
import time

# ─────────────────────────────────────────────────────────────
# 1) 분류 카테고리
# ─────────────────────────────────────────────────────────────
CATEGORIES = [
    "white cotton t-shirt", "white cotton shirt", "cotton t-shirt", "cotton shirt", "cotton fabric",
    "white linen shirt", "linen shirt", "linen pants", "blue denim jacket", "denim jeans", "polyester shirt",
    "nylon jacket", "tencel dress", "rayon blouse",
    "acrylic sweater", "fleece jacket", "wool sweater", "cashmere sweater",
    "angora sweater", "alpaca coat", "mohair sweater", "leather jacket",
    "suede coat", "tweed jacket", "silk blouse", "fur coat",
    "other clothes"
]

# ─────────────────────────────────────────────────────────────
# 2) 세탁 분류 규칙
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 3) 모델 준비
# ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ─────────────────────────────────────────────────────────────
# 4) 시리얼(아두이노) 초기화
# ─────────────────────────────────────────────────────────────
def init_serial(port='/dev/tty.usbserial-140', baud=9600, timeout=2):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(timeout)  # 아두이노 리셋 대기
        print(f"[Serial] Connected: {port} @ {baud}")
        return ser
    except serial.SerialException as e:
        print(f"[Serial] 연결 실패: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 5) 카메라 열기 (macOS: AVFoundation 우선)
# ─────────────────────────────────────────────────────────────
def init_camera(cam_index=0, width=1280, height=720, fps=30):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS,          fps)

    print(f"[Camera] Opened at index {cam_index}")
    return cap

# ─────────────────────────────────────────────────────────────
# 6) 흰색 감지(하이브리드)
# ─────────────────────────────────────────────────────────────
def is_white_hybrid(img_bgr, morph_kernel_size=15, mask_thresh=0.18, v_thresh=158):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean  = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,  kernel)

    h, w = img_bgr.shape[:2]
    mask_ratio = cv2.countNonZero(mask_clean) / (h * w)
    v_mean = float(np.mean(hsv[:, :, 2]))
    is_white = (mask_ratio > mask_thresh) and (v_mean > v_thresh)
    return is_white, mask_ratio, v_mean

# ─────────────────────────────────────────────────────────────
# 7) CLIP 분류
# ─────────────────────────────────────────────────────────────
def classify_clip(img_bgr, min_conf=0.30):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inputs = clip_processor(text=CATEGORIES, images=pil_img,
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
        probs = out.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    label = CATEGORIES[idx]
    conf  = float(probs[idx])
    if conf < min_conf:
        return "other clothes", conf
    return label, conf

# ─────────────────────────────────────────────────────────────
# 8) 세탁 분류 문자열
# ─────────────────────────────────────────────────────────────
def get_laundry_type(label):
    for washable in WASHABLE:
        if washable in label:
            return "Normal_Washing"
    for dry in DRY_CLEAN:
        if dry in label:
            return "Dry_Cleaning"
    return "etc"

# ─────────────────────────────────────────────────────────────
# 9) 전송 함수: 모든 RX 라인 CMD에 출력 + 'done'까지 대기
# ─────────────────────────────────────────────────────────────
def send_serial(ser, message, last_sent, wait_timeout=2.0):
    if ser is None:
        return last_sent
    if message == last_sent:
        return last_sent

    try:
        ser.write((message + '\n').encode())
        print(f"[TX] {message}")

        t0 = time.time()
        while time.time() - t0 < wait_timeout:
            while ser.in_waiting:
                resp = ser.readline().decode(errors='ignore').strip()
                if resp:
                    print(f"[RX] {resp}")  # CMD 터미널에 모든 라인 표시
                if resp.lower() == "done":
                    return message
            time.sleep(0.01)
        print("[Serial] ACK timeout")
    except Exception as e:
        print(f"[Serial] 전송 오류: {e}")
    return last_sent

# ─────────────────────────────────────────────────────────────
# 10) 메인 루프
# ─────────────────────────────────────────────────────────────
def main():
    cap = init_camera(cam_index=0, width=1280, height=720, fps=30)
    ser = init_serial(port='/dev/tty.usbserial-140', baud=9600, timeout=2)

    last_sent = None
    print("▶ 분류 시작 — 미리보기 창에 포커스 두고 'q'를 누르면 종료됩니다.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임 읽기 실패")
                break

            # (a) 분류
            label, conf = classify_clip(frame)

            # (b) 흰색 여부
            is_white, _, _ = is_white_hybrid(frame)

            # (c) 라벨 보정(white prefix)
            send_label = label
            if is_white and "white" not in label:
                send_label = "white_" + label
            elif not is_white and label.startswith("white "):
                send_label = label.replace("white ", "")

            # (d) 세탁 타입
            laundry = get_laundry_type(label)

            # (e) 전송: "라벨|세탁타입"
            full_msg = f"{send_label}|{laundry}"
            last_sent = send_serial(ser, full_msg, last_sent, wait_timeout=2.0)

            # (f) 화면 디스플레이
            info = f"{label} ({conf:.2f}) | {'white' if is_white else 'non-white'} | {laundry}"
            cv2.putText(frame, info, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Laundry Camera Classifier", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if ser:
            ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()