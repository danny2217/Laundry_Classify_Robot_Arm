import cv2
import numpy as np
import re
import threading
from collections import deque, Counter
from PIL import Image, ImageTk
from transformers import CLIPProcessor, CLIPModel
import torch
import serial
import time
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import json  # ★persist lock

# =====================[ 사용자 설정 ]=====================
CAM_INDEX = 0
FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 30
MAJORITY_N = 7
WARMUP_FRAMES = 8

# ★즉시 잠김 플래그
INSTANT_LOCK_ON_READY = True

# ★시리얼 신호 디바운스/쿨다운
READY_DEBOUNCE_SEC = 0.10    # ready 신호 노이즈 필터
CAPTURE_COOLDOWN_SEC = 0.80  # 잠김 직후 다음 잠김까지 대기

# 아두이노 포트 (mac: /dev/cu.usbserial*, /dev/cu.usbmodem*)
SERIAL_PORT = '/dev/tty.usbserial-140'
SERIAL_BAUD = 9600

# (이하 캡처 윈도우 상수는 즉시잠김 모드에선 사용되지 않지만, 호환을 위해 유지)
CAPTURE_SEC = 3.5
WHITE_VOTE_THRESH = 0.60

# ★LOCK 상태 영구 저장 파일
LOCK_PERSIST_PATH = "last_lock.json"

# =====================[ 분류 카테고리/세탁법 ]=====================
CATEGORIES = [
    "white cotton t-shirt", "white cotton shirt", "cotton t-shirt", "cotton shirt", "cotton fabric",
    "white linen shirt", "linen shirt", "linen pants", "blue denim jacket", "denim jeans", "polyester shirt",
    "nylon jacket", "tencel dress", "rayon blouse",
    "acrylic sweater", "fleece jacket", "wool sweater", "cashmere sweater",
    "angora sweater", "alpaca coat", "mohair sweater", "leather jacket",
    "suede coat", "tweed jacket", "silk blouse", "fur coat",
    "other clothes", "wool blanket", "wool", "silk"
]
WASHABLE = [
    "cotton", "cotton t-shirt", "cotton shirt", "white cotton t-shirt", "white cotton shirt",
    "polyester shirt", "nylon jacket", "linen shirt", "linen pants",
    "tencel dress", "rayon blouse", "acrylic sweater", "other clothes"
]
DRY_CLEAN = [
    "denim jeans", "blue denim jacket", "fleece jacket", "wool sweater", "cashmere sweater",
    "angora sweater", "alpaca coat", "mohair sweater", "leather jacket",
    "suede coat", "tweed jacket", "silk blouse", "silk", "fur coat",
    "wool blanket", "wool"
]

# =====================[ 모델 로드 ]=====================
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# =====================[ 시리얼 ]=====================
def init_serial(port=None, baud=9600, timeout=2):
    if not port:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(timeout)  # 보드 리셋 대기
        print(f"[Serial] Connected: {port} @ {baud}")
        while ser.in_waiting:
            _ = ser.readline()
        return ser
    except Exception as e:
        print(f"[Serial] 연결 실패: {e}")
        return None

# =====================[ 카메라 & 전처리 ]=====================
def init_camera(cam_index=0, width=1280, height=720, fps=30):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS,          fps)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
        cap.set(cv2.CAP_PROP_GAIN, 0)
    except Exception:
        pass
    print(f"[Camera] Opened at index {cam_index}")
    return cap

def _gamma(img, gamma):
    g = max(0.2, min(3.0, float(gamma)))
    inv = 1.0 / g
    table = (np.linspace(0, 1, 256) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)

def preprocess_for_stability(img_bgr):
    # 그레이월드 + CLAHE + 약한 감마(암부 보정)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    b *= (mean_gray / mean_b); g *= (mean_gray / mean_g); r *= (mean_gray / mean_r)
    wb = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)

    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    stab = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    vmean = cv2.cvtColor(stab, cv2.COLOR_BGR2HSV)[:,:,2].mean()
    if vmean < 110:  # 어두우면 살짝 밝게
        stab = _gamma(stab, 0.7)
    return stab

# =====================[ 화이트 판정 ]=====================
def colorfulness_metric(img_bgr):
    B, G, R = cv2.split(img_bgr.astype(np.float32))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2)

WHITE_CFG = {
    "roi_ratio": 0.70,
    "cf_max": 35.0,
    "v_med_min": 55.0,
    "v_thr_min": 135.0,
    "v_thr_pct": 75,
    "s_thr_base": 36.0,
    "lab_chroma_thr": 14.0,
    "L_thr": 170.0,
    "ratio_min": 0.18,
    "morph": 7,
    "hi_pct": 0.20
}

def _central_roi(img, ratio):
    h, w = img.shape[:2]
    rh, rw = int(h*ratio), int(w*ratio)
    y0 = (h - rh) // 2; x0 = (w - rw) // 2
    return img[y0:y0+rh, x0:x0+rw]

def is_white_tolerant(orig_bgr, cfg=WHITE_CFG):
    roi = _central_roi(orig_bgr, cfg["roi_ratio"])

    # 역광/저조도 보정
    hsv0 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_med = np.median(hsv0[:,:,2])
    boosted = roi.copy()
    if v_med < cfg["v_med_min"]:
        boosted = _gamma(boosted, 0.6)
        lab_b = cv2.cvtColor(boosted, cv2.COLOR_BGR2LAB)
        Lb, Ab, Bb = cv2.split(lab_b)
        Lb = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(Lb)
        boosted = cv2.cvtColor(cv2.merge([Lb, Ab, Bb]), cv2.COLOR_LAB2BGR)

    cf = colorfulness_metric(boosted)
    if cf > cfg["cf_max"]:
        Vmean = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)[:,:,2].mean()
        return False, 0.0, float(Vmean), float(cf)

    hsv = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1].astype(np.float32)
    V = hsv[:,:,2].astype(np.float32)
    lab = cv2.cvtColor(boosted, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0].astype(np.float32)
    a = lab[:,:,1].astype(np.float32) - 128.0
    b = lab[:,:,2].astype(np.float32) - 128.0
    chroma = np.sqrt(a*a + b*b)

    v_thr = max(cfg["v_thr_min"], np.percentile(V, cfg["v_thr_pct"]) - 8.0)
    s_thr = cfg["s_thr_base"]

    mask_main = (V >= v_thr) & (S <= s_thr) & (chroma <= cfg["lab_chroma_thr"]) & (L >= cfg["L_thr"])

    hi_cut = np.percentile(V, 100*(1.0 - cfg["hi_pct"]))
    hi = V >= hi_cut
    mask_hi = hi & (S <= (s_thr+4)) & (chroma <= (cfg["lab_chroma_thr"]+2))

    mask = (mask_main | mask_hi).astype(np.uint8)*255

    k = int(cfg["morph"])
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    ratio = (mask > 0).mean()
    is_white = ratio >= cfg["ratio_min"]
    return bool(is_white), float(ratio), float(V.mean()), float(cf)

# =====================[ CLIP 분류/라벨 파싱 ]=====================
_FABRIC_PATTERNS = [
    ("wool blanket", "wool"), ("cashmere", "cashmere"), ("angora", "angora"),
    ("alpaca", "alpaca"), ("mohair", "mohair"), ("wool", "wool"),
    ("denim", "denim"), ("cotton", "cotton"), ("linen", "linen"),
    ("polyester", "polyester"), ("nylon", "nylon"), ("tencel", "tencel"),
    ("rayon", "rayon"), ("acrylic", "acrylic"), ("fleece", "fleece"),
    ("leather", "leather"), ("suede", "suede"), ("tweed", "tweed"),
    ("silk", "silk"), ("fur", "fur"), ("other clothes", "other"), ("other", "other"),
]
_FABRIC_KEYS = [k for (k,_) in _FABRIC_PATTERNS if k not in ("other clothes","other")]

def extract_fabric(label: str) -> str:
    low = label.lower()
    for kw, canon in _FABRIC_PATTERNS:
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return canon
    return "other"

def classify_clip(img_bgr, min_conf=0.30):
    # 동적 임계(어두울수록 완화)
    v_mean_scene = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:,:,2].mean()
    dyn_min_conf = min_conf * (0.85 if v_mean_scene < 110 else 1.0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inputs = clip_processor(text=CATEGORIES, images=pil_img,
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
        probs = out.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    order = np.argsort(-probs)
    idx1, idx2, idx3 = order[:3]
    label1, p1 = CATEGORIES[idx1], float(probs[idx1])

    if label1 == "other clothes" and p1 < 0.60:
        for k in [idx2, idx3]:
            lbl = CATEGORIES[k]
            if any(key in lbl for key in _FABRIC_KEYS) and float(probs[k]) > 0.18:
                return lbl, float(probs[k])
        return CATEGORIES[idx2], float(probs[idx2])

    if p1 < dyn_min_conf:
        for k in [idx2, idx3]:
            lbl = CATEGORIES[k]
            if any(key in lbl for key in _FABRIC_KEYS) and (probs[k] > p1*0.85 or probs[k] > 0.22):
                return lbl, float(probs[k])

    return label1, p1

def get_laundry_type(label):
    for washable in WASHABLE:
        if washable in label:
            return "Normal_Washing"
    for dry in DRY_CLEAN:
        if dry in label:
            return "Dry_Cleaning"
    return "etc"

# =====================[ 워커 스레드 ]=====================
class VideoWorker(threading.Thread):
    def __init__(self, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.cap = None
        self.ser = init_serial(SERIAL_PORT, SERIAL_BAUD) if SERIAL_PORT else None

        # 실시간 히스토리
        self.white_hist = deque(maxlen=MAJORITY_N)
        self.label_hist = deque(maxlen=MAJORITY_N)

        # 상태기계
        self.mode = "live"            # 'live' | 'locked'
        self.serial_ready = False     # 'sort:ready' 수신 여부

        # 누적(과거 호환용 변수)
        self.cap_labels = []
        self.cap_confs  = []
        self.cap_white  = []
        self.cap_ratios = []
        self.cap_vmeans = []
        self.cap_cfs    = []

        # 락 결과 저장
        self.lock_data = None

        # ★ready 엣지/쿨다운 상태
        self._last_ready_ts = 0.0
        self._prev_serial_ready = False
        self._last_capture_end_ts = 0.0

        # ★즉시잠김 플래그
        self.instant_lock_pending = False

    # ----- 시리얼 수신(ready 감지) -----
    def _serial_poll(self):
        if not self.ser:
            return
        while self.ser.in_waiting:
            resp = self.ser.readline().decode(errors='ignore').strip()
            if not resp:
                continue
            print("[RX]", resp)
            low = resp.lower()
            if "sort:ready" in low:
                self.serial_ready = True

    # ----- 즉시 잠금 확정 -----
    def _lock_from_instant(self, label, conf, is_white, ratio, vmean, cf):
        color_str = "White" if is_white else "Color"
        laundry = get_laundry_type(label)
        fabric = extract_fabric(label).title()

        self.lock_data = dict(
            fabric=fabric,
            confidence=float(conf),
            laundry=laundry,
            color=color_str,
            white_ratio=float(ratio),
            vmean=float(vmean),
            cf=float(cf),
            votes="1 / 1"
        )

        self.mode = "locked"
        self.serial_ready = False
        self._last_capture_end_ts = time.time()

        print(f"[LOCK:instant] {fabric} | {laundry} | {color_str} (conf:{conf:.2f})")

        # 디스크에 저장
        try:
            with open(LOCK_PERSIST_PATH, "w", encoding="utf-8") as f:
                json.dump(self.lock_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[LOCK] persist save error:", e)

        # 아두이노로 전송
        if self.ser:
            msg = f"{laundry}|{color_str}"
            try:
                self.ser.write((msg + '\n').encode())
                print("[TX]", msg)
            except Exception as e:
                print("[Serial] write error:", e)

    def run(self):
        try:
            self.cap = init_camera(CAM_INDEX, FRAME_W, FRAME_H, FRAME_FPS)
            warmup = WARMUP_FRAMES
            t_prev = time.time()

            while not self.stop_event.is_set():
                self._serial_poll()

                # ★'ready' 상승엣지 + 디바운스 + 쿨다운
                now = time.time()
                rising_edge = (self.serial_ready and not self._prev_serial_ready)
                debounced = (now - self._last_ready_ts) >= READY_DEBOUNCE_SEC
                cooled = (now - self._last_capture_end_ts) >= CAPTURE_COOLDOWN_SEC

                if rising_edge and debounced and cooled:
                    self._last_ready_ts = now
                    if INSTANT_LOCK_ON_READY:
                        self.instant_lock_pending = True

                self._prev_serial_ready = self.serial_ready

                ok, frame = self.cap.read()
                if not ok:
                    print("프레임 읽기 실패"); break

                stab = preprocess_for_stability(frame)
                label, conf = classify_clip(stab, min_conf=0.30)
                is_white, ratio, vmean, cf = is_white_tolerant(frame, cfg=WHITE_CFG)

                # 실시간 히스토리(라이브 표시용)
                self.white_hist.append(1 if is_white else 0)
                self.label_hist.append(label)

                now2 = time.time()
                fps = 1.0 / max(1e-6, (now2 - t_prev))
                t_prev = now2

                # ===== 즉시 잠김 처리 =====
                if self.instant_lock_pending:
                    # 웜업 중이어도 즉시 잠김을 원하면 바로 확정
                    self._lock_from_instant(label, conf, bool(is_white), ratio, vmean, cf)
                    self.instant_lock_pending = False

                # ===== 상태별 표시 =====
                if self.mode == "locked" and self.lock_data:
                    # 확정값 유지
                    cur_fabric = self.lock_data['fabric']
                    cur_laundry= self.lock_data['laundry']
                    cur_color  = self.lock_data['color']
                    cur_ratio  = self.lock_data['white_ratio']
                    cur_vmean  = self.lock_data['vmean']
                    cur_cf     = self.lock_data['cf']
                    conf       = self.lock_data['confidence']
                    mode_text  = "locked"
                    warm = 0
                else:
                    # 평시 라이브
                    need = int(np.ceil(len(self.white_hist) * 0.7))
                    smoothed_white = (sum(self.white_hist) >= need)
                    cur_color = "White" if smoothed_white else "Color"
                    most_label = Counter(self.label_hist).most_common(1)[0][0]
                    cur_laundry = get_laundry_type(most_label)
                    cur_fabric = extract_fabric(most_label).title()
                    cur_ratio = ratio; cur_vmean=vmean; cur_cf=cf
                    mode_text = "live"
                    warm = max(0, warmup)
                    if warmup > 0:
                        warmup -= 1

                # UI 공유상태 업데이트
                with self.lock:
                    self.shared.update({
                        "fabric": cur_fabric,
                        "confidence": float(conf),
                        "laundry": cur_laundry,
                        "color": cur_color,
                        "white_ratio": float(cur_ratio),
                        "vmean": float(cur_vmean),
                        "cf": float(cur_cf),
                        "votes": f"{sum(self.white_hist)}/{len(self.white_hist)}",
                        "fps": float(fps),
                        "warmup": warm,
                        "frame": stab.copy(),
                        "mode": mode_text
                    })
        finally:
            if self.cap: self.cap.release()
            if self.ser: self.ser.close()

# =====================[ Tkinter UI (두 창) ]=====================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Laundry Classifier – Dashboard")
        self.root.minsize(900, 540)
        self.root.configure(bg="#1f2124")

        self.shared = {
            "fabric": "—", "confidence": 0.0, "laundry": "—", "color": "—",
            "white_ratio": 0.0, "vmean": 0.0, "cf": 0.0, "votes": "0/0",
            "fps": 0.0, "warmup": WARMUP_FRAMES, "frame": None, "mode": "live"
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.base_width = 1280
        self.base_height = 720

        self.font_big = tkfont.Font(family="Helvetica", size=28, weight="bold")
        self.font_mid = tkfont.Font(family="Helvetica", size=20)
        self.font_sm  = tkfont.Font(family="Helvetica", size=14)

        self.container = tk.Frame(self.root, bg="#1f2124")
        self.container.place(relx=0.5, rely=0.5, anchor="center", width=int(self.root.winfo_width()*0.96))

        self._build_dashboard(self.container)

        self.preview = tk.Toplevel(self.root)
        self.preview.title("Camera Preview")
        self.preview.geometry("960x540")
        self.preview.configure(bg="#000")
        self.preview_img_lbl = tk.Label(self.preview, bg="#000")
        self.preview_img_lbl.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", self._on_preview_resize)
        self.prev_w, self.prev_h = 960, 540
        self.preview_imgtk = None

        self.root.bind("<Configure>", self._on_root_resize)

        # ★마지막 LOCK 상태 복구 시도 (UI 초기값을 곧바로 잠김으로)
        persisted = None
        try:
            with open(LOCK_PERSIST_PATH, "r", encoding="utf-8") as f:
                persisted = json.load(f)
        except Exception:
            persisted = None

        if persisted:
            self.shared.update({
                "fabric": persisted.get("fabric", "—"),
                "confidence": float(persisted.get("confidence", 0.0)),
                "laundry": persisted.get("laundry", "—"),
                "color": persisted.get("color", "—"),
                "white_ratio": float(persisted.get("white_ratio", 0.0)),
                "vmean": float(persisted.get("vmean", 0.0)),
                "cf": float(persisted.get("cf", 0.0)),
                "votes": persisted.get("votes", "0/0"),
                "fps": 0.0,
                "warmup": 0,
                "frame": None,
                "mode": "locked"
            })

        self.worker = VideoWorker(self.shared, self.lock, self.stop_event)

        # ★워커 내부 상태도 잠김으로 동기화(재시작 시 계속 인식되는 느낌 유지)
        if persisted:
            self.worker.lock_data = persisted
            self.worker.mode = "locked"
            self.worker.serial_ready = False

        self.worker.start()

        self._tick_ui()
        self._tick_preview()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.preview.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_dashboard(self, parent):
        self.title_lbl = tk.Label(parent, text="현재 분류 결과", font=self.font_big,
                                  fg="#e8eaed", bg="#1f2124", justify="center")
        self.title_lbl.pack(pady=(10, 16))

        self.info_frame = tk.Frame(parent, bg="#1f2124")
        self.info_frame.pack(pady=4, fill="x", expand=False)

        self.info_frame.grid_columnconfigure(0, weight=1, uniform="cols")
        self.info_frame.grid_columnconfigure(1, weight=1, uniform="cols")

        def mk_row(r, name):
            name_lbl = tk.Label(self.info_frame, text=name, font=self.font_mid,
                                fg="#e8eaed", bg="#1f2124", anchor="e")
            name_lbl.grid(row=r, column=0, sticky="e", padx=(0, 18), pady=8)
            val_lbl  = tk.Label(self.info_frame, text="—", font=self.font_mid,
                                fg="#e8eaed", bg="#1f2124", anchor="w")
            val_lbl.grid(row=r, column=1, sticky="w", padx=(18, 0), pady=8)
            return val_lbl

        self.val_fabric  = mk_row(0, "Fabric")
        self.val_laundry = mk_row(1, "Laundry")
        self.val_color   = mk_row(2, "Color")

        self.pb_frame = tk.Frame(parent, bg="#1f2124")
        self.pb_frame.pack(pady=(18, 6))
        self.pb_label = tk.Label(self.pb_frame, text="White ratio", font=self.font_sm,
                                 fg="#e8eaed", bg="#1f2124")
        self.pb_label.pack(side="left", padx=(0, 12))
        self.pr_white = ttk.Progressbar(self.pb_frame, orient="horizontal",
                                        mode="determinate", length=600, maximum=100)
        self.pr_white.pack(side="left", fill="x", expand=False)

        self.lbl_dbg = tk.Label(parent, text="", font=self.font_sm, fg="#22c55e", bg="#1f2124")
        self.lbl_dbg.pack(pady=(10, 14))

    def _on_root_resize(self, event):
        w = max(900, self.root.winfo_width())
        h = max(540, self.root.winfo_height())
        target_w = int(w * 0.96)
        self.container.update_idletasks()
        req = self.container.winfo_reqwidth()
        cont_w = max(target_w, req + 24)
        self.container.place(relx=0.5, rely=0.5, anchor="center", width=cont_w)
        scale_w = cont_w / float(self.base_width)
        scale_h = h / float(self.base_height)
        scale = max(0.7, min(1.8, min(scale_w, scale_h)))
        self.font_big.configure(size=int(28 * scale))
        self.font_mid.configure(size=int(20 * scale))
        self.font_sm.configure(size=int(14 * scale))
        self.pr_white.config(length=int(cont_w * 0.55))

    def _on_preview_resize(self, event):
        self.prev_w = max(320, self.preview.winfo_width())
        self.prev_h = max(180, self.preview.winfo_height())

    def _tick_ui(self):
        with self.lock:
            st = self.shared.copy()

        # 상태 표기
        title = "현재 분류 결과"
        if st['mode'] == "locked":
            title += "  (Locked)"
        self.title_lbl.config(text=title)

        self.val_fabric.config(text=st['fabric'])
        self.val_laundry.config(text=st['laundry'])
        self.val_color.config(text=st['color'])
        self.pr_white['value'] = int(st['white_ratio'] * 100.0)

        if st['warmup'] > 0 and st['mode'] != "locked":
            dbg = f"Warming up... ({st['warmup']}) | fps:{st['fps']:.1f}"
        else:
            dbg = f"cf:{st['cf']:.1f}  Vmean:{st['vmean']:.0f}  votes:{st['votes']}  conf:{st['confidence']:.2f}  fps:{st['fps']:.1f}"
        self.lbl_dbg.config(text=dbg)

        if not self.stop_event.is_set():
            self.root.after(100, self._tick_ui)
        else:
            self._really_close()

    def _tick_preview(self):
        frame = None
        with self.lock:
            if self.shared["frame"] is not None:
                frame = self.shared["frame"].copy()

        if frame is not None:
            disp = cv2.resize(frame, (self.prev_w, self.prev_h), interpolation=cv2.INTER_AREA)
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(disp)
            self.preview_imgtk = ImageTk.PhotoImage(image=im)
            self.preview_img_lbl.config(image=self.preview_imgtk)

        if not self.stop_event.is_set():
            self.root.after(16, self._tick_preview)
        else:
            self._really_close()

    def _on_close(self):
        self.stop_event.set()

    def _really_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

# =====================[ 엔트리 포인트 ]=====================
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.state('zoomed')
    except Exception:
        root.geometry("1400x900")
    app = App(root)
    root.mainloop()