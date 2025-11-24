# util.py
import cv2
import numpy as np

def get_limits(bgr_color, h_tol=10, s_tol=100, v_tol=100):
    """
    BGR 색상 하나를 받아 HSV 하한/상한을 반환합니다.

    Args:
        bgr_color (list or tuple of int): [B, G, R] 각 0~255 값
        h_tol (int): Hue 허용 편차 (0~179)
        s_tol (int): Saturation 허용 편차 (0~255)
        v_tol (int): Value(명도) 허용 편차 (0~255)

    Returns:
        lower (ndarray): HSV 하한값 [H, S, V]
        upper (ndarray): HSV 상한값 [H, S, V]
    """
    # 1×1 이미지로 만들어 BGR→HSV 변환
    bgr = np.uint8([[bgr_color]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 허용 범위 계산
    lower = np.array([max(h - h_tol, 0),
                      max(s - s_tol, 0),
                      max(v - v_tol, 0)], dtype=np.uint8)
    upper = np.array([min(h + h_tol, 179),
                      min(s + s_tol, 255),
                      min(v + v_tol, 255)], dtype=np.uint8)

    return lower, upper