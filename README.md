# Laundry_Classify_Robot_Arm
#  AI 기반 자동 세탁물 분류 시스템  
Camera → AI(CLIP Zero-Shot) → Laundry Rule Engine → (옵션: Arduino Robot Arm)

본 프로젝트는 **카메라로 옷을 비추면 소재·색상·세탁법을 자동으로 분류하는 AI 시스템**입니다.  
전통적인 학습 기반 모델 대신 OpenAI의 **CLIP Zero-shot Model**을 사용하여  
데이터셋 없이도 즉시 분류가 가능하도록 설계된 것이 핵심 특징입니다.

---

#  프로젝트 특징

###  Zero-shot AI(CLIP) 기반 분류  
- 모델을 따로 학습(train)할 필요 없음  
- 텍스트 라벨(cotton shirt, denim jeans…)을 기반으로 의미적 유사도로 분류  
- 조명·각도 변화에도 강한 성능

###  Hybrid White Detection  
흰색 의류는 패턴·주름 때문에 일반 HSV 필터로 인식이 어렵기 때문에  
HSV + Morphology(MOPEN/Mclose) + V(명도) 기반 **하이브리드 알고리즘** 적용.

###  인식 정보  
카메라 또는 이미지 입력 시 아래 정보를 제공:

- 소재(Fabric Material)
- 흰색 여부(White / Non-white)
- 세탁 분류(Normal Washing / Dry Cleaning)
- Confidence(신뢰도)
- Mask Ratio / V_mean

###  실시간 카메라 분류(camera_classify.py)
웹캠 프레임을 실시간으로 처리하며 결과를 화면에 출력합니다.

###  이미지 파일 기반 테스트(classify_images.py)
test_images 폴더 안의 jpg/png 이미지를 일괄 분석합니다.

###  로봇팔 연동(laundry_classify.py)
AI 결과를 시리얼로 Arduino에 전달해  
SCServo 기반 로봇팔이 자동으로 배치하도록 설계.

---

#  프로젝트 구조

/project-root
│
├── camera_classify.py # 실시간 카메라 기반 AI 분류
├── classify_images.py # 이미지(폴더) 기반 AI 분류
├── laundry_classify.py # AI + Arduino Serial 연동 (Robot Arm)
│
├── test.py # 기능 테스트용
├── util.py # HSV 필터링 및 기타 유틸
├── utils.py # 공용 유틸 코드
│
├── train_resnet.py # (실험) ResNet 학습 코드<환경적 한계로 인해 사용 X>
├── train_resnet_fast.py
├── train_vits.py
├── train_vits_fast.py
│
├── test_images/ # 이미지 기반 테스트용 폴더
│ └── *.jpg / *.png
│
└── README.md

yaml
코드 복사

---

#  설치 방법

### 1) 기본 패키지 설치
```bash
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install transformers
pip install pyserial
2) CLIP Zero-shot 모델 다운로드 (자동)
코드 실행 시 자동으로 다운로드됩니다:

openai/clip-vit-base-patch32
(HuggingFace Transformers 기반)

▶ 실행 방법
 1. 실시간 카메라 분류
bash
코드 복사
python camera_classify.py
 2. 이미지 폴더 분류
bash
코드 복사
python classify_images.py
 3. 로봇팔 연동
아두이노 연결 후:

bash
코드 복사
python laundry_classify.py
아두이노 포트(macOS):
/dev/tty.usbmodemXXXXX 또는 /dev/tty.usbserial-XXXXX
코드 상단 init_serial()의 port 변경 필요.

 모델 설명 — CLIP Zero-Shot
이 프로젝트는 학습 데이터 없이 분류 모델을 만드는 것을 목표로 했기 때문에
OpenAI의 CLIP 모델을 Zero-shot 방식으로 사용했습니다.

CLIP 특징:

텍스트-이미지 쌍을 학습한 거대 비전/언어 모델

이미지와 텍스트 문장의 의미적 유사도 계산 가능

학습 과정 없이 실시간 분류 가능

라벨을 텍스트로만 추가하면 바로 새로운 카테고리 확장 가능

본 프로젝트는 학습된 CLIP 모델 위에서
단순히 텍스트 라벨 리스트를 입력하여 즉시 분류합니다.

 White Detection Algorithm
(Hybrid HSV + Morphology)

흰색 옷 인식은 카메라·조명·패턴 때문에 어렵기 때문에
다음 3단계를 결합했습니다:

HSV 영역에서 흰색 범위 마스크 추출

Morphology Close/Open 연산으로 노이즈 제거

비율(mask_ratio) + 명도(V_mean) 기준으로 최종 판정

 Arduino / Robot Arm 연동
Python → Serial → Arduino → SCServo

전송 포맷:

php-template
코드 복사
<material>|<laundry_type>
예: white_cotton shirt|Normal_Washing
로봇팔 동작:

초음파 센서 감지

잡기(pick_up)

분류 지점 이동(move_to_sort)

소재별로 배치(move_to_cotton / _polyester / _another)

 성능
✔ Cotton / Linen / Denim / Polyester 등 주요 소재 robust
✔ White Detection 성공률 대폭 상승
✔ 카메라 실시간 환경에서도 안정적 분류
✔ Zero-shot 기반 → 새로운 카테고리 확장이 매우 쉬움
 License
본 프로젝트는 MIT License 기반으로 배포됩니다.

사용된 모델 라이선스
CLIP 모델:

OpenAI에서 공개한 MIT License 기반 모델

HuggingFace Transformers를 통해 로드함

코드 및 모델 사용에 저작권 문제 없음
(단, OpenAI/CLIP의 공식 문서 및 논문 출처 표시는 권장)

[참고 출처]
https://github.com/openai/CLIP

https://huggingface.co/openai/clip-vit-base-patch32

Author
장성환 (2025)
AI + Computer Vision + Robotics Integration

문의 및 협업: danny2217@naver.com / github id: danny2217
