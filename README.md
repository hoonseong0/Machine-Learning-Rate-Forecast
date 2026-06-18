# Machine-Learning-Rate-Forecast

https://machine-learning-rate-forecast-enkgpybjjqkngtcjmqhzj3.streamlit.app/

# 📈 AI 원/달러 환율 예측 스마트 가이드 (AI Exchange Rate Prediction Platform)

> **글로벌 거시경제 지표와 다차원 가격 분석 알고리즘을 결합하여 내일의 원/달러 환율 방향성을 객관적 확률로 제시하는 데이터 기반 금융 분석 대시보드 플랫폼입니다.**

본 플랫폼은 외화 자산 운용 효율성을 제고하고 환차손 리스크를 최소화하기 위해 설계되었습니다. 단순한 기술적 지표 분석을 넘어, 다각적 거시경제 데이터의 상관관계를 연산하는 복합 AI 알고리즘을 탑재하여 스마트한 환전 가이드를 제공합니다.

---

## ✨ 핵심 제공 기능 (Key Features)

### 1. 🔄 실시간 거시경제 데이터 동기화 파이프라인
* 매일 접속 시각을 기준으로 최신 금융 시장 데이터(`USD/KRW`, `달러 인덱스(DXY)`, `미 국채 10년물 금리`, `국제 금 가격`, `KOSPI`, `S&P 500`)를 자동 수집 및 동기화합니다.
* 시장의 패러다임 변화(Concept Drift)로 인한 예측력 저하를 방지하기 위해, 상시 최신 5년 데이터셋만 유지하며 유연한 흐름을 추적합니다.

### 2. 📊 다차원 지표 분석 및 가격 가속도 산출
* 환율의 단기 및 중기 방향성을 읽기 위해 **5일/20일 이동평균선**, 가격 에너지를 측정하는 **전일대비 모멘텀 수익률**, 통계적 변동성 극단선인 **볼린저 밴드(Bollinger Bands)**, 과매수/과매도 구간을 탐지하는 **RSI 상대강도지수**를 유기적으로 연산합니다.

### 3. 🧠 최신 트렌드 강화형 복합 AI 예측 엔진
* 과거의 낡은 시장 규칙에 발목 잡히지 않도록, **최신 1개년 데이터에 2배의 시간 가중치(Time Weighting)**를 부여하여 학습 시 동적 트렌드 반영도를 극대화했습니다.
* 각각 다른 시장 성향(예민한 추세 포착, 리스크 제어 및 안정성 확보 등)을 가진 **3개의 고도화된 부스팅 알고리즘을 결합(Ensemble)**하여 시장 노이즈를 제어하고 단일 모델의 편향을 원천 차단합니다.

### 4. 📱 모바일 최적화 프리미엄 반응형 UI/UX
* 토스(Toss) 등 최신 핀테크 앱 스타일의 카드 레이아웃과 선명한 테두리, 그림자 효과를 반영하여 시각적 밀도와 전문성을 높였습니다.
* 모바일 환경(스마트폰 해상도) 접속 시 메인 타이틀의 반응형 줄바꿈을 지원하며, 빠른 이동 메뉴 클릭 시 **사이드바 제어판이 자동으로 스르륵 접히는 스마트 웹 액션**이 구현되어 있습니다.

---

## 🛠️ 기술 스택 (Technologies Used)

* **Frontend / UI:** Streamlit (Custom CSS & JavaScript Event Capturing Injection)
* **Data Pipeline:** Pandas, NumPy, yfinance
* **Visualization:** Plotly (Interactive Chart Engine)
* **Predictive Algorithm:** LightGBM, XGBoost, CatBoost, Scikit-Learn

---

## 📂 파일 구조 (Repository Structure)

```text
├── app_final.py              # 대시보드 화면 제어 및 통합 분석 백엔드 파이썬 코드
├── financial_data_5years.csv # 거시경제 지표 및 환율 누적 데이터셋 (자동 갱신)
├── requirements.txt         # 클라우드 배포 및 로컬 구동을 위한 필수 라이브러리 목록
└── README.md                # 프로젝트 매뉴얼 및 기술 소개서 (현재 파일)
