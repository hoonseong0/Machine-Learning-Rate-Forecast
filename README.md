# Machine-Learning-Rate-Forecast

https://machine-learning-rate-forecast-enkgpybjjqkngtcjmqhzj3.streamlit.app/

# 📈 AI 원/달러 환율 예측 스마트 가이드 (AI Exchange Rate Prediction Platform)

> **글로벌 거시경제 지표와 다차원 가격 분석 알고리즘을 결합하여 내일의 원/달러 환율 방향성을 객관적 확률로 제시하는 데이터 기반 금융 분석 대시보드 플랫폼입니다.**

본 플랫폼은 외화 자산 운용 효율성을 제고하고 환차손 리스크를 최소화하기 위해 설계되었습니다. 단순한 기술적 지표 분석을 넘어, 다각적 거시경제 데이터의 상관관계를 연산하는 복합 AI 알고리즘을 탑재하여 스마트한 환전 가이드를 제공합니다.

---

✨ 핵심 제공 기능 (Key Features)
1. 🔄 실시간 거시경제 데이터 동기화 파이프라인 (Automated Macro-Data Pipeline)
실시간 데이터 연동: 매일 접속 시각을 기준으로 글로벌 금융 시장의 핵심 지표(USD/KRW, 달러 인덱스(DXY), 미 국채 10년물 금리, 국제 금 가격, KOSPI, S&P 500)를 자동 수집하고 병합합니다.

Concept Drift 방어 로직: 시장 패러다임 변화에 모델이 오염되는 것을 방지하기 위해, 전체 데이터셋을 상시 최신 5년 단위로 슬라이딩 윈도우(Sliding Window) 처리하여 유연한 트렌드 추적 환경을 유지합니다.

2. 📊 다차원 피처 엔지니어링 및 시장 관성 측정 (Multi-dimensional Feature Engineering)
시계열 파생 변수 최적화: 단순 가격 지표를 넘어 시장의 구조적 움직임을 포착하기 위해 5일/20일 이동평균선(MA)을 산출하여 단/중기 추세 강도를 분석합니다.

변동성 및 모멘텀 지표: 환율의 가속도를 측정하는 전일 대비 수익률(Momentum Return), 통계적 가격 이탈 극단선을 나타내는 볼린저 밴드(Bollinger Bands), 추세 전환의 선행 지표인 RSI(상대강도지수)를 유기적으로 연산하여 모델의 판단력을 강화합니다.

3. 🧠 시계열 가중치 기반 하이브리드 앙상블 엔진 (Time-Weighted Hybrid Ensemble Engine)
동적 트렌드 가중치 부여: 과거의 낡은 패턴에 모델이 과적합되는 것을 막기 위해, 최신 1개년 데이터에 2.0의 시간 가중치(Time Weighting)를 부여하여 최근 시장 변화에 대한 학습 민감도를 극대화했습니다.

3대장 부스팅 알고리즘 결합: 빠르고 예민하게 추세를 포착하는 LightGBM, 강력한 규제로 과적합을 방어하는 XGBoost, 데이터 특성을 안정적으로 잡아내는 CatBoost를 결합했습니다.

소프트 보팅(Soft Voting) 로직: 단일 모델의 편향적 오류(승률의 함정)를 차단하기 위해, 세 알고리즘의 상승 예측 확률(%)을 직접 평균 내는 소프트 보팅 방식을 채택하여 보수적이고 안정적인 최종 확신도를 도출합니다.

4. 📱 UI/UX 및 실무 적용성 강화 (Production-Ready Dashboard)
스마트 환전 의사결정 지원: AI 엔진이 연산한 최종 상승 확률을 기반으로, 직관적인 신뢰도(Confidence Level) 기준을 적용하여 즉각적인 환전 및 투자 스탠스(매수/관망/매도)를 텍스트 가이드로 제공합니다.

모델 판단 근거 시각화(XAI): 머신러닝 엔진이 내일의 방향성을 연산할 때 사용한 핵심 지표들의 기여도(Feature Importance) TOP 7을 투명하게 시각화하여, 블랙박스 모델의 한계를 극복하고 분석의 객관성을 증명합니다.

모바일 반응형 인터랙션: 최신 핀테크 서비스 기준의 카드형 레이아웃을 채택했으며, 모바일 환경(768px 이하) 접속 시 사이드바 네비게이션 클릭 이벤트를 캡처링하여 화면을 쾌적하게 닫아주는 JS 로직을 주입했습니다.

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
