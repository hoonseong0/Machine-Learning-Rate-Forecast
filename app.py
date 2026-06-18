import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 페이지 설정 및 사용자 인터페이스(UI) 테마
# ==========================================
st.set_page_config(
    page_title="AI 환율 예측 스마트 가이드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    html { scroll-behavior: smooth; }
    .main-title { font-size: 2.5rem !important; font-weight: 800; color: #0F172A; margin-bottom: 5px; letter-spacing: -0.05rem; }
    .sub-title { font-size: 1.1rem !important; color: #64748B; margin-bottom: 2rem; font-weight: 500; }
    
    .premium-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px 24px;
        border-radius: 12px;
        border: 2px solid #CBD5E1 !important; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; 
    }
    
    .advice-box { background-color: #F0F9FF; padding: 24px; border-left: 6px solid #0EA5E9; border-radius: 8px; margin-top: 15px; }
    .weather-box { background-color: #FFFBEB; padding: 24px; border-left: 6px solid #F59E0B; border-radius: 8px; margin-top: 15px; }
    .indicator-box { background-color: #F8FAFC; padding: 18px; border-radius: 8px; border: 1px solid #E2E8F0; margin-top: 15px; margin-bottom: 20px; }
    
    h3 { color: #0F172A !important; font-weight: 700 !important; font-size: 1.6rem !important; margin-top: 1.5rem !important; }
    h4 { color: #1E293B !important; font-weight: 600 !important; }

    .nav-btn {
        display: block;
        padding: 12px 15px;
        margin-bottom: 10px;
        background-color: #FFFFFF;
        color: #334155 !important;
        text-decoration: none !important;
        border-radius: 8px;
        border: 1.5px solid #CBD5E1; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); 
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease-in-out;
        text-align: center;
    }
    .nav-btn:hover {
        background-color: #F8FAFC;
        border-color: #0EA5E9; 
        color: #0F172A !important;
        transform: translateY(-2px); 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .mobile-break { display: none; } 
    @media (max-width: 768px) {      
        .mobile-break { display: block; } 
        .main-title { font-size: 1.8rem !important; line-height: 1.4; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div id='home'></div>", unsafe_allow_html=True)

# ==========================================
# 2. 글로벌 금융 데이터 수집 및 자동 분석 알고리즘
# ==========================================
@st.cache_data(ttl=3600) 
def get_data_and_predict():
    csv_file = 'financial_data_5years.csv'
    try:
        df = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')
    except:
        st.error("데이터 파일이 없습니다. 'financial_data_5years.csv'를 준비해주세요.")
        st.stop()

    tickers = {
        'USD_KRW': 'USDKRW=X', 'Gold': 'GC=F', 'US10Y': '^TNX',
        'DXY': 'DX-Y.NYB', 'KOSPI': '^KS11', 'SP500': '^GSPC'
    }
    
    last_date = df.index[-1]
    today = datetime.today()
    
    # 1) 실시간 거시경제 데이터 동기화
    if last_date.date() < today.date():
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        
        new_data = pd.DataFrame()
        for col, ticker in tickers.items():
            try:
                temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not temp.empty:
                    close_data = temp['Close'] if 'Close' in temp else temp.iloc[:, 0]
                    new_data[col] = close_data
            except Exception:
                continue
        
        if not new_data.empty:
            new_data.index = pd.to_datetime(new_data.index).normalize()
            df = pd.concat([df, new_data])
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            df = df.fillna(method='ffill')
            
            # 오래된 과거 데이터 필터링 (최신 5년 유지)
            five_years_ago = today - timedelta(days=365 * 5)
            df = df[df.index >= five_years_ago]
            df.to_csv(csv_file)

    # 2) 기술적 보조 지표 및 파생 변수 생성
    df['USD_KRW_MA5'] = df['USD_KRW'].rolling(window=5).mean()
    df['USD_KRW_MA20'] = df['USD_KRW'].rolling(window=20).mean()
    df['USD_KRW_Return'] = df['USD_KRW'].pct_change()
    
    std = df['USD_KRW'].rolling(window=20).std()
    df['BB_Upper'] = df['USD_KRW_MA20'] + (std * 2)
    df['BB_Lower'] = df['USD_KRW_MA20'] - (std * 2)
    
    delta = df['USD_KRW'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Target'] = (df['USD_KRW'].shift(-1) > df['USD_KRW']).astype(int)
    df_clean = df.dropna()

    # 3) 데이터 정규화 및 분석 단위 구조화
    feature_cols = [c for c in df_clean.columns if c != 'Target']
    X = df_clean[feature_cols]
    y = df_clean['Target']
    
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]
    latest_date_real = X.index[-1]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_latest_scaled = scaler.transform(X_latest)
    
    # 4) 최근 1년 시장 트렌드에 대한 중요도 강화
    one_year_ago = X_train.index[-1] - timedelta(days=365)
    sample_weights = np.where(X_train.index >= one_year_ago, 2.0, 1.0)
    
    # 5) 복합 AI 예측 모델 구동 (시장 노이즈 제어 알고리즘 적용)
    lgb_model = lgb.LGBMClassifier(learning_rate=0.03, n_estimators=100, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    xgb_model = xgb.XGBClassifier(learning_rate=0.03, n_estimators=100, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_seed=42)
    cat_model = CatBoostClassifier(learning_rate=0.03, iterations=100, depth=5, random_seed=42, verbose=0)
    
    lgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    cat_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # 6) 개별 AI 결과 기반 종합 확률 산출
    p_lgb = lgb_model.predict_proba(X_latest_scaled)[0][1]
    p_xgb = xgb_model.predict_proba(X_latest_scaled)[0][1]
    p_cat = cat_model.predict_proba(X_latest_scaled)[0][1]
    prob_up = (p_lgb + p_xgb + p_cat) / 3
    
    return df_clean, prob_up, latest_date_real, X_latest, X.iloc[-2], lgb_model, X_train

with st.spinner("AI가 글로벌 금융 데이터를 분석하고 있습니다..."):
    df_clean, prob_up, latest_date, latest_data, prev_data, lgb_model, X_train = get_data_and_predict()

# ==========================================
# 3. 사이드바 제어판 및 네비게이션
# ==========================================
st.sidebar.markdown("### ⚙️ 대시보드 제어판")

if st.sidebar.button("🔄 최신 금융 정보 동기화", use_container_width=True):
    st.cache_data.clear() 
    st.rerun() 

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 차트 조회 기간 설정**")
period_option = st.sidebar.radio("기간을 선택하세요", ["1주일", "1개월", "3개월", "6개월", "1년"], index=2, label_visibility="collapsed")

days_dict = {"1주일": 7, "1개월": 30, "3개월": 90, "6개월": 180, "1년": 365}
chart_days = days_dict[period_option]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 빠른 이동 메뉴")

sidebar_nav = """
    <a href="#home" class="nav-btn">🏠 맨 위로 (홈)</a>
    <a href="#chart-section" class="nav-btn">📊 실시간 환율 차트</a>
    <a href="#ai-market-analysis" class="nav-btn">💡 오늘의 시장 상황 분석</a>
    <a href="#ai-feature-importance" class="nav-btn">🤖 AI 핵심 지표 분석</a>
"""
st.sidebar.markdown(sidebar_nav, unsafe_allow_html=True)

# 모바일 환경 사이드바 자동 닫힘 처리
st.components.v1.html(
    """
    <script>
    const parentDoc = window.parent.document;
    
    if (!parentDoc.getElementById('smooth-sidebar-closer')) {
        const marker = parentDoc.createElement('div');
        marker.id = 'smooth-sidebar-closer';
        marker.style.display = 'none';
        parentDoc.body.appendChild(marker);

        parentDoc.addEventListener('click', function(e) {
            const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
            
            if (sidebar && sidebar.contains(e.target)) {
                const isTarget = e.target.closest('.stButton button') || 
                                 e.target.closest('[data-testid="stRadio"]') || 
                                 e.target.closest('.nav-btn');
                
                if (isTarget && window.parent.innerWidth <= 768) {
                    setTimeout(function() {
                        const closeBtn = sidebar.querySelector('button');
                        if (closeBtn) {
                            closeBtn.click();
                        }
                    }, 150); 
                }
            }
        }, true); 
    }
    </script>
    """,
    height=0, width=0
)

# ==========================================
# 4. 상단 메인 패널 (핵심 지표 요약)
# ==========================================
st.markdown('<div class="main-title">📈 AI 환율 예측 <br class="mobile-break">스마트 가이드</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">글로벌 금융 데이터 기준일: {latest_date.strftime("%Y년 %m월 %d일")}</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
current_rate = df_clean['USD_KRW'].iloc[-1]
prev_rate = df_clean['USD_KRW'].iloc[-2]
rate_diff = current_rate - prev_rate

with col1:
    st.metric(label="현재 원/달러 환율 종가", value=f"{current_rate:,.2f} 원", delta=f"{rate_diff:+.2f} 원")
with col2:
    pred_text = "상승 예상 📈" if prob_up >= 0.5 else "하락 안정 📉"
    st.metric(label="내일 환율 추세 방향성", value=pred_text)
with col3:
    conf_pct = (prob_up if prob_up >= 0.5 else (1 - prob_up)) * 100
    if conf_pct >= 70: conf_status = "매우 높음 (확실한 추세)"
    elif conf_pct >= 60: conf_status = "보통 (참고 지표)"
    else: conf_status = "낮음 (관망 필요)"
    st.metric(label="AI 종합 예측 확신도", value=f"{conf_pct:.1f}%", delta=conf_status, delta_color="off")

# ==========================================
# 5. 중앙 메인 차트
# ==========================================
st.markdown("<div id='chart-section'></div>", unsafe_allow_html=True)
st.markdown("### 📊 실시간 원/달러 환율 추세 차트")

chart_df = df_clean.tail(chart_days)
fig = go.Figure()
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['USD_KRW'], name='환율 종가', mode='lines', line=dict(color='#0F172A', width=3)))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['USD_KRW_MA20'], name='20일 중기 트렌드선', mode='lines', line=dict(color='#6366F1', width=1.5, dash='dot')))

y_min, y_max = chart_df['USD_KRW'].min() * 0.995, chart_df['USD_KRW'].max() * 1.005
fig.update_layout(
    template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), 
    yaxis=dict(range=[y_min, y_max], tickformat=",.1f", gridcolor='#F1F5F9'), 
    xaxis=dict(gridcolor='#F1F5F9'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
    height=480
)

st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 6. 하단 심층 분석 가이드
# ==========================================
st.markdown("<hr style='margin-top: 3rem; border: 1px dashed #CBD5E1;'><div id='ai-market-analysis'></div>", unsafe_allow_html=True)
st.markdown("### 💡 AI가 분석한 오늘의 시장 상황")

if prob_up >= 0.65:
    action_text, icon = "**달러가 비싸질 확률이 높습니다.** 환전 계획이 있으시다면 **오늘 미리 처리하시는 것을 추천**합니다.", "💸"
elif prob_up <= 0.35:
    action_text, icon = "**달러가 저렴해질 확률이 높습니다.** 급하지 않다면 환전을 **며칠 미루고 관망하시는 것이 유리**할 수 있습니다.", "⏳"
else:
    action_text, icon = "**방향성이 뚜렷하지 않은 날입니다.** 시장에 큰 영향을 줄 경제 지표 발표를 기다리며 분할 환전하는 것을 추천합니다.", "⚖️"

dxy_change = latest_data['DXY'].values[0] - prev_data['DXY']
us10y_change = latest_data['US10Y'].values[0] - prev_data['US10Y']

if dxy_change > 0 and us10y_change > 0:
    weather_text = "현재 **미국 달러의 가치(DXY)가 강세**를 보이고 있으며, **미국채 금리도 함께 상승**하고 있어 당분간 환율이 높게 유지될 가능성이 큽니다."
elif dxy_change < 0 and us10y_change < 0:
    weather_text = "미국 달러의 글로벌 가치가 떨어지고 국채 금리도 하락하고 있어, 원화 가치가 상대적으로 회복(환율 하락)될 좋은 환경이 조성되고 있습니다."
else:
    weather_text = "글로벌 달러 가치와 금리의 방향이 혼재되어 있습니다. 시장이 새로운 이슈(뉴스)를 기다리며 눈치 보기를 하고 있는 상황입니다."

st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown(f'<div class="advice-box"><h4>{icon} 스마트 환전 가이드</h4><p style="font-size: 1.1rem; color: #334155; margin-bottom:0px;">{action_text}</p></div>', unsafe_allow_html=True)
st.markdown(f'<div class="weather-box"><h4>🔍 주요 원인 요약</h4><p style="font-size: 1rem; color: #475569; margin-bottom:0px;">{weather_text}</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. AI 핵심 지표 기여도 분석
# ==========================================
st.markdown("<br><hr style='margin-top: 1rem; border: 1px dashed #CBD5E1;'><div id='ai-feature-importance'></div>", unsafe_allow_html=True)
st.markdown("### 🤖 AI 핵심 지표 판단 근거 분석")

name_mapping = {
    'USD_KRW': '원/달러 환율 종가', 'Gold': '국제 금 가격', 'US10Y': '미 국채 10년물 금리',
    'DXY': '달러 인덱스 (DXY)', 'KOSPI': '코스피 지수', 'SP500': 'S&P 500 지수',
    'USD_KRW_MA5': '단기 5일 이동평균선', 'USD_KRW_MA20': '중기 20일 이동평균선',
    'USD_KRW_Return': '환율 전일대비 수익률', 'BB_Upper': '볼린저밴드 상단벽',
    'BB_Lower': '볼린저밴드 하단벽', 'RSI': 'RSI 상대강도지수'
}

indicator_explanations = {
    'USD_KRW': '자신의 직전 가격 패턴을 분석하여 추세의 지속 여부를 판단합니다.',
    'Gold': '대표적인 안전자산으로, 달러 가치와 보통 반대로 움직이며 위험 심리를 대변합니다.',
    'US10Y': '미국 금리의 척도로, 이 수치가 상승하면 자본이 달러 자산으로 흡수되어 환율이 강해집니다.',
    'DXY': '글로벌 주요 6개 통화 대비 달러의 절대 가치로, 원/달러 환율과 직결되는 나침반입니다.',
    'KOSPI': '국내 증시의 매도세/매수세를 확인하며, 외국인 투자자의 자금 이탈 규모를 감지합니다.',
    'SP500': '글로벌 위험자산 선호도를 뜻하며, 폭락 시 안전자산인 달러 풀수요가 급증합니다.',
    'USD_KRW_MA5': '일주일간의 단기 모멘텀 에너지를 파악하여 단기 상방/하방 압력을 측정합니다.',
    'USD_KRW_MA20': '시장의 중기 생명선으로, 현재 환율이 평균 대비 과도하게 튀었는지 확인합니다.',
    'USD_KRW_Return': '전날 가파르게 올랐다면 심리적 차익실현 매물이 나올 확률을 계산합니다.',
    'BB_Upper': '통계적 상승 극단선으로, 이 벽에 부딪히면 단기 저항을 받을 확률이 높습니다.',
    'BB_Lower': '통계적 하락 극단선으로, 이 바닥선에 도달하면 강한 기술적 반등 흐름이 연출됩니다.',
    'RSI': '과매수(70이상) 혹은 과매도(30이하) 상태를 감지하여 기술적 추세 전환점을 예측합니다.'
}

st.markdown('<div class="premium-card">', unsafe_allow_html=True)

feat_imp = pd.Series(lgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
top_features_all = feat_imp.tail(7)

fig_imp = go.Figure(go.Bar(
    x=top_features_all.values, 
    y=[name_mapping.get(col, col) for col in top_features_all.index], 
    orientation='h',
    marker=dict(color='#6366F1', line=dict(color='#4F46E5', width=1)) 
))
fig_imp.update_layout(
    title="🤖 AI 분석 엔진에서 채택된 결정적 영향력 TOP 7 지표",
    xaxis_title="지표별 상대적 기여도 (비중)",
    yaxis_title="시장 변수 명",
    template="plotly_white",
    height=320,
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("""
    <div class="indicator-box">
        📌 <strong>부연 설명:</strong> 위 그래프의 수치는 복합 인공지능이 내일의 원/달러 환율 방향성을 분석할 때, 
        가장 중요하게 판단 기준으로 삼은 <strong>핵심 지표들의 기여도(비중)</strong>입니다. 
        단순히 환율 하나만 보는 것이 아니라, 글로벌 거시경제 변수와 기술적 보조 지표들을 종합 분석하여 객관적인 단서를 도출해 냅니다.
    </div>
""", unsafe_allow_html=True)

st.markdown("<br><h4 style='color:#1E293B; margin-bottom: 5px;'>📅 중요도 상위 변수들의 최신 1주일(7 영업일) 고정 추세 분석</h4>", unsafe_allow_html=True)
st.write("인공지능이 판단 비중을 가장 높게 둔 상위 3가지 지표의 실시간 가격 변동 가속도를 시각화합니다.")

sorted_features = feat_imp.sort_values(ascending=False)
top_3_indicators = sorted_features.index[:3] 

chart_cols = st.columns(3)
week_df = df_clean.tail(7) 

custom_colors = ['#0EA5E9', '#F43F5E', '#10B981']

for i, col_name in enumerate(top_3_indicators):
    with chart_cols[i]:
        fig_week = go.Figure()
        
        fig_week.add_trace(go.Scatter(
            x=week_df.index, 
            y=week_df[col_name], 
            mode='lines',
            name=col_name,
            line=dict(color=custom_colors[i], width=3)
        ))
        
        w_min, w_max = week_df[col_name].min(), week_df[col_name].max()
        y_range = [w_min * 0.998, w_max * 1.002] if w_min != w_max else [w_min - 1, w_max + 1]
        
        h_name = name_mapping.get(col_name, col_name)
        fig_week.update_layout(
            title=f"우선순위 {i+1} : {h_name}",
            template="plotly_white",
            height=240,
            margin=dict(l=5, r=5, t=40, b=5),
            xaxis=dict(tickformat="%m/%d", nticks=7),
            yaxis=dict(range=y_range, tickformat=",.2f")
        )
        st.plotly_chart(fig_week, use_container_width=True)
        
        explanation = indicator_explanations.get(col_name, "시장 변동성에 유의미한 영향력을 행사하는 지표입니다.")
        st.caption(f"💡 **해설:** {explanation}")

st.markdown('</div>', unsafe_allow_html=True)