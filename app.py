import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests
import google.generativeai as genai

# ==========================================
# 網頁基本設定
# ==========================================
st.set_page_config(page_title="AI 股票訊號分析工具", layout="wide")
st.title("📈 AI 智能股票技術分析工具 (專業完整版)")

# ==========================================
# 核心功能函數
# ==========================================
def calculate_indicators(df, short_w, long_w):
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df['MA_Short'] = df['Close'].rolling(window=short_w).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_w).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    conditions = [df['Close'] > df['Close'].shift(1), df['Close'] < df['Close'].shift(1)]
    choices = [1, -1]
    df['Direction'] = np.select(conditions, choices, default=0)
    df['OBV'] = (df['Volume'] * df['Direction']).cumsum()
    return df

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    table = pd.read_html(io.StringIO(response.text))[0]
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()

@st.cache_data(ttl=86400)
def get_hsi_tickers():
    url = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    for df in tables:
        if 'Ticker' in df.columns:
            numbers = df['Ticker'].astype(str).str.extract(r'(\d+)')[0].dropna()
            return numbers.apply(lambda x: x.zfill(4) + '.HK').tolist()
    return ["0700.HK", "9988.HK", "0005.HK"]

# ==========================================
# 側邊欄設定
# ==========================================
st.sidebar.header("⚙️ 設定參數")
ma_short_window = st.sidebar.slider("短期平均線 (MA) 天數", 5, 30, 20)
ma_long_window = st.sidebar.slider("長期平均線 (MA) 天數", 30, 100, 50)

# ==========================================
# 頁面標籤 (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🔍 單股詳細分析", "📡 主動市場掃描"])

# ------------------------------------------
# Tab 1: 單股詳細分析 (完整專業圖表版)
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.text_input("輸入股票代碼", value="AAPL").upper()
    with col2:
        period_dict = {"3 個月": "3mo", "6 個月": "6mo", "1 年": "1y"}
        period_label = st.selectbox("時間範圍", list(period_dict.keys()), index=1)
        period = period_dict[period_label]
    
    if st.button("開始分析"):
        with st.spinner('計算專業指標中...'):
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                df = calculate_indicators(df, ma_short_window, ma_long_window)
                
                # 繪製專業 4 層圖表
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
                                    row_heights=[0.4, 0.2, 0.2, 0.2],
                                    subplot_titles=("股價與保歷加通道", "MACD 動能", "RSI", "OBV 資金流向"))
                
                # Layer 1: 股價 + MA + Bollinger Bands
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收市價', line=dict(color='#1f77b4', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], name='MA Short', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], name='MA Long', line=dict(color='#2ca02c', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB 頂', line=dict(color='gray', dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB 底', line=dict(color='gray', dash='dot'), fill='tonexty'), row=1, col=1)
                
                # Layer 2: MACD
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD 柱'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=2, col=1)
                
                # Layer 3: RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9467bd')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                # Layer 4: OBV
                fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', line=dict(color='#e377c2')), row=4, col=1)
                
                fig.update_layout(height=800, hovermode="x unified", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("搵唔到數據。")

# ------------------------------------------
# Tab 2: 主動市場掃描 (加入 AI 報告)
# ------------------------------------------
with tab2:
    universe_choice = st.radio("選擇股票池：", ["自訂名單", "🇺🇸 S&P 500", "🇭🇰 恒生指數"], horizontal=True)
    tickers = get_sp500_tickers() if universe_choice == "🇺🇸 S&P 500" else (get_hsi_tickers() if universe_choice == "🇭🇰 恒生指數" else st.text_area("設定名單", value="AAPL, 0700.HK").split(','))
    
    if st.button("🚀 開始掃描"):
        results = []
        progress_bar = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                df = yf.download(t.strip(), period="6mo", progress=False)
                if not df.empty:
                    df = calculate_indicators(df, ma_short_window, ma_long_window)
                    latest = df.iloc[-1]
                    if latest['Close'] <= latest['BB_Lower'] and latest['RSI'] < 35:
                        results.append({"股票": t.strip(), "現價": round(float(latest['Close']),2), "訊號": "💎 撈底"})
            except: continue
            progress_bar.progress((i + 1) / len(tickers))
            
        if results:
            st.dataframe(pd.DataFrame(results))
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"分析這些股票並寫一段廣東話市場簡報: {results}")
                st.markdown("### 🤖 智能市場總結")
                st.info(response.text)
            except:
                st.error("AI 報告生成失敗 (請檢查 Secrets 設定)")