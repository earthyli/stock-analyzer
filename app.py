import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import requests
import io

# ==========================================
# 網頁基本設定
# ==========================================
st.set_page_config(page_title="AI 股票訊號分析工具", layout="wide")
st.title("📈 AI 智能股票技術分析工具")

# ==========================================
# 核心功能函數 (Functions)
# ==========================================
def calculate_indicators(df, short_w, long_w):
    """計算技術指標 (MA 與 RSI)"""
    # 修正 yfinance 新版本可能帶來的 MultiIndex 欄位問題
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    
    df['MA_Short'] = df['Close'].rolling(window=short_w).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_w).mean()
    
    # 計算 RSI (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """自動從 Wikipedia 抓取 S&P 500 最新成份股"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    table = pd.read_html(io.StringIO(response.text))[0]
    # 將欄位 Symbol 抽出來，並處理 Yahoo Finance 格式 (例如 BRK.B -> BRK-B)
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

@st.cache_data(ttl=86400)
def get_hsi_tickers():
    """自動從 Wikipedia 抓取恒生指數最新成份股"""
    url = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    for df in tables:
        if 'Ticker' in df.columns:
            # 提取數字並補齊 4 位數字加上 .HK
            numbers = df['Ticker'].astype(str).str.extract(r'(\d+)')[0].dropna()
            tickers = numbers.apply(lambda x: x.zfill(4) + '.HK').tolist()
            return tickers
    return ["0700.HK", "9988.HK", "0005.HK"] # 保底名單

# ==========================================
# 側邊欄設定 (Sidebar)
# ==========================================
st.sidebar.header("⚙️ 設定參數")
ma_short_window = st.sidebar.slider("短期平均線 (MA) 天數", 5, 30, 20)
ma_long_window = st.sidebar.slider("長期平均線 (MA) 天數", 30, 100, 50)

# ==========================================
# 頁面標籤 (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🔍 單股詳細分析", "📡 主動市場掃描 (Screener)"])

# ------------------------------------------
# Tab 1: 單股詳細分析
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.text_input("輸入股票代碼 (例如: AAPL, TSLA, 0700.HK)", value="AAPL").upper()
    with col2:
        period = st.selectbox("選取時間範圍", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    
    if st.button("開始分析", key="analyze_single"):
        with st.spinner('正在從 Yahoo Finance 獲取數據...'):
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty:
                st.error("❌ 搵唔到數據，請檢查股票代碼是否正確。")
            else:
                df = calculate_indicators(df, ma_short_window, ma_long_window)
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                # 訊號判斷
                signal = "🔵 觀望 (Neutral)"
                signal_color = "#6c757d"
                
                if prev['MA_Short'] <= prev['MA_Long'] and latest['MA_Short'] > latest['MA_Long']:
                    signal = "🟢 買入訊號 (BUY) - 黃金交叉"
                    signal_color = "#28a745"
                elif prev['MA_Short'] >= prev['MA_Long'] and latest['MA_Short'] < latest['MA_Long']:
                    signal = "🔴 賣出訊號 (SELL) - 死亡交叉"
                    signal_color = "#dc3545"
                
                # 顯示當前價錢與訊號
                st.markdown(f"### {ticker} 當前訊號: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("最新收市價", f"${float(latest['Close']):.2f}")
                col_m2.metric("當前 RSI (14)", f"{float(latest['RSI']):.2f}")
                col_m3.metric("短期 / 長期 MA", f"{float(latest['MA_Short']):.2f} / {float(latest['MA_Long']):.2f}")
                
                # 繪製 Plotly 圖表
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
                
                # 上圖：收市價與 MA
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收市價 (Close)', line=dict(color='#1f77b4', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], name=f'{ma_short_window} MA', line=dict(color='#ff7f0e', width=1.5, dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], name=f'{ma_long_window} MA', line=dict(color='#2ca02c', width=1.5)), row=1, col=1)
                
                # 下圖：RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9467bd', width=1.5)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超買 (70)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超賣 (30)", row=2, col=1)
                
                fig.update_layout(height=600, title_text=f"{ticker} 技術分析圖表", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# Tab 2: 主動市場掃描 (Screener)
# ------------------------------------------
with tab2:
    st.markdown("### 📡 掃描心水股票池，尋找交易機會")
    
    universe_choice = st.radio(
        "請選擇要掃描的股票池：",
        ["自訂名單", "🇺🇸 S&P 500 最新成份股 (自動抓取)", "🇭🇰 恒生指數 最新成份股 (自動抓取)"],
        horizontal=True
    )
    
    tickers = []
    if universe_choice == "自訂名單":
        default_watchlist = "AAPL, MSFT, NVDA, TSLA, 0700.HK, 0005.HK"
        watchlist_input = st.text_area("設定要掃描的股票名單 (用逗號分隔)", value=default_watchlist)
        tickers = [t.strip().upper() for t in watchlist_input.split(',')]
    elif universe_choice == "🇺🇸 S&P 500 最新成份股 (自動抓取)":
        with st.spinner('正在從 Wikipedia 抓取 S&P 500 名單...'):
            tickers = get_sp500_tickers()
            st.info(f"✅ 成功獲取 {len(tickers)} 隻 S&P 500 成份股！")
    elif universe_choice == "🇭🇰 恒生指數 最新成份股 (自動抓取)":
        with st.spinner('正在從 Wikipedia 抓取 恒生指數 名單...'):
            tickers = get_hsi_tickers()
            st.info(f"✅ 成功獲取 {len(tickers)} 隻恒生指數成份股！")

    if st.button("🚀 開始全網掃描", key="scan_btn"):
        if not tickers:
            st.warning("股票名單為空！")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, t in enumerate(tickers):
                status_text.text(f"正在掃描: {t} ({i+1}/{len(tickers)})...")
                try:
                    # 只需要最近 6 個月數據作訊號判斷，加快速度
                    df = yf.download(t, period="6mo", progress=False)
                    
                    if not df.empty and len(df) > ma_long_window:
                        df = calculate_indicators(df, ma_short_window, ma_long_window)
                        latest = df.iloc[-1]
                        prev = df.iloc[-2]
                        
                        signal = "觀望"
                        if prev['MA_Short'] <= prev['MA_Long'] and latest['MA_Short'] > latest['MA_Long']:
                            signal = "🟢 買入 (黃金交叉)"
                        elif prev['MA_Short'] >= prev['MA_Long'] and latest['MA_Short'] < latest['MA_Long']:
                            signal = "🔴 賣出 (死亡交叉)"
                        
                        # --- RSI 預警邏輯 ---
                        display_signal = signal
                        if signal == "觀望":
                            if latest['RSI'] > 70:
                                display_signal = "⚠️ 留意 (RSI 超買)"
                            elif latest['RSI'] < 30:
                                display_signal = "👀 留意 (RSI 超賣)"
                        
                        # 篩選條件：只要 display_signal 唔係「觀望」就上榜
                        if display_signal != "觀望":
                            results.append({
                                "股票代碼": t,
                                "最新股價": round(float(latest['Close']), 2),
                                "訊號": display_signal,
                                "RSI": round(float(latest['RSI']), 2)
                            })
                except Exception:
                    pass # 忽略抓取失敗的股票
                
                # 更新進度條
                progress_bar.progress((i + 1) / len(tickers))
                
            status_text.text("✅ 掃描完成！")
            
            if results:
                st.success(f"發現 {len(results)} 隻值得留意的股票！")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("暫時未發現有強烈買賣訊號的股票。")