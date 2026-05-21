import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests

# ==========================================
# 網頁基本設定
# ==========================================
st.set_page_config(page_title="AI 股票訊號分析工具", layout="wide")
st.title("📈 AI 智能股票技術分析工具 (專業指標版)")

# ==========================================
# 核心功能函數 (加入 MACD, BB, OBV)
# ==========================================
def calculate_indicators(df, short_w, long_w):
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    
    # 1. 移動平均線 (MA)
    df['MA_Short'] = df['Close'].rolling(window=short_w).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_w).mean()
    
    # 2. 相對強弱指數 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. 保歷加通道 (Bollinger Bands - 20日標準)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # 4. 平滑異同移動平均線 (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 5. 能量潮指標 (OBV)
    # 若今日收市升，Volume加；跌則減；平則不變
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
tab1, tab2 = st.tabs(["🔍 單股詳細分析 (多指標)", "📡 主動市場掃描 (高勝率過濾)"])

# ------------------------------------------
# Tab 1: 單股詳細分析
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.text_input("輸入股票代碼 (例如: AAPL, TSLA, 0700.HK)", value="AAPL").upper()
    with col2:
        period_dict = {"1 星期": "5d", "1 個月": "1mo", "3 個月": "3mo", "6 個月": "6mo", "1 年": "1y", "2 年": "2y", "5 年": "5y"}
        period_label = st.selectbox("選取時間範圍", list(period_dict.keys()), index=3)
        period = period_dict[period_label]
    
    if st.button("開始分析", key="analyze_single"):
        with st.spinner('正在獲取數據與計算指標...'):
            fetch_period = period if period not in ["5d", "1mo", "3mo"] else "6mo"
            df = yf.download(ticker, period=fetch_period, progress=False)
            
            if df.empty:
                st.error("❌ 搵唔到數據。")
            else:
                df = calculate_indicators(df, ma_short_window, ma_long_window)
                
                if period == "5d": df = df.tail(5)
                elif period == "1mo": df = df.tail(22)
                elif period == "3mo": df = df.tail(63)
                    
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                # 綜合訊號判斷
                signal = "🔵 觀望 (Neutral)"
                signal_color = "#6c757d"
                
                # MACD 與 MA 雙重確認
                is_golden_cross = prev['MA_Short'] <= prev['MA_Long'] and latest['MA_Short'] > latest['MA_Long']
                is_death_cross = prev['MA_Short'] >= prev['MA_Long'] and latest['MA_Short'] < latest['MA_Long']
                
                if is_golden_cross:
                    if latest['MACD_Hist'] > 0:
                        signal = "🔥 強力買入 (MA黃金交叉 + MACD動能支持)"
                        signal_color = "#198754"
                    else:
                        signal = "🟢 買入 (MA黃金交叉，但需注意 MACD 未轉強)"
                        signal_color = "#28a745"
                        
                elif is_death_cross:
                    signal = "🔴 賣出 (MA死亡交叉)"
                    signal_color = "#dc3545"
                    
                elif latest['Close'] <= latest['BB_Lower'] and latest['RSI'] < 35:
                    signal = "💎 撈底機會 (觸及保歷加通道底 + RSI超賣)"
                    signal_color = "#0dcaf0"
                
                st.markdown(f"### {ticker} 當前綜合訊號: <span style='color:{signal_color}; font-weight:bold;'>{signal}</span>", unsafe_allow_html=True)
                
                # 繪製 4 層 Plotly 專業圖表
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.04, 
                                    row_heights=[0.4, 0.2, 0.2, 0.2],
                                    subplot_titles=("股價與保歷加通道", "MACD 動能", "RSI", "OBV 資金流向"))
                
                # Row 1: 股價 + MA + Bollinger Bands
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收市價', line=dict(color='#1f77b4', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], name=f'{ma_short_window}MA', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], name=f'{ma_long_window}MA', line=dict(color='#2ca02c', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB 頂部', line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB 底部', line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
                
                # Row 2: MACD
                macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD 柱', marker_color=macd_colors), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=2, col=1)
                
                # Row 3: RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9467bd', width=1.5)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                # Row 4: OBV
                fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', line=dict(color='#e377c2', width=2)), row=4, col=1)
                
                fig.update_layout(height=800, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# Tab 2: 主動市場掃描 (高勝率過濾)
# ------------------------------------------
with tab2:
    st.markdown("### 📡 掃描心水股票池，尋找高勝率交易機會")
    
    universe_choice = st.radio("請選擇要掃描的股票池：", ["自訂名單", "🇺🇸 S&P 500 最新成份股", "🇭🇰 恒生指數 最新成份股"], horizontal=True)
    
    tickers = []
    if universe_choice == "自訂名單":
        watchlist_input = st.text_area("設定要掃描的股票名單", value="AAPL, MSFT, NVDA, TSLA, 0700.HK, 0005.HK")
        tickers = [t.strip().upper() for t in watchlist_input.split(',')]
    elif universe_choice == "🇺🇸 S&P 500 最新成份股":
        tickers = get_sp500_tickers()
    elif universe_choice == "🇭🇰 恒生指數 最新成份股":
        tickers = get_hsi_tickers()

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
                    df = yf.download(t, period="6mo", progress=False)
                    if not df.empty and len(df) > ma_long_window:
                        df = calculate_indicators(df, ma_short_window, ma_long_window)
                        latest = df.iloc[-1]
                        prev = df.iloc[-2]
                        
                        signal = "觀望"
                        
                        # 1. 雙重突破策略 (MA交叉 + MACD支持)
                        if prev['MA_Short'] <= prev['MA_Long'] and latest['MA_Short'] > latest['MA_Long']:
                            signal = "🔥 強力買入 (MA+MACD)" if latest['MACD_Hist'] > 0 else "🟢 買入 (MA)"
                        elif prev['MA_Short'] >= prev['MA_Long'] and latest['MA_Short'] < latest['MA_Long']:
                            signal = "🔴 賣出 (MA交叉)"
                            
                        # 2. 撈底反彈策略 (BB底 + RSI超賣)
                        if signal == "觀望" and latest['Close'] <= latest['BB_Lower'] and latest['RSI'] < 35:
                            signal = "💎 撈底 (觸及BB底+超賣)"
                            
                        # 3. 風險預警 (BB頂 + RSI超買)
                        if signal == "觀望" and latest['Close'] >= latest['BB_Upper'] and latest['RSI'] > 70:
                            signal = "⚠️ 風險 (觸及BB頂+超買)"
                        
                        if signal != "觀望":
                            results.append({
                                "股票代碼": t,
                                "最新股價": round(float(latest['Close']), 2),
                                "訊號": signal,
                                "MACD 動能": "正向 🔼" if latest['MACD_Hist'] > 0 else "負向 🔽",
                                "RSI": round(float(latest['RSI']), 2)
                            })
                except Exception:
                    pass
                progress_bar.progress((i + 1) / len(tickers))
                
            status_text.text("✅ 掃描完成！")
            
            if results:
                st.success(f"發現 {len(results)} 隻符合高勝率策略的股票！")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("暫時未發現有強烈買賣訊號的股票。")