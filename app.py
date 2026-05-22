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
st.title("📈 智能股票技術分析工具 (專業指標版)")

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
def get_sp500_mapping():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        table = pd.read_html(io.StringIO(response.text))[0]
        tickers = table['Symbol'].str.replace('.', '-', regex=False)
        names = table['Security']
        return dict(zip(tickers, names))
    except:
        return {"AAPL": "Apple Inc.", "MSFT": "Microsoft"}

@st.cache_data(ttl=86400)
def get_hsi_mapping():
    url = 'https://zh.wikipedia.org/wiki/%E6%81%92%E7%94%9F%E6%8C%87%E6%95%B8'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            has_code = any('代號' in c or '編號' in c for c in cols)
            has_name = any('公司' in c or '名稱' in c or '簡稱' in c or '成份股' in c for c in cols)
            
            if has_code and has_name:
                code_col = df.columns[[i for i, c in enumerate(cols) if '代號' in c or '編號' in c][0]]
                name_col = df.columns[[i for i, c in enumerate(cols) if '公司' in c or '名稱' in c or '簡稱' in c or '成份股' in c][0]]
                
                mapping = {}
                for _, row in df.iterrows():
                    code_val = str(row[code_col]).strip()
                    numbers = ''.join(filter(str.isdigit, code_val))
                    if numbers:
                        ticker = numbers.zfill(4) + '.HK'
                        mapping[ticker] = str(row[name_col]).strip()
                if mapping:
                    return mapping
    except Exception:
        pass
    return {"0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "0005.HK": "匯豐控股", "0388.HK": "香港交易所"}

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
        ticker = st.text_input("輸入股票代碼 (例如: AAPL, TSLA, 0700.HK)", value="0700.HK").upper()
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
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                
                try:
                    stock_name = yf.Ticker(ticker).info.get('shortName', '')
                    title_display = f"{ticker} {stock_name}" if stock_name else ticker
                except:
                    title_display = ticker
                
                signal = "🔵 觀望 (Neutral)"
                signal_color = "#6c757d"
                explanation = "目前未有明顯突破或極端情緒，建議耐心等待明確方向。"
                
                is_golden_cross = prev['MA_Short'] <= prev['MA_Long'] and latest['MA_Short'] > latest['MA_Long']
                is_death_cross = prev['MA_Short'] >= prev['MA_Long'] and latest['MA_Short'] < latest['MA_Long']
                
                if is_golden_cross:
                    if latest['MACD_Hist'] > 0:
                        signal = "🔥 強力買入 (MA黃金交叉 + MACD動能支持)"
                        signal_color = "#198754"
                        explanation = "📌 **策略：長線趨勢**\n大趨勢已向上突破，且資金動力強勁，勝率較高，適合建倉或加注。"
                    else:
                        signal = "🟢 買入 (MA黃金交叉，但需注意 MACD 未轉強)"
                        signal_color = "#28a745"
                        explanation = "📌 **策略：長線趨勢**\n大趨勢初步轉好，但短期推動力未算強烈，可能會有反覆，建議分注買入。"
                        
                elif is_death_cross:
                    signal = "🔴 賣出 (MA死亡交叉)"
                    signal_color = "#dc3545"
                    explanation = "📌 **策略：風險控制**\n大趨勢已確認轉弱，下行風險增加，建議減倉或嚴格執行止蝕。"
                    
                elif latest['Close'] <= latest['BB_Lower'] and latest['RSI'] < 35:
                    signal = "💎 撈底機會 (觸及保歷加通道底 + RSI超賣)"
                    signal_color = "#0dcaf0"
                    explanation = "📌 **策略：短線博弈 (逆勢操作)**\n市場出現過度恐慌，短線極度超賣。適合小注博取技術性反彈，有賺即走，若跌穿前低位必須立即止蝕。"
                
                st.markdown(f"### {title_display} 當前綜合訊號 (數據日期: {latest_date})")
                st.markdown(f"<span style='color:{signal_color}; font-weight:bold; font-size:22px;'>{signal}</span>", unsafe_allow_html=True)
                st.info(explanation)
                
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.04, 
                                    row_heights=[0.4, 0.2, 0.2, 0.2],
                                    subplot_titles=("股價與保歷加通道", "MACD 動能", "RSI", "OBV 資金流向"))
                
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收市價', line=dict(color='#1f77b4', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], name=f'{ma_short_window}MA', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], name=f'{ma_long_window}MA', line=dict(color='#2ca02c', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB 頂部', line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB 底部', line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
                
                macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD 柱', marker_color=macd_colors), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9467bd', width=1.5)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
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
    ticker_name_map = {}
    
    if universe_choice == "自訂名單":
        watchlist_input = st.text_area("設定要掃描的股票名單", value="AAPL, MSFT, NVDA, TSLA, 0700.HK, 0005.HK")
        tickers = [t.strip().upper() for t in watchlist_input.split(',')]
    elif universe_choice == "🇺🇸 S&P 500 最新成份股":
        with st.spinner('正在從 Wikipedia 抓取 S&P 500 名單...'):
            ticker_name_map = get_sp500_mapping()
            tickers = list(ticker_name_map.keys())
            st.info(f"✅ 成功獲取 {len(tickers)} 隻 S&P 500 成份股！")
    elif universe_choice == "🇭🇰 恒生指數 最新成份股":
        with st.spinner('正在從中文版 Wikipedia 抓取 恒生指數 名單...'):
            ticker_name_map = get_hsi_mapping()
            tickers = list(ticker_name_map.keys())
            st.info(f"✅ 成功獲取 {len(tickers)} 隻恒生指數成份股！")
            
    if st.button("🚀 開始全網掃描", key="scan_btn"):
        if not tickers:
            st.warning("股票名單為空！")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, t in enumerate(tickers):
                comp_name = ticker_name_map.get(t, "")
                if not comp_name:
                    try:
                        comp_name = yf.Ticker(t).info.get('shortName', 'N/A')
                    except:
                        comp_name = "N/A"
                        
                display_label = f"{t} ({comp_name})" if comp_name != "N/A" else t
                status_text.text(f"正在掃描: {display_label} ({i+1}/{len(tickers)})...")
                
                try:
                    df = yf.download(t, period="6mo", progress=False)
                    if not df.empty and len(df) > ma_long_window:
                        df = calculate_indicators(df, ma_short_window, ma_long_window)
                        latest = df.iloc[-1]
                        prev = df.iloc[-2]
                        latest_date = df.index[-1].strftime('%Y-%m-%d')
                        
                        signal = "觀望"
                        strategy_type = ""
                        advice = ""
                        
                        if prev['MA_Short'] <= prev['MA_Long'] and latest['MA_Short'] > latest['MA_Long']:
                            if latest['MACD_Hist'] > 0:
                                signal = "🔥 強力買入"
                                strategy_type = "📈 長線趨勢"
                                advice = "動能與趨勢俱佳，適合建倉"
                            else:
                                signal = "🟢 買入"
                                strategy_type = "📈 長線趨勢"
                                advice = "趨勢轉好但動能稍弱，宜分注買入"
                        elif prev['MA_Short'] >= prev['MA_Long'] and latest['MA_Short'] < latest['MA_Long']:
                            signal = "🔴 賣出"
                            strategy_type = "📉 風險控制"
                            advice = "大勢轉弱，建議減倉或止蝕"
                            
                        if signal == "觀望" and latest['Close'] <= latest['BB_Lower'] and latest['RSI'] < 35:
                            signal = "💎 撈底"
                            strategy_type = "⚡ 短線博弈"
                            advice = "超賣極端狀態，博反彈，嚴設止蝕"
                            
                        if signal == "觀望" and latest['Close'] >= latest['BB_Upper'] and latest['RSI'] > 70:
                            signal = "⚠️ 風險"
                            strategy_type = "⚠️ 風險控制"
                            advice = "短期升幅急，留意回調，準備食糊"
                        
                        if signal != "觀望":
                            results.append({
                                "股票代碼": t,
                                "公司名稱": comp_name,
                                "數據日期": latest_date,
                                "最新股價": round(float(latest['Close']), 2),
                                "策略分類": strategy_type,
                                "訊號": signal,
                                "MACD 動能": "正向 🔼" if latest['MACD_Hist'] > 0 else "負向 🔽",
                                "RSI": round(float(latest['RSI']), 2),
                                "操作建議": advice
                            })
                except Exception:
                    pass
                progress_bar.progress((i + 1) / len(tickers))
                
            status_text.text("✅ 掃描完成！")
            
            if results:
                st.success(f"發現 {len(results)} 隻符合高勝率策略的股票！")
                df_results = pd.DataFrame(results)
                cols = ["股票代碼", "公司名稱", "最新股價", "訊號", "策略分類", "MACD 動能", "RSI", "操作建議", "數據日期"]
                st.dataframe(df_results[cols], use_container_width=True)
            else:
                st.info("暫時未發現有強烈買賣訊號的股票。")