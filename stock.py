# -*- coding: utf-8 -*-
"""
台股 AI 綜合量化分析儀表板 - 獨立核心模組
"""
import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 資料清洗函式 ---
def _clean_yf_data(data: pd.DataFrame) -> pd.DataFrame:
    """穩健清洗 yfinance MultiIndex 與欄位標準化"""
    cleaned = data.copy()
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [str(col[0]).lower().strip() for col in cleaned.columns]
    else:
        cleaned.columns = [str(col).lower().strip() for col in cleaned.columns]
    cleaned.columns.name = None
    if 'adj close' in cleaned.columns:
        cleaned['close'] = cleaned['adj close']
    target_cols = ['open', 'high', 'low', 'close', 'volume']
    existing_cols = [c for c in target_cols if c in cleaned.columns]
    return cleaned[existing_cols].dropna()

# --- 2. 單檔量化分析核心類別 ---
class StockAnalyzer:
    def __init__(self, ticker: str = "3231.TW", period: str = "1y"):
        self.ticker = ticker.strip().upper()
        if not (self.ticker.endswith('.TW') or self.ticker.endswith('.TWO') or self.ticker.startswith('^')):
            self.ticker = f"{self.ticker}.TW"
        self.period = period
        self.df = pd.DataFrame()

    def fetch_and_calculate(self):
        raw = yf.download(self.ticker, period=self.period, auto_adjust=False, progress=False)
        if raw.empty:
            raise ValueError(f"查無標的代號【{self.ticker}】的歷史數據。")
        self.df = _clean_yf_data(raw)
        if len(self.df) < 30:
            raise ValueError("歷史交易日筆數不足，無法計算全套技術指標。")

        # 均線五線譜矩陣
        for p in [5, 10, 20, 60, 120, 240]:
            self.df[f'ma_{p}'] = self.df['close'].rolling(p).mean()

        # 乖離率與累積報酬率
        self.df['bias_20'] = ((self.df['close'] - self.df['ma_20']) / self.df['ma_20']) * 100
        self.df['ret_1d'] = self.df['close'].pct_change(1) * 100

        # RSI(14)
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = np.where(loss == 0, 0, gain / loss)
        self.df['rsi_14'] = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))

        # MACD (12, 26, 9)
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd_dif'] = ema12 - ema26
        self.df['macd_dem'] = self.df['macd_dif'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd_dif'] - self.df['macd_dem']

        # 布林通道 (20MA, 2σ)
        bb_std = self.df['close'].rolling(20).std()
        self.df['bb_upper'] = self.df['ma_20'] + 2 * bb_std
        self.df['bb_lower'] = self.df['ma_20'] - 2 * bb_std
        self.df['bb_width'] = ((self.df['bb_upper'] - self.df['bb_lower']) / self.df['ma_20']) * 100

        # ATR(14) 動態停損
        prev_c = self.df['close'].shift(1)
        tr = pd.concat([
            self.df['high'] - self.df['low'],
            (self.df['high'] - prev_c).abs(),
            (self.df['low'] - prev_c).abs()
        ], axis=1).max(axis=1)
        self.df['atr_14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        return self

    def generate_chart(self):
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.45, 0.18, 0.18, 0.19],
            subplot_titles=(f"{self.ticker} K線 / 均線群 / 布林通道", "RSI (14) 強弱動能", "MACD 指標", "成交量與 5MA 均量")
        )
        # Row 1
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['open'], high=self.df['high'], low=self.df['low'], close=self.df['close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['bb_upper'], line=dict(color='rgba(150,150,200,0.4)', width=1), name="布林上軌"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['bb_lower'], fill='tonexty', fillcolor='rgba(200,220,255,0.12)', line=dict(color='rgba(150,150,200,0.4)', width=1), name="布林下軌"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['ma_5'], line=dict(color='#FFA500', width=1.1), name="MA5"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['ma_20'], line=dict(color='#990099', width=1.3), name="MA20(月線)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['ma_60'], line=dict(color='#0066CC', width=1.2), name="MA60(季線)"), row=1, col=1)

        # Row 2
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['rsi_14'], line=dict(color='#636EFA', width=1.3), name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)

        # Row 3
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['macd_dif'], line=dict(color='#FF6600', width=1.1), name="DIF"), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['macd_dem'], line=dict(color='#0088FF', width=1.1), name="DEM"), row=3, col=1)
        m_colors = np.where(self.df['macd_hist'] >= 0, '#EF553B', '#00CC96')
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd_hist'], marker_color=m_colors, name="MACD柱狀體"), row=3, col=1)

        # Row 4
        v_colors = np.where(self.df['close'] >= self.df['open'], '#EF553B', '#00CC96')
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['volume'], marker_color=v_colors, name="成交量"), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['vol_ma5'], line=dict(color='#E67E22', width=1.2), name="5日均量"), row=4, col=1)

        fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, hovermode="x unified", height=850, margin=dict(l=40, r=30, t=50, b=30))
        return fig

    def generate_report(self):
        last = self.df.iloc[-1]
        c = last['close']
        ma20 = last['ma_20']
        rsi = last['rsi_14']
        macd_h = last['macd_hist']
        atr = last['atr_14']
        bias = last['bias_20']
        stop_loss = c - 2 * atr

        matrix = pd.DataFrame({
            "量化分析維度": ["最新收盤價", "20日均線 (MA20)", "20日乖離率 (%)", "14日 RSI 強弱值", "MACD 柱狀體", "布林帶寬 BB Width (%)", "ATR(14) 波動度", "動態防守停損價 (2×ATR)"],
            "即時數值": [f"{c:.2f} 元", f"{ma20:.2f} 元", f"{bias:+.2f}%", f"{rsi:.1f}", f"{macd_h:+.2f}", f"{last['bb_width']:.2f}%", f"{atr:.2f} 元", f"{stop_loss:.2f} 元"],
            "多空狀態判定": [
                "基準參考",
                "站上月線 (偏多)" if c >= ma20 else "跌破月線 (偏空)",
                "正乖離偏高" if bias > 5 else ("負乖離過大" if bias < -5 else "常態區間"),
                "超買區 (>70)" if rsi > 70 else ("超賣區 (<30)" if rsi < 30 else "中性格局 (30-70)"),
                "多方擴張" if macd_h > 0 else "空方主導",
                "通道擴張" if last['bb_width'] > 15 else "通道收斂壓縮",
                "高波動" if atr > (c * 0.03) else "低波動整理",
                "跌破此價位強制出場"
            ]
        })
        summary = (
            f"🎯【{self.ticker} 多空透視診斷報告】\n"
            f"1. 價格與均線：收盤價 {c:.2f} 元，位於月線 ({ma20:.2f} 元) 之"
            f"{'上方，短多架構確立' if c >= ma20 else '下方，短線偏弱整理'}，乖離率為 {bias:+.2f}%。\n"
            f"2. 動能與擺盪：RSI(14) 為 {rsi:.1f}，MACD 柱狀體為 {macd_h:+.2f}。\n"
            f"3. 波動與風控：當前 ATR 為 {atr:.2f} 元，建議防守停損價位設為 【{stop_loss:.2f} 元】。"
        )
        return summary, matrix

# --- 3. 自選股雷達掃描 ---
def scan_watchlist(watchlist_str: str) -> pd.DataFrame:
    if not watchlist_str or not watchlist_str.strip():
        return pd.DataFrame({"提示": ["請輸入股票代號清單"]})
    tickers = [t.strip().upper() for t in watchlist_str.replace("，", ",").split(",") if t.strip()]
    results = []
    for t in tickers:
        sym = t if (t.endswith('.TW') or t.endswith('.TWO') or t.startswith('^')) else f"{t}.TW"
        try:
            raw = yf.download(sym, period="2mo", auto_adjust=False, progress=False)
            if raw.empty or len(raw) < 22:
                continue
            clean_df = _clean_yf_data(raw)
            c = clean_df['close'].iloc[-1]
            pct = ((c - clean_df['close'].iloc[-2]) / clean_df['close'].iloc[-2]) * 100
            ma20 = clean_df['close'].rolling(20).mean().iloc[-1]
            bias = ((c - ma20) / ma20) * 100

            delta = clean_df['close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            rs = np.where(loss.iloc[-1] == 0, 0, gain.iloc[-1] / loss.iloc[-1])
            rsi = 100 if loss.iloc[-1] == 0 else (100 - (100 / (1 + rs)))

            sentiment = "🟢 強勢偏多" if (c >= ma20 and rsi >= 50) else ("🟡 高檔震盪" if c >= ma20 else ("🟠 弱勢反彈" if rsi >= 50 else "🔴 空頭承壓"))
            results.append({
                "股票代號": sym, "最新收盤價": round(c, 2), "單日漲跌幅 (%)": f"{pct:+.2f}%",
                "MA20 (月線)": round(ma20, 2), "20日乖離率 (%)": f"{bias:+.2f}%", "RSI(14)": round(rsi, 1), "多空評定": sentiment
            })
        except Exception:
            continue
    return pd.DataFrame(results) if results else pd.DataFrame({"提示": ["無有效行情數據"]})

# --- 4. Gradio 介面包裝函式 ---
def ui_analyze(ticker, period):
    try:
        analyzer = StockAnalyzer(ticker, period).fetch_and_calculate()
        fig = analyzer.generate_chart()
        summary, matrix = analyzer.generate_report()
        return fig, summary, matrix
    except Exception as e:
        empty_fig = go.Figure().update_layout(title=f"⚠️ 圖表加載失敗: {str(e)}", template="plotly_white")
        return empty_fig, f"❌ 執行發生異常：{str(e)}", pd.DataFrame({"提示": ["無資料"], "狀態": [str(e)]})

# --- 5. UI 佈局構建 ---
def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="台股 AI 綜合量化儀表板") as app:
        gr.Markdown("# 📈 台股 AI 綜合量化量能觀測儀表板")
        gr.Markdown("整合 **4 層深度技術圖表、短中長均線矩陣、動能擺盪、ATR 動態風控與智慧自選雷達** 的模組化系統。")
        with gr.Tabs():
            with gr.TabItem("📊 單檔技術與模型透視"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ticker_in = gr.Textbox(label="股票代號", value="3231.TW", placeholder="例如: 3231, 2330")
                        period_in = gr.Dropdown(label="回溯週期", choices=["3mo", "6mo", "1y", "2y", "3y", "5y"], value="1y")
                        btn_analyze = gr.Button("🚀 啟動全方位量化透視", variant="primary")
                        matrix_out = gr.Dataframe(headers=["量化分析維度", "即時數值", "多空狀態判定"], interactive=False, wrap=True)
                    with gr.Column(scale=3):
                        summary_out = gr.Textbox(label="🤖 AI 量化多空解析總評", lines=4, interactive=False)
                        plot_out = gr.Plot(label="專業 4 層互動量價圖表")
                btn_analyze.click(fn=ui_analyze, inputs=[ticker_in, period_in], outputs=[plot_out, summary_out, matrix_out], concurrency_limit=5)

            with gr.TabItem("🎯 智慧自選股雷達"):
                with gr.Column():
                    watch_in = gr.Textbox(label="批次股票代號清單 (以逗號分隔)", value="2330, 2317, 2454, 2303, 2308, 3231, 2603")
                    btn_scan = gr.Button("⚡ 執行自選股全域雷達掃描", variant="primary")
                    radar_out = gr.Dataframe(headers=["股票代號", "最新收盤價", "單日漲跌幅 (%)", "MA20 (月線)", "20日乖離率 (%)", "RSI(14)", "多空評定"], interactive=False, wrap=True)
                btn_scan.click(fn=scan_watchlist, inputs=[watch_in], outputs=[radar_out], concurrency_limit=3)
    return app

if __name__ == "__main__":
    demo = create_app()
    demo.queue().launch()
