from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="글로벌 주요 주식 대시보드",
    page_icon="🌍",
    layout="wide",
)


STOCKS = {
    "미국": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Alphabet": "GOOGL",
        "Amazon": "AMZN",
        "Meta": "META",
        "Tesla": "TSLA",
        "JPMorgan": "JPM",
    },
    "한국": {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "현대차": "005380.KS",
        "NAVER": "035420.KS",
        "셀트리온": "068270.KS",
    },
    "유럽": {
        "ASML": "ASML.AS",
        "SAP": "SAP.DE",
        "Nestlé": "NESN.SW",
        "LVMH": "MC.PA",
        "Novo Nordisk": "NOVO-B.CO",
    },
    "일본": {
        "Toyota": "7203.T",
        "Sony": "6758.T",
        "SoftBank Group": "9984.T",
        "Nintendo": "7974.T",
    },
    "중국·홍콩": {
        "Tencent": "0700.HK",
        "Alibaba": "9988.HK",
        "BYD": "1211.HK",
    },
    "인도": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy": "TCS.NS",
        "Infosys": "INFY.NS",
    },
}

PERIODS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "3년": "3y",
    "5년": "5y",
    "최대": "max",
}


def build_catalog():
    rows = []
    for region, companies in STOCKS.items():
        for company, ticker in companies.items():
            rows.append(
                {
                    "label": f"{company} · {ticker} ({region})",
                    "company": company,
                    "ticker": ticker,
                    "region": region,
                }
            )
    return pd.DataFrame(rows)


CATALOG = build_catalog()
LABEL_TO_TICKER = dict(zip(CATALOG["label"], CATALOG["ticker"]))
TICKER_TO_NAME = dict(zip(CATALOG["ticker"], CATALOG["company"]))
TICKER_TO_REGION = dict(zip(CATALOG["ticker"], CATALOG["region"]))


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, period):
    """여러 종목의 OHLCV 데이터를 내려받아 종목별 DataFrame으로 반환."""
    tickers = tuple(dict.fromkeys(tickers))
    if not tickers:
        return {}

    raw = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        timeout=20,
    )

    result = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level_0 = raw.columns.get_level_values(0)
                level_1 = raw.columns.get_level_values(1)
                if ticker in level_0:
                    frame = raw[ticker].copy()
                elif ticker in level_1:
                    frame = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
            else:
                frame = raw.copy()

            frame.columns = [str(column) for column in frame.columns]
            frame = frame.dropna(how="all")
            if "Close" in frame.columns:
                frame = frame.dropna(subset=["Close"])
            if not frame.empty:
                result[ticker] = frame
        except (KeyError, TypeError, ValueError):
            continue
    return result


def format_price(value):
    if pd.isna(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def company_name(ticker):
    return TICKER_TO_NAME.get(ticker, ticker)


def make_summary(data):
    rows = []
    for ticker, frame in data.items():
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if close.empty:
            continue
        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if len(close) > 1 else latest
        first = float(close.iloc[0])
        daily_change = (latest / previous - 1) * 100 if previous else 0.0
        period_change = (latest / first - 1) * 100 if first else 0.0
        volume = frame["Volume"].iloc[-1] if "Volume" in frame else None
        rows.append(
            {
                "종목": company_name(ticker),
                "티커": ticker,
                "지역": TICKER_TO_REGION.get(ticker, "직접 입력"),
                "최근 종가": latest,
                "전일 대비(%)": daily_change,
                "기간 수익률(%)": period_change,
                "최근 거래량": volume,
                "최근 거래일": close.index[-1].strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def normalized_frame(data):
    normalized = {}
    for ticker, frame in data.items():
        close = frame.get("Adj Close", frame.get("Close"))
        if close is None:
            continue
        close = close.dropna()
        if not close.empty and float(close.iloc[0]) != 0:
            normalized[company_name(ticker)] = close / float(close.iloc[0]) * 100
    return pd.DataFrame(normalized)


def raw_download_frame(data):
    frames = []
    for ticker, frame in data.items():
        exported = frame.copy().reset_index()
        exported.insert(0, "Ticker", ticker)
        exported.insert(1, "Company", company_name(ticker))
        frames.append(exported)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


st.title("🌍 글로벌 주요 주식 대시보드")
st.caption(
    "Yahoo Finance 시장 데이터를 활용한 글로벌 대표 종목 비교 · "
    f"조회 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

with st.sidebar:
    st.header("조회 설정")
    regions = st.multiselect(
        "시장 선택",
        options=list(STOCKS.keys()),
        default=["미국", "한국", "유럽", "일본"],
    )
    available = CATALOG[CATALOG["region"].isin(regions)]
    default_tickers = ["AAPL", "MSFT", "NVDA", "005930.KS", "000660.KS", "ASML.AS", "7203.T"]
    default_labels = [
        label
        for label, ticker in LABEL_TO_TICKER.items()
        if ticker in default_tickers and label in set(available["label"])
    ]
    selected_labels = st.multiselect(
        "종목 선택 (최대 12개 권장)",
        options=available["label"].tolist(),
        default=default_labels,
    )
    custom_text = st.text_input(
        "티커 직접 입력",
        placeholder="예: AMD, BRK-B, 035720.KS",
        help="여러 종목은 쉼표로 구분하세요.",
    )
    period_label = st.selectbox("조회 기간", options=list(PERIODS.keys()), index=3)
    chart_theme = st.selectbox("차트 테마", ["plotly_white", "plotly_dark"], index=0)
    st.divider()
    st.caption("데이터는 거래소 상황과 Yahoo Finance 정책에 따라 지연되거나 일부 누락될 수 있습니다.")

selected_tickers = [LABEL_TO_TICKER[label] for label in selected_labels]
custom_tickers = [item.strip().upper() for item in custom_text.split(",") if item.strip()]
tickers = list(dict.fromkeys(selected_tickers + custom_tickers))

if not tickers:
    st.info("왼쪽에서 시장과 종목을 한 개 이상 선택해 주세요.")
    st.stop()

if len(tickers) > 20:
    st.warning("원활한 조회를 위해 처음 20개 종목만 표시합니다.")
    tickers = tickers[:20]

with st.spinner("Yahoo Finance에서 주가 데이터를 불러오는 중입니다..."):
    try:
        price_data = download_prices(tuple(tickers), PERIODS[period_label])
    except Exception as error:
        st.error(f"데이터를 불러오지 못했습니다: {error}")
        st.stop()

missing = [ticker for ticker in tickers if ticker not in price_data]
if missing:
    st.warning("데이터를 찾지 못한 티커: " + ", ".join(missing))

if not price_data:
    st.error("표시할 주가 데이터가 없습니다. 티커를 확인한 뒤 다시 시도해 주세요.")
    st.stop()

summary = make_summary(price_data)
normalized = normalized_frame(price_data)

st.subheader("시장 한눈에 보기")
metric_columns = st.columns(min(len(summary), 4))
for index, row in summary.head(8).iterrows():
    column = metric_columns[index % len(metric_columns)]
    column.metric(
        label=f"{row['종목']} ({row['티커']})",
        value=format_price(row["최근 종가"]),
        delta=f"{row['전일 대비(%)']:+.2f}%",
    )

overview_tab, detail_tab, table_tab = st.tabs(
    ["📈 수익률 비교", "🕯️ 종목 상세", "📋 데이터 표"]
)

with overview_tab:
    st.subheader(f"기간 수익률 비교 · 시작값 = 100 ({period_label})")
    if not normalized.empty:
        fig = px.line(
            normalized,
            x=normalized.index,
            y=normalized.columns,
            labels={"value": "지수화 가격", "variable": "종목", "Date": "날짜"},
            template=chart_theme,
        )
        fig.update_traces(line_width=2)
        fig.update_layout(
            height=520,
            hovermode="x unified",
            legend_title_text="",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    ranking = summary.sort_values("기간 수익률(%)", ascending=True)
    ranking_fig = px.bar(
        ranking,
        x="기간 수익률(%)",
        y="종목",
        orientation="h",
        color="기간 수익률(%)",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text_auto=".2f",
        template=chart_theme,
    )
    ranking_fig.update_layout(
        height=max(350, len(ranking) * 42),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(ranking_fig, use_container_width=True)

with detail_tab:
    detail_ticker = st.selectbox(
        "상세 종목",
        options=list(price_data.keys()),
        format_func=lambda ticker: f"{company_name(ticker)} ({ticker})",
    )
    chart_type = st.radio(
        "차트 유형", ["캔들차트", "종가 라인"], horizontal=True
    )
    detail = price_data[detail_ticker].copy()

    if chart_type == "캔들차트" and all(
        column in detail.columns for column in ["Open", "High", "Low", "Close"]
    ):
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Candlestick(
                x=detail.index,
                open=detail["Open"],
                high=detail["High"],
                low=detail["Low"],
                close=detail["Close"],
                name="OHLC",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            ),
            row=1,
            col=1,
        )
        if "Volume" in detail.columns:
            colors = [
                "#16a34a" if close >= open_ else "#dc2626"
                for open_, close in zip(detail["Open"], detail["Close"])
            ]
            fig.add_trace(
                go.Bar(x=detail.index, y=detail["Volume"], marker_color=colors, name="거래량"),
                row=2,
                col=1,
            )
        fig.update_layout(
            title=f"{company_name(detail_ticker)} ({detail_ticker})",
            template=chart_theme,
            height=650,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=20, r=20, t=55, b=20),
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=detail.index,
                y=detail["Close"],
                mode="lines",
                name="종가",
                line=dict(color="#2563eb", width=2),
            )
        )
        for window, color in [(20, "#f59e0b"), (60, "#8b5cf6")]:
            if len(detail) >= window:
                fig.add_trace(
                    go.Scatter(
                        x=detail.index,
                        y=detail["Close"].rolling(window).mean(),
                        mode="lines",
                        name=f"{window}일 이동평균",
                        line=dict(color=color, width=1.5),
                    )
                )
        fig.update_layout(
            title=f"{company_name(detail_ticker)} ({detail_ticker})",
            template=chart_theme,
            height=560,
            hovermode="x unified",
            yaxis_title="가격",
            legend_title_text="",
            margin=dict(l=20, r=20, t=55, b=20),
        )
    st.plotly_chart(fig, use_container_width=True)

with table_tab:
    display_summary = summary.copy()
    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "최근 종가": st.column_config.NumberColumn(format="%.2f"),
            "전일 대비(%)": st.column_config.NumberColumn(format="%+.2f%%"),
            "기간 수익률(%)": st.column_config.NumberColumn(format="%+.2f%%"),
            "최근 거래량": st.column_config.NumberColumn(format="localized"),
        },
    )
    export_data = raw_download_frame(price_data)
    st.download_button(
        "원자료 CSV 다운로드",
        data=export_data.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"global_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "본 대시보드는 정보·교육 목적으로 제공되며 투자 권유가 아닙니다. "
    "yfinance는 Yahoo Finance와 공식적으로 제휴된 라이브러리가 아닙니다."
)
