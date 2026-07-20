import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="글로벌 에너지 Top 10 분석",
    page_icon="⚡",
    layout="wide",
)


# 글로벌 대형 에너지 기업 10개 대표군입니다.
# 시가총액 순위는 주가와 환율에 따라 계속 변할 수 있습니다.
ENERGY_TOP10 = {
    "Saudi Aramco": {
        "ticker": "2222.SR",
        "country": "사우디아라비아",
        "segment": "통합 석유·가스",
    },
    "Exxon Mobil": {
        "ticker": "XOM",
        "country": "미국",
        "segment": "통합 석유·가스",
    },
    "Chevron": {
        "ticker": "CVX",
        "country": "미국",
        "segment": "통합 석유·가스",
    },
    "PetroChina": {
        "ticker": "0857.HK",
        "country": "중국",
        "segment": "통합 석유·가스",
    },
    "Shell": {
        "ticker": "SHEL",
        "country": "영국",
        "segment": "통합 석유·가스",
    },
    "TotalEnergies": {
        "ticker": "TTE",
        "country": "프랑스",
        "segment": "통합 에너지",
    },
    "ConocoPhillips": {
        "ticker": "COP",
        "country": "미국",
        "segment": "탐사·생산",
    },
    "CNOOC": {
        "ticker": "0883.HK",
        "country": "중국",
        "segment": "탐사·생산",
    },
    "Petrobras": {
        "ticker": "PBR",
        "country": "브라질",
        "segment": "통합 석유·가스",
    },
    "BP": {
        "ticker": "BP",
        "country": "영국",
        "segment": "통합 에너지",
    },
}

MARKET_DRIVERS = {
    "WTI 원유": "CL=F",
    "브렌트유": "BZ=F",
    "천연가스": "NG=F",
    "에너지 ETF(XLE)": "XLE",
    "S&P 500": "^GSPC",
}

PERIODS = {
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "3년": "3y",
    "5년": "5y",
}

SEGMENT_COLORS = {
    "통합 석유·가스": "#0f766e",
    "통합 에너지": "#2563eb",
    "탐사·생산": "#ea580c",
    "직접 입력": "#64748b",
}


CATALOG = pd.DataFrame(
    [
        {
            "label": f"{company} · {details['ticker']} ({details['country']})",
            "company": company,
            **details,
        }
        for company, details in ENERGY_TOP10.items()
    ]
)
LABEL_TO_TICKER = dict(zip(CATALOG["label"], CATALOG["ticker"]))
TICKER_TO_NAME = dict(zip(CATALOG["ticker"], CATALOG["company"]))
TICKER_TO_COUNTRY = dict(zip(CATALOG["ticker"], CATALOG["country"]))
TICKER_TO_SEGMENT = dict(zip(CATALOG["ticker"], CATALOG["segment"]))


def company_name(ticker):
    return TICKER_TO_NAME.get(ticker, ticker)


def company_country(ticker):
    return TICKER_TO_COUNTRY.get(ticker, "직접 입력")


def company_segment(ticker):
    return TICKER_TO_SEGMENT.get(ticker, "직접 입력")


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, period):
    """Yahoo Finance OHLCV 데이터를 종목별 DataFrame으로 정리합니다."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def download_company_info(ticker):
    try:
        return yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}


def price_series(frame):
    if "Adj Close" in frame.columns and frame["Adj Close"].notna().any():
        return frame["Adj Close"].dropna()
    return frame["Close"].dropna()


def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, math.nan)
    return 100 - 100 / (1 + relative_strength)


def add_indicators(frame):
    result = frame.copy()
    close = result["Close"]
    result["SMA20"] = close.rolling(20).mean()
    result["SMA50"] = close.rolling(50).mean()
    result["SMA200"] = close.rolling(200).mean()
    result["EMA12"] = close.ewm(span=12, adjust=False).mean()
    result["EMA26"] = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = result["EMA12"] - result["EMA26"]
    result["MACD_SIGNAL"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["RSI14"] = calculate_rsi(close)
    standard_deviation = close.rolling(20).std()
    result["BB_UPPER"] = result["SMA20"] + 2 * standard_deviation
    result["BB_LOWER"] = result["SMA20"] - 2 * standard_deviation
    return result


def period_return(close, trading_days):
    if len(close) <= trading_days:
        return math.nan
    reference = float(close.iloc[-trading_days - 1])
    if reference == 0:
        return math.nan
    return (float(close.iloc[-1]) / reference - 1) * 100


def ytd_return(close):
    current_year = close.index[-1].year
    current_year_data = close[close.index.year == current_year]
    if current_year_data.empty or float(current_year_data.iloc[0]) == 0:
        return math.nan
    return (float(current_year_data.iloc[-1]) / float(current_year_data.iloc[0]) - 1) * 100


def maximum_drawdown(close):
    drawdown = close / close.cummax() - 1
    return float(drawdown.min() * 100)


def build_summary(price_data):
    records = []
    for ticker, frame in price_data.items():
        close = price_series(frame)
        if close.empty:
            continue
        daily_returns = close.pct_change().dropna()
        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if len(close) > 1 else latest
        first = float(close.iloc[0])
        records.append(
            {
                "기업": company_name(ticker),
                "티커": ticker,
                "국가": company_country(ticker),
                "사업구조": company_segment(ticker),
                "최근 종가": latest,
                "1일(%)": (latest / previous - 1) * 100 if previous else 0,
                "1개월(%)": period_return(close, 21),
                "3개월(%)": period_return(close, 63),
                "YTD(%)": ytd_return(close),
                "조회기간(%)": (latest / first - 1) * 100 if first else math.nan,
                "연환산 변동성(%)": daily_returns.std() * math.sqrt(252) * 100,
                "최대낙폭(%)": maximum_drawdown(close),
            }
        )
    return pd.DataFrame(records)


def normalized_prices(price_data, use_company_names=True):
    result = {}
    for ticker, frame in price_data.items():
        close = price_series(frame)
        if close.empty or float(close.iloc[0]) == 0:
            continue
        label = f"{company_name(ticker)} ({ticker})" if use_company_names else ticker
        result[label] = close / float(close.iloc[0]) * 100
    return pd.DataFrame(result)


def returns_frame(price_data, use_company_names=True):
    result = {}
    for ticker, frame in price_data.items():
        close = price_series(frame)
        label = company_name(ticker) if use_company_names else ticker
        result[label] = close.pct_change()
    return pd.DataFrame(result).dropna(how="all")


def format_number(value, digits=2):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def compact_number(value, currency=""):
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    for threshold, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(value) >= threshold:
            return f"{currency}{value / threshold:,.2f}{suffix}"
    return f"{currency}{value:,.2f}"


def technical_assessment(frame):
    technical = add_indicators(frame).dropna(subset=["Close"])
    latest = technical.iloc[-1]
    score = 0
    evidence = []

    for column, label in [("SMA20", "20일선"), ("SMA50", "50일선"), ("SMA200", "200일선")]:
        if pd.notna(latest[column]):
            if latest["Close"] > latest[column]:
                score += 1
                evidence.append(f"종가가 {label} 위")
            else:
                score -= 1
                evidence.append(f"종가가 {label} 아래")

    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_SIGNAL"]):
        if latest["MACD"] > latest["MACD_SIGNAL"]:
            score += 1
            evidence.append("MACD가 시그널선 위")
        else:
            score -= 1
            evidence.append("MACD가 시그널선 아래")

    rsi = latest["RSI14"]
    if pd.notna(rsi):
        if rsi >= 70:
            score -= 1
            evidence.append(f"RSI {rsi:.1f}: 과매수 구간")
        elif rsi <= 30:
            evidence.append(f"RSI {rsi:.1f}: 과매도 구간")
        elif rsi >= 50:
            score += 1
            evidence.append(f"RSI {rsi:.1f}: 양의 모멘텀")
        else:
            score -= 1
            evidence.append(f"RSI {rsi:.1f}: 약한 모멘텀")

    if score >= 4:
        label, color = "강한 상승 추세", "#16a34a"
    elif score >= 2:
        label, color = "상승 우위", "#65a30d"
    elif score >= -1:
        label, color = "중립·혼조", "#d97706"
    elif score >= -3:
        label, color = "하락 우위", "#ea580c"
    else:
        label, color = "강한 하락 추세", "#dc2626"
    return score, label, color, evidence, technical


st.title("⚡ 글로벌 에너지 Top 10 전문 분석")
st.caption(
    "대형 통합 에너지·석유·가스 기업의 가격, 위험, 원자재 민감도와 기업가치를 비교합니다. "
    f"조회 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

with st.sidebar:
    st.header("에너지 분석 설정")
    default_labels = CATALOG["label"].tolist()
    selected_labels = st.multiselect(
        "글로벌 대형 에너지 종목",
        options=default_labels,
        default=default_labels,
        help="전체 10개가 기본 선택됩니다.",
    )
    custom_text = st.text_input(
        "추가 티커 직접 입력",
        placeholder="예: OXY, SLB, NEE",
        help="여러 티커는 쉼표로 구분합니다.",
    )
    period_label = st.selectbox("분석 기간", list(PERIODS.keys()), index=2)
    theme = st.selectbox("차트 테마", ["plotly_white", "plotly_dark"])
    st.divider()
    st.caption("국가별 상장 시장과 표시 통화가 다르므로 절대 주가를 직접 비교하지 마세요.")

selected_tickers = [LABEL_TO_TICKER[label] for label in selected_labels]
custom_tickers = [item.strip().upper() for item in custom_text.split(",") if item.strip()]
tickers = list(dict.fromkeys(selected_tickers + custom_tickers))

if not tickers:
    st.info("왼쪽에서 분석할 종목을 한 개 이상 선택하세요.")
    st.stop()
if len(tickers) > 20:
    st.warning("조회 안정성을 위해 처음 20개 티커만 사용합니다.")
    tickers = tickers[:20]

all_download_tickers = list(dict.fromkeys(tickers + list(MARKET_DRIVERS.values())))
with st.spinner("에너지 기업과 원자재 시장 데이터를 불러오는 중입니다..."):
    try:
        all_prices = download_prices(tuple(all_download_tickers), PERIODS[period_label])
    except Exception as error:
        st.error(f"Yahoo Finance 데이터를 불러오지 못했습니다: {error}")
        st.stop()

stock_prices = {ticker: all_prices[ticker] for ticker in tickers if ticker in all_prices}
driver_prices = {
    name: all_prices[ticker]
    for name, ticker in MARKET_DRIVERS.items()
    if ticker in all_prices
}
missing = [ticker for ticker in tickers if ticker not in stock_prices]
if missing:
    st.warning("데이터를 확인할 수 없는 티커: " + ", ".join(missing))
if not stock_prices:
    st.error("분석 가능한 종목 데이터가 없습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

summary = build_summary(stock_prices)
normalized = normalized_prices(stock_prices)
stock_returns = returns_frame(stock_prices)

best_row = summary.loc[summary["조회기간(%)"].idxmax()]
lowest_volatility_row = summary.loc[summary["연환산 변동성(%)"].idxmin()]
average_return = summary["조회기간(%)"].mean()
average_drawdown = summary["최대낙폭(%)"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("분석 종목", f"{len(summary)}개")
k2.metric("기간 수익률 1위", best_row["기업"], f"{best_row['조회기간(%)']:+.2f}%")
k3.metric("평균 기간 수익률", f"{average_return:+.2f}%")
k4.metric(
    "최저 변동성",
    lowest_volatility_row["기업"],
    f"연 {lowest_volatility_row['연환산 변동성(%)']:.1f}%",
    delta_color="off",
)

overview_tab, technical_tab, commodity_tab, fundamental_tab, risk_tab, data_tab = st.tabs(
    [
        "🌍 Top 10 개요",
        "📈 기술적 분석",
        "🛢️ 원자재 민감도",
        "🏢 기업·배당",
        "⚖️ 위험·상관관계",
        "📋 전체 데이터",
    ]
)

with overview_tab:
    st.subheader(f"상대 주가 성과 · 시작값 100 ({period_label})")
    if not normalized.empty:
        performance_fig = px.line(
            normalized,
            x=normalized.index,
            y=normalized.columns,
            labels={"value": "지수화 가격", "variable": "기업", "Date": "날짜"},
            template=theme,
        )
        performance_fig.update_traces(line_width=2)
        performance_fig.update_layout(
            height=550,
            hovermode="x unified",
            legend_title_text="",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        performance_fig.add_hline(y=100, line_dash="dot", line_color="gray")
        st.plotly_chart(performance_fig, width="stretch")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("조회기간 수익률 순위")
        ranking = summary.sort_values("조회기간(%)")
        ranking_fig = px.bar(
            ranking,
            x="조회기간(%)",
            y="기업",
            orientation="h",
            color="사업구조",
            color_discrete_map=SEGMENT_COLORS,
            text="조회기간(%)",
            template=theme,
        )
        ranking_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        ranking_fig.update_layout(
            height=max(440, len(ranking) * 39),
            legend_title_text="",
            margin=dict(l=20, r=35, t=20, b=20),
        )
        ranking_fig.add_vline(x=0, line_color="gray")
        st.plotly_chart(ranking_fig, width="stretch")

    with right:
        st.subheader("수익률–위험 분포")
        risk_return_fig = px.scatter(
            summary,
            x="연환산 변동성(%)",
            y="조회기간(%)",
            color="사업구조",
            color_discrete_map=SEGMENT_COLORS,
            text="기업",
            hover_data=["티커", "국가", "최대낙폭(%)"],
            template=theme,
        )
        risk_return_fig.update_traces(textposition="top center", marker=dict(size=12))
        risk_return_fig.update_layout(
            height=520,
            legend_title_text="",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(risk_return_fig, width="stretch")
        st.caption("오른쪽일수록 변동성이 높고, 위쪽일수록 조회기간 수익률이 높습니다.")

with technical_tab:
    technical_ticker = st.selectbox(
        "기술적 분석 기업",
        options=list(stock_prices.keys()),
        format_func=lambda ticker: f"{company_name(ticker)} ({ticker})",
        key="energy_technical_ticker",
    )
    score, signal, signal_color, evidence, technical = technical_assessment(
        stock_prices[technical_ticker]
    )
    close = price_series(stock_prices[technical_ticker])
    last = technical.iloc[-1]
    daily_change = period_return(close, 1)
    annual_volatility = close.pct_change().std() * math.sqrt(252) * 100

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("최근 종가", format_number(close.iloc[-1]), f"{daily_change:+.2f}%")
    t2.metric("RSI(14)", format_number(last["RSI14"], 1))
    t3.metric("연환산 변동성", f"{annual_volatility:.1f}%")
    t4.metric("최대낙폭", f"{maximum_drawdown(close):.1f}%")
    t5.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{signal_color}18;"
        f"border:1px solid {signal_color};height:82px'>"
        f"<div style='font-size:13px'>규칙 기반 추세 판정</div>"
        f"<div style='font-size:18px;font-weight:700;color:{signal_color}'>{signal}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    technical_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.19, 0.19],
    )
    technical_fig.add_trace(
        go.Candlestick(
            x=technical.index,
            open=technical["Open"],
            high=technical["High"],
            low=technical["Low"],
            close=technical["Close"],
            name="OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )
    for column, color, label in [
        ("SMA20", "#f59e0b", "20일선"),
        ("SMA50", "#2563eb", "50일선"),
        ("SMA200", "#7c3aed", "200일선"),
    ]:
        technical_fig.add_trace(
            go.Scatter(
                x=technical.index,
                y=technical[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=1.3),
            ),
            row=1,
            col=1,
        )
    technical_fig.add_trace(
        go.Scatter(
            x=technical.index,
            y=technical["RSI14"],
            name="RSI(14)",
            line=dict(color="#0f766e"),
        ),
        row=2,
        col=1,
    )
    technical_fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=2, col=1)
    technical_fig.add_hline(y=30, line_dash="dot", line_color="#2563eb", row=2, col=1)
    technical_fig.add_trace(
        go.Scatter(
            x=technical.index,
            y=technical["MACD"],
            name="MACD",
            line=dict(color="#7c3aed"),
        ),
        row=3,
        col=1,
    )
    technical_fig.add_trace(
        go.Scatter(
            x=technical.index,
            y=technical["MACD_SIGNAL"],
            name="Signal",
            line=dict(color="#f59e0b"),
        ),
        row=3,
        col=1,
    )
    technical_fig.update_layout(
        title=f"{company_name(technical_ticker)} ({technical_ticker}) · 가격 / RSI / MACD",
        template=theme,
        height=820,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=20, r=20, t=75, b=20),
    )
    technical_fig.update_yaxes(title_text="가격", row=1, col=1)
    technical_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    technical_fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(technical_fig, width="stretch")

    with st.expander("추세 판정 근거", expanded=True):
        for item in evidence:
            st.write("• " + item)
        st.caption(
            f"종합 점수 {score:+d}. 이동평균·RSI·MACD 기반의 과거 추세 요약이며 "
            "미래 가격 예측이나 매수·매도 신호가 아닙니다."
        )

with commodity_tab:
    st.subheader("원유·천연가스·시장지수와 상대 성과")
    combined_series = {}
    for name, frame in driver_prices.items():
        close = price_series(frame)
        if not close.empty and float(close.iloc[0]) != 0:
            combined_series[name] = close / float(close.iloc[0]) * 100

    commodity_ticker = st.selectbox(
        "비교할 에너지 기업",
        options=list(stock_prices.keys()),
        format_func=lambda ticker: f"{company_name(ticker)} ({ticker})",
        key="commodity_ticker",
    )
    selected_close = price_series(stock_prices[commodity_ticker])
    combined_series[f"{company_name(commodity_ticker)} ({commodity_ticker})"] = (
        selected_close / float(selected_close.iloc[0]) * 100
    )
    combined = pd.DataFrame(combined_series)
    driver_fig = px.line(
        combined,
        x=combined.index,
        y=combined.columns,
        labels={"value": "지수화 가격", "variable": "자산", "Date": "날짜"},
        template=theme,
    )
    driver_fig.update_traces(line_width=2)
    driver_fig.update_layout(
        height=560,
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    driver_fig.add_hline(y=100, line_dash="dot", line_color="gray")
    st.plotly_chart(driver_fig, width="stretch")

    selected_return = selected_close.pct_change().rename("기업")
    sensitivity_rows = []
    for driver_name, frame in driver_prices.items():
        driver_return = price_series(frame).pct_change().rename("시장요인")
        aligned = pd.concat([selected_return, driver_return], axis=1).dropna()
        correlation = aligned["기업"].corr(aligned["시장요인"]) if len(aligned) > 2 else math.nan
        variance = aligned["시장요인"].var()
        beta = (
            aligned["기업"].cov(aligned["시장요인"]) / variance
            if len(aligned) > 2 and variance != 0
            else math.nan
        )
        sensitivity_rows.append(
            {
                "시장요인": driver_name,
                "상관계수": correlation,
                "민감도(베타)": beta,
                "관측일수": len(aligned),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    st.dataframe(
        sensitivity,
        width="stretch",
        hide_index=True,
        column_config={
            "상관계수": st.column_config.NumberColumn(format="%.3f"),
            "민감도(베타)": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "상관계수는 함께 움직인 정도를, 베타는 시장요인 수익률이 1% 변할 때 기업 수익률이 "
        "통계적으로 얼마나 움직였는지를 나타냅니다. 인과관계를 의미하지 않습니다."
    )

with fundamental_tab:
    fundamental_ticker = st.selectbox(
        "기업·배당 정보 조회",
        options=list(stock_prices.keys()),
        format_func=lambda ticker: f"{company_name(ticker)} ({ticker})",
        key="energy_fundamental_ticker",
    )
    with st.spinner("기업 재무 정보를 불러오는 중입니다..."):
        info = download_company_info(fundamental_ticker)

    if not info:
        st.warning("Yahoo Finance에서 이 기업의 상세 정보를 제공하지 않았습니다.")
    else:
        full_name = info.get("longName") or company_name(fundamental_ticker)
        st.subheader(f"{full_name} ({fundamental_ticker})")
        description_items = [
            info.get("sector"),
            info.get("industry"),
            info.get("country"),
            info.get("exchange"),
            info.get("currency"),
        ]
        st.caption(" · ".join(item for item in description_items if item))
        currency_code = info.get("currency", "")
        currency_symbol = {"USD": "$", "KRW": "₩", "EUR": "€", "GBP": "£"}.get(
            currency_code, ""
        )

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("시가총액", compact_number(info.get("marketCap"), currency_symbol))
        f2.metric("후행 PER", format_number(info.get("trailingPE")))
        f3.metric("선행 PER", format_number(info.get("forwardPE")))
        f4.metric("배당수익률", format_percent(info.get("dividendYield")))

        f5, f6, f7, f8 = st.columns(4)
        f5.metric("매출 성장률", format_percent(info.get("revenueGrowth")))
        f6.metric("영업이익률", format_percent(info.get("operatingMargins")))
        f7.metric("ROE", format_percent(info.get("returnOnEquity")))
        f8.metric("부채비율", format_number(info.get("debtToEquity")))

        facts = pd.DataFrame(
            [
                ["기업가치(EV)", compact_number(info.get("enterpriseValue"), currency_symbol)],
                ["EV/EBITDA", format_number(info.get("enterpriseToEbitda"))],
                ["주가매출비율(P/S)", format_number(info.get("priceToSalesTrailing12Months"))],
                ["주가순자산비율(P/B)", format_number(info.get("priceToBook"))],
                ["배당성향", format_percent(info.get("payoutRatio"))],
                ["잉여현금흐름", compact_number(info.get("freeCashflow"), currency_symbol)],
                ["영업현금흐름", compact_number(info.get("operatingCashflow"), currency_symbol)],
                ["총현금", compact_number(info.get("totalCash"), currency_symbol)],
                ["총부채", compact_number(info.get("totalDebt"), currency_symbol)],
                ["52주 최고가", format_number(info.get("fiftyTwoWeekHigh"))],
                ["52주 최저가", format_number(info.get("fiftyTwoWeekLow"))],
                ["목표주가 평균", format_number(info.get("targetMeanPrice"))],
            ],
            columns=["지표", "값"],
        )
        st.dataframe(facts, width="stretch", hide_index=True)

        st.info(
            "에너지 기업은 유가·가스 가격뿐 아니라 생산량, 매장량, 정제마진, 설비투자, 부채, "
            "배당 지속가능성과 정부 정책의 영향을 받습니다. PER와 배당수익률만으로 판단하지 마세요."
        )

with risk_tab:
    st.subheader("에너지 기업 일간 수익률 상관관계")
    if stock_returns.shape[1] >= 2:
        correlation = stock_returns.corr()
        correlation_fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            template=theme,
        )
        correlation_fig.update_layout(height=650, coloraxis_colorbar_title="상관계수")
        st.plotly_chart(correlation_fig, width="stretch")
    else:
        st.info("상관관계 분석에는 두 개 이상의 기업이 필요합니다.")

    st.subheader("XLE 대비 베타와 위험지표")
    xle_frame = driver_prices.get("에너지 ETF(XLE)")
    xle_return = price_series(xle_frame).pct_change().rename("XLE") if xle_frame is not None else None
    risk_records = []
    for ticker, frame in stock_prices.items():
        close = price_series(frame)
        stock_return = close.pct_change().rename("Stock")
        beta = math.nan
        if xle_return is not None:
            aligned = pd.concat([stock_return, xle_return], axis=1).dropna()
            variance = aligned["XLE"].var()
            if len(aligned) > 2 and variance != 0:
                beta = aligned["Stock"].cov(aligned["XLE"]) / variance
        risk_records.append(
            {
                "기업": company_name(ticker),
                "티커": ticker,
                "XLE 베타": beta,
                "연환산 변동성(%)": stock_return.std() * math.sqrt(252) * 100,
                "최대낙폭(%)": maximum_drawdown(close),
                "상승일 비율(%)": (stock_return.dropna() > 0).mean() * 100,
            }
        )
    risk_table = pd.DataFrame(risk_records)
    st.dataframe(
        risk_table,
        width="stretch",
        hide_index=True,
        column_config={
            "XLE 베타": st.column_config.NumberColumn(format="%.2f"),
            "연환산 변동성(%)": st.column_config.NumberColumn(format="%.2f"),
            "최대낙폭(%)": st.column_config.NumberColumn(format="%.2f"),
            "상승일 비율(%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

with data_tab:
    st.subheader("글로벌 에너지 Top 10 성과표")
    st.dataframe(
        summary.sort_values("조회기간(%)", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "최근 종가": st.column_config.NumberColumn(format="%.2f"),
            "1일(%)": st.column_config.NumberColumn(format="%+.2f"),
            "1개월(%)": st.column_config.NumberColumn(format="%+.2f"),
            "3개월(%)": st.column_config.NumberColumn(format="%+.2f"),
            "YTD(%)": st.column_config.NumberColumn(format="%+.2f"),
            "조회기간(%)": st.column_config.NumberColumn(format="%+.2f"),
            "연환산 변동성(%)": st.column_config.NumberColumn(format="%.2f"),
            "최대낙폭(%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    export_frames = []
    for ticker, frame in stock_prices.items():
        exported = frame.copy().reset_index()
        exported.insert(0, "Ticker", ticker)
        exported.insert(1, "Company", company_name(ticker))
        exported.insert(2, "Country", company_country(ticker))
        exported.insert(3, "Segment", company_segment(ticker))
        export_frames.append(exported)
    export_data = pd.concat(export_frames, ignore_index=True)
    st.download_button(
        "에너지 주식 원자료 CSV 다운로드",
        data=export_data.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"global_energy_top10_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch",
    )

st.divider()
st.caption(
    "본 페이지는 정보·교육용 분석 도구이며 투자 권유가 아닙니다. 에너지 기업 순위는 주가와 환율에 따라 "
    "변동될 수 있고, Yahoo Finance 데이터는 지연·수정·누락될 수 있습니다."
)
