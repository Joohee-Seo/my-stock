import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="AI 반도체 전문 분석",
    page_icon="🧠",
    layout="wide",
)


UNIVERSE = {
    "AI 가속기·GPU": {
        "NVIDIA": "NVDA",
        "AMD": "AMD",
        "Broadcom": "AVGO",
        "Arm Holdings": "ARM",
        "Intel": "INTC",
    },
    "파운드리·제조": {
        "TSMC": "TSM",
        "GlobalFoundries": "GFS",
        "UMC": "UMC",
    },
    "HBM·메모리": {
        "SK하이닉스": "000660.KS",
        "삼성전자": "005930.KS",
        "Micron": "MU",
    },
    "반도체 장비": {
        "ASML": "ASML",
        "Applied Materials": "AMAT",
        "Lam Research": "LRCX",
        "KLA": "KLAC",
    },
    "네트워킹·통신": {
        "Marvell": "MRVL",
        "Qualcomm": "QCOM",
        "Astera Labs": "ALAB",
    },
    "AI 반도체 ETF": {
        "VanEck Semiconductor ETF": "SMH",
        "iShares Semiconductor ETF": "SOXX",
    },
}

PERIODS = {
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "3년": "3y",
    "5년": "5y",
}

CATEGORY_COLORS = {
    "AI 가속기·GPU": "#7c3aed",
    "파운드리·제조": "#2563eb",
    "HBM·메모리": "#0891b2",
    "반도체 장비": "#ea580c",
    "네트워킹·통신": "#16a34a",
    "AI 반도체 ETF": "#64748b",
    "직접 입력": "#334155",
}


def make_catalog():
    records = []
    for category, companies in UNIVERSE.items():
        for company, ticker in companies.items():
            records.append(
                {
                    "label": f"{company} · {ticker}",
                    "company": company,
                    "ticker": ticker,
                    "category": category,
                }
            )
    return pd.DataFrame(records)


CATALOG = make_catalog()
LABEL_TO_TICKER = dict(zip(CATALOG["label"], CATALOG["ticker"]))
TICKER_TO_NAME = dict(zip(CATALOG["ticker"], CATALOG["company"]))
TICKER_TO_CATEGORY = dict(zip(CATALOG["ticker"], CATALOG["category"]))


def stock_name(ticker):
    return TICKER_TO_NAME.get(ticker, ticker)


def stock_category(ticker):
    return TICKER_TO_CATEGORY.get(ticker, "직접 입력")


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, period):
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


def adjusted_close(frame):
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
    return 100 - (100 / (1 + relative_strength))


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
    std20 = close.rolling(20).std()
    result["BB_UPPER"] = result["SMA20"] + 2 * std20
    result["BB_LOWER"] = result["SMA20"] - 2 * std20
    result["RETURN"] = adjusted_close(result).pct_change()
    return result


def return_for_days(close, days):
    if len(close) <= days or float(close.iloc[-days - 1]) == 0:
        return math.nan
    return (float(close.iloc[-1]) / float(close.iloc[-days - 1]) - 1) * 100


def ytd_return(close):
    current_year = close.index[-1].year
    ytd = close[close.index.year == current_year]
    if ytd.empty or float(ytd.iloc[0]) == 0:
        return math.nan
    return (float(ytd.iloc[-1]) / float(ytd.iloc[0]) - 1) * 100


def max_drawdown(close):
    running_high = close.cummax()
    drawdown = close / running_high - 1
    return float(drawdown.min() * 100)


def build_summary(price_data):
    records = []
    for ticker, frame in price_data.items():
        close = adjusted_close(frame)
        if close.empty:
            continue
        returns = close.pct_change().dropna()
        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if len(close) > 1 else latest
        period_return = (latest / float(close.iloc[0]) - 1) * 100
        annual_volatility = float(returns.std() * math.sqrt(252) * 100) if len(returns) > 1 else math.nan
        records.append(
            {
                "기업": stock_name(ticker),
                "티커": ticker,
                "밸류체인": stock_category(ticker),
                "종가": latest,
                "1일(%)": (latest / previous - 1) * 100 if previous else 0,
                "1개월(%)": return_for_days(close, 21),
                "3개월(%)": return_for_days(close, 63),
                "YTD(%)": ytd_return(close),
                "조회기간(%)": period_return,
                "연환산 변동성(%)": annual_volatility,
                "최대낙폭(%)": max_drawdown(close),
            }
        )
    return pd.DataFrame(records)


def normalized_prices(price_data):
    series = {}
    for ticker, frame in price_data.items():
        close = adjusted_close(frame)
        if not close.empty and float(close.iloc[0]) != 0:
            series[f"{stock_name(ticker)} ({ticker})"] = close / float(close.iloc[0]) * 100
    return pd.DataFrame(series)


def daily_returns(price_data):
    series = {}
    for ticker, frame in price_data.items():
        close = adjusted_close(frame)
        if not close.empty:
            series[stock_name(ticker)] = close.pct_change()
    return pd.DataFrame(series).dropna(how="all")


def format_number(value, digits=2):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def compact_number(value, currency=""):
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for threshold, suffix in units:
        if abs(value) >= threshold:
            return f"{currency}{value / threshold:,.2f}{suffix}"
    return f"{currency}{value:,.2f}"


def technical_signal(frame):
    data = add_indicators(frame).dropna(subset=["Close"])
    latest = data.iloc[-1]
    score = 0
    evidence = []

    if pd.notna(latest["SMA20"]):
        if latest["Close"] > latest["SMA20"]:
            score += 1
            evidence.append("종가가 20일 이동평균 위")
        else:
            score -= 1
            evidence.append("종가가 20일 이동평균 아래")

    if pd.notna(latest["SMA50"]):
        if latest["Close"] > latest["SMA50"]:
            score += 1
            evidence.append("종가가 50일 이동평균 위")
        else:
            score -= 1
            evidence.append("종가가 50일 이동평균 아래")

    if pd.notna(latest["SMA200"]):
        if latest["Close"] > latest["SMA200"]:
            score += 1
            evidence.append("장기 200일선 위")
        else:
            score -= 1
            evidence.append("장기 200일선 아래")

    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_SIGNAL"]):
        if latest["MACD"] > latest["MACD_SIGNAL"]:
            score += 1
            evidence.append("MACD가 시그널선 위")
        else:
            score -= 1
            evidence.append("MACD가 시그널선 아래")

    rsi = latest.get("RSI14", math.nan)
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
    return score, label, color, evidence, data


st.title("🧠 AI 반도체 주식 전문 분석")
st.caption(
    "AI 가속기부터 파운드리·HBM·장비·네트워킹까지 밸류체인별로 비교합니다. "
    f"조회 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

with st.sidebar:
    st.header("분석 설정")
    categories = st.multiselect(
        "AI 반도체 밸류체인",
        options=list(UNIVERSE.keys()),
        default=list(UNIVERSE.keys()),
    )
    available = CATALOG[CATALOG["category"].isin(categories)]
    default_tickers = ["NVDA", "AMD", "AVGO", "TSM", "000660.KS", "MU", "ASML", "AMAT", "SMH"]
    available_labels = available["label"].tolist()
    default_labels = [
        label for label in available_labels if LABEL_TO_TICKER[label] in default_tickers
    ]
    labels = st.multiselect(
        "비교 종목",
        options=available_labels,
        default=default_labels,
        help="그래프 가독성과 데이터 조회 속도를 위해 12개 이하를 권장합니다.",
    )
    custom_text = st.text_input(
        "Yahoo Finance 티커 직접 입력",
        placeholder="예: MCHP, 042700.KQ",
        help="여러 티커는 쉼표로 구분합니다.",
    )
    period_label = st.selectbox("분석 기간", list(PERIODS.keys()), index=2)
    theme = st.selectbox("차트 테마", ["plotly_white", "plotly_dark"])
    st.divider()
    st.caption("가격 데이터는 실시간이 아닐 수 있으며 거래소별 통화가 서로 다릅니다.")

selected = [LABEL_TO_TICKER[label] for label in labels]
custom = [item.strip().upper() for item in custom_text.split(",") if item.strip()]
tickers = list(dict.fromkeys(selected + custom))

if not tickers:
    st.info("왼쪽에서 분석할 종목을 한 개 이상 선택하세요.")
    st.stop()

if len(tickers) > 20:
    st.warning("조회 안정성을 위해 처음 20개 종목만 분석합니다.")
    tickers = tickers[:20]

with st.spinner("AI 반도체 종목 데이터를 불러오고 있습니다..."):
    try:
        prices = download_prices(tuple(tickers), PERIODS[period_label])
    except Exception as error:
        st.error(f"Yahoo Finance 데이터를 불러오지 못했습니다: {error}")
        st.stop()

missing = [ticker for ticker in tickers if ticker not in prices]
if missing:
    st.warning("데이터를 확인할 수 없는 티커: " + ", ".join(missing))
if not prices:
    st.error("분석할 수 있는 데이터가 없습니다. 티커를 확인해 주세요.")
    st.stop()

summary = build_summary(prices)
normalized = normalized_prices(prices)
returns = daily_returns(prices)

benchmark_ticker = "SMH" if "SMH" in prices else next(iter(prices))
benchmark_name = stock_name(benchmark_ticker)
best_row = summary.loc[summary["조회기간(%)"].idxmax()]
lowest_risk_row = summary.loc[summary["연환산 변동성(%)"].idxmin()]
average_return = summary["조회기간(%)"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("분석 종목", f"{len(summary)}개")
kpi2.metric("기간 수익률 1위", best_row["기업"], f"{best_row['조회기간(%)']:+.2f}%")
kpi3.metric("평균 기간 수익률", f"{average_return:+.2f}%")
kpi4.metric(
    "최저 변동성",
    lowest_risk_row["기업"],
    f"연 {lowest_risk_row['연환산 변동성(%)']:.1f}%",
    delta_color="off",
)

overview_tab, technical_tab, fundamental_tab, risk_tab, data_tab = st.tabs(
    ["🌐 밸류체인 개요", "📈 기술적 분석", "🏢 기업·밸류에이션", "⚖️ 위험·상관관계", "📋 전체 데이터"]
)

with overview_tab:
    st.subheader(f"상대 주가 성과 · 시작값 100 ({period_label})")
    if not normalized.empty:
        fig = px.line(
            normalized,
            x=normalized.index,
            y=normalized.columns,
            labels={"value": "지수화 가격", "variable": "종목", "Date": "날짜"},
            template=theme,
        )
        fig.update_traces(line_width=2)
        fig.update_layout(
            height=540,
            hovermode="x unified",
            legend_title_text="",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("기간 수익률 순위")
        ranking = summary.sort_values("조회기간(%)")
        ranking_fig = px.bar(
            ranking,
            x="조회기간(%)",
            y="기업",
            orientation="h",
            color="밸류체인",
            color_discrete_map=CATEGORY_COLORS,
            text="조회기간(%)",
            template=theme,
        )
        ranking_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        ranking_fig.update_layout(
            height=max(420, len(ranking) * 38),
            legend_title_text="",
            margin=dict(l=20, r=30, t=20, b=20),
        )
        ranking_fig.add_vline(x=0, line_color="gray")
        st.plotly_chart(ranking_fig, use_container_width=True)

    with right:
        st.subheader("수익률–변동성 위치")
        scatter = px.scatter(
            summary,
            x="연환산 변동성(%)",
            y="조회기간(%)",
            color="밸류체인",
            color_discrete_map=CATEGORY_COLORS,
            text="기업",
            hover_data=["티커", "최대낙폭(%)"],
            template=theme,
        )
        scatter.update_traces(textposition="top center", marker=dict(size=12))
        scatter.update_layout(
            height=520,
            legend_title_text="",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(scatter, use_container_width=True)
        st.caption("오른쪽일수록 가격 변동이 컸으며, 위쪽일수록 조회기간 수익률이 높았습니다.")

with technical_tab:
    detail_ticker = st.selectbox(
        "기술적 분석 종목",
        options=list(prices.keys()),
        format_func=lambda ticker: f"{stock_name(ticker)} ({ticker})",
        key="technical_ticker",
    )
    score, signal, signal_color, evidence, technical = technical_signal(prices[detail_ticker])
    close = adjusted_close(prices[detail_ticker])
    latest = technical.iloc[-1]
    daily_change = return_for_days(close, 1)
    rsi_value = latest.get("RSI14", math.nan)
    volatility = close.pct_change().std() * math.sqrt(252) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("최근 종가", format_number(close.iloc[-1]), f"{daily_change:+.2f}%")
    c2.metric("RSI(14)", format_number(rsi_value, 1))
    c3.metric("연환산 변동성", f"{volatility:.1f}%")
    c4.metric("최대낙폭", f"{max_drawdown(close):.1f}%")
    c5.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{signal_color}18;"
        f"border:1px solid {signal_color};height:82px'>"
        f"<div style='font-size:13px'>규칙 기반 추세 판정</div>"
        f"<div style='font-size:18px;font-weight:700;color:{signal_color}'>{signal}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    price_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.19, 0.19],
    )
    price_fig.add_trace(
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
        price_fig.add_trace(
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
    price_fig.add_trace(
        go.Scatter(x=technical.index, y=technical["RSI14"], name="RSI(14)", line=dict(color="#0f766e")),
        row=2,
        col=1,
    )
    price_fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=2, col=1)
    price_fig.add_hline(y=30, line_dash="dot", line_color="#2563eb", row=2, col=1)
    price_fig.add_trace(
        go.Scatter(x=technical.index, y=technical["MACD"], name="MACD", line=dict(color="#7c3aed")),
        row=3,
        col=1,
    )
    price_fig.add_trace(
        go.Scatter(
            x=technical.index,
            y=technical["MACD_SIGNAL"],
            name="Signal",
            line=dict(color="#f59e0b"),
        ),
        row=3,
        col=1,
    )
    price_fig.update_layout(
        title=f"{stock_name(detail_ticker)} ({detail_ticker}) · 가격 / RSI / MACD",
        template=theme,
        height=820,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=20, r=20, t=75, b=20),
    )
    price_fig.update_yaxes(title_text="가격", row=1, col=1)
    price_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    price_fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(price_fig, use_container_width=True)

    with st.expander("추세 판정 근거 보기", expanded=True):
        for item in evidence:
            st.write("• " + item)
        st.caption(
            f"종합 점수 {score:+d}. 이 판정은 이동평균·RSI·MACD만 사용한 규칙 기반 요약이며 "
            "미래 가격 예측이나 매수·매도 신호가 아닙니다."
        )

with fundamental_tab:
    fundamental_ticker = st.selectbox(
        "기업 정보 조회",
        options=list(prices.keys()),
        format_func=lambda ticker: f"{stock_name(ticker)} ({ticker})",
        key="fundamental_ticker",
    )
    with st.spinner("기업 정보와 밸류에이션을 불러오는 중입니다..."):
        info = download_company_info(fundamental_ticker)

    if not info:
        st.warning("Yahoo Finance에서 이 종목의 기업 정보를 제공하지 않았습니다.")
    else:
        company_title = info.get("longName") or stock_name(fundamental_ticker)
        st.subheader(f"{company_title} ({fundamental_ticker})")
        st.caption(
            " · ".join(
                filter(
                    None,
                    [info.get("sector"), info.get("industry"), info.get("country"), info.get("exchange")],
                )
            )
        )
        currency_symbol = "₩" if info.get("currency") == "KRW" else "$" if info.get("currency") == "USD" else ""

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("시가총액", compact_number(info.get("marketCap"), currency_symbol))
        f2.metric("후행 PER", format_number(info.get("trailingPE")))
        f3.metric("선행 PER", format_number(info.get("forwardPE")))
        f4.metric("주가매출비율(P/S)", format_number(info.get("priceToSalesTrailing12Months")))

        f5, f6, f7, f8 = st.columns(4)
        f5.metric("매출 성장률", f"{info.get('revenueGrowth') * 100:.1f}%" if info.get("revenueGrowth") is not None else "-")
        f6.metric("영업이익률", f"{info.get('operatingMargins') * 100:.1f}%" if info.get("operatingMargins") is not None else "-")
        f7.metric("총이익률", f"{info.get('grossMargins') * 100:.1f}%" if info.get("grossMargins") is not None else "-")
        f8.metric("부채비율", format_number(info.get("debtToEquity")))

        facts = pd.DataFrame(
            [
                ["기업가치(EV)", compact_number(info.get("enterpriseValue"), currency_symbol)],
                ["EV/EBITDA", format_number(info.get("enterpriseToEbitda"))],
                ["주가순자산비율(P/B)", format_number(info.get("priceToBook"))],
                ["PEG", format_number(info.get("pegRatio"))],
                ["잉여현금흐름", compact_number(info.get("freeCashflow"), currency_symbol)],
                ["총현금", compact_number(info.get("totalCash"), currency_symbol)],
                ["총부채", compact_number(info.get("totalDebt"), currency_symbol)],
                ["52주 최고가", format_number(info.get("fiftyTwoWeekHigh"))],
                ["52주 최저가", format_number(info.get("fiftyTwoWeekLow"))],
                ["애널리스트 목표주가 평균", format_number(info.get("targetMeanPrice"))],
            ],
            columns=["지표", "값"],
        )
        st.dataframe(facts, use_container_width=True, hide_index=True)

        summary_text = info.get("longBusinessSummary")
        if summary_text:
            with st.expander("Yahoo Finance 기업 설명(영문)"):
                st.write(summary_text)

        st.info(
            "서로 다른 국가·회계기준·사업모델의 기업을 PER 하나로 직접 비교하면 왜곡될 수 있습니다. "
            "AI 반도체 기업은 성장률, 영업이익률, 잉여현금흐름, 설비투자와 밸류체인 위치를 함께 보세요."
        )

with risk_tab:
    st.subheader("일간 수익률 상관관계")
    if returns.shape[1] >= 2:
        correlation = returns.corr()
        heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            template=theme,
        )
        heatmap.update_layout(height=620, coloraxis_colorbar_title="상관계수")
        st.plotly_chart(heatmap, use_container_width=True)
        st.caption("상관계수 1에 가까울수록 같은 방향, -1에 가까울수록 반대 방향으로 움직인 경향이 강합니다.")
    else:
        st.info("상관관계 분석에는 두 개 이상의 종목이 필요합니다.")

    st.subheader(f"{benchmark_name} 대비 민감도와 위험")
    risk_rows = []
    benchmark_return = returns.get(stock_name(benchmark_ticker))
    for ticker, frame in prices.items():
        name = stock_name(ticker)
        series = adjusted_close(frame).pct_change().rename(name)
        beta = math.nan
        if benchmark_return is not None and ticker != benchmark_ticker:
            aligned = pd.concat([series, benchmark_return.rename("benchmark")], axis=1).dropna()
            variance = aligned["benchmark"].var()
            if not aligned.empty and variance != 0:
                beta = aligned[name].cov(aligned["benchmark"]) / variance
        elif ticker == benchmark_ticker:
            beta = 1.0
        close_series = adjusted_close(frame)
        risk_rows.append(
            {
                "기업": name,
                "티커": ticker,
                f"베타({benchmark_ticker}=1)": beta,
                "연환산 변동성(%)": series.std() * math.sqrt(252) * 100,
                "최대낙폭(%)": max_drawdown(close_series),
                "상승일 비율(%)": (series.dropna() > 0).mean() * 100,
            }
        )
    risk_table = pd.DataFrame(risk_rows)
    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            f"베타({benchmark_ticker}=1)": st.column_config.NumberColumn(format="%.2f"),
            "연환산 변동성(%)": st.column_config.NumberColumn(format="%.2f"),
            "최대낙폭(%)": st.column_config.NumberColumn(format="%.2f"),
            "상승일 비율(%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

with data_tab:
    st.subheader("AI 반도체 종목 성과표")
    st.dataframe(
        summary.sort_values("조회기간(%)", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "종가": st.column_config.NumberColumn(format="%.2f"),
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
    for ticker, frame in prices.items():
        exported = frame.copy().reset_index()
        exported.insert(0, "Ticker", ticker)
        exported.insert(1, "Company", stock_name(ticker))
        exported.insert(2, "ValueChain", stock_category(ticker))
        export_frames.append(exported)
    export_data = pd.concat(export_frames, ignore_index=True)
    st.download_button(
        "AI 반도체 원자료 CSV 다운로드",
        data=export_data.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ai_semiconductor_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "본 페이지는 정보·교육용 분석 도구이며 투자 권유가 아닙니다. 기술적 지표는 과거 가격을 요약할 뿐 "
    "미래 수익률을 보장하지 않습니다. Yahoo Finance 데이터는 지연·수정·누락될 수 있습니다."
)
