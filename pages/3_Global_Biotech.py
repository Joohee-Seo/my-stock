import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="글로벌 바이오 Top 10", page_icon="🧬", layout="wide")


# 시가총액과 글로벌 인지도, 기술 플랫폼 다양성을 함께 고려한 대표 바이오 10종목입니다.
# 순수 시가총액 순위는 주가·환율·기업분류 기준에 따라 달라질 수 있습니다.
BIOTECH_TOP10 = {
    "Amgen": {"ticker": "AMGN", "country": "미국", "platform": "항체·단백질 치료제"},
    "Gilead Sciences": {"ticker": "GILD", "country": "미국", "platform": "항바이러스·세포치료"},
    "Regeneron": {"ticker": "REGN", "country": "미국", "platform": "항체 치료제"},
    "Vertex Pharmaceuticals": {"ticker": "VRTX", "country": "미국", "platform": "희귀질환·유전자치료"},
    "CSL": {"ticker": "CSL.AX", "country": "호주", "platform": "혈장·백신·희귀질환"},
    "Moderna": {"ticker": "MRNA", "country": "미국", "platform": "mRNA"},
    "argenx": {"ticker": "ARGX", "country": "네덜란드", "platform": "면역질환 항체"},
    "Genmab": {"ticker": "GMAB", "country": "덴마크", "platform": "이중항체·항암"},
    "BioNTech": {"ticker": "BNTX", "country": "독일", "platform": "mRNA·면역항암"},
    "CRISPR Therapeutics": {"ticker": "CRSP", "country": "스위스", "platform": "CRISPR 유전자 편집"},
}

BENCHMARKS = {"바이오 ETF(XBI)": "XBI", "바이오 ETF(IBB)": "IBB", "S&P 500": "^GSPC"}
PERIODS = {"3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y", "3년": "3y", "5년": "5y"}
PLATFORM_COLORS = {
    "항체·단백질 치료제": "#2563eb", "항바이러스·세포치료": "#0891b2",
    "항체 치료제": "#7c3aed", "희귀질환·유전자치료": "#db2777",
    "혈장·백신·희귀질환": "#ea580c", "mRNA": "#16a34a",
    "면역질환 항체": "#0f766e", "이중항체·항암": "#9333ea",
    "mRNA·면역항암": "#65a30d", "CRISPR 유전자 편집": "#dc2626", "직접 입력": "#64748b",
}

CATALOG = pd.DataFrame([
    {"label": f"{name} · {meta['ticker']} ({meta['country']})", "company": name, **meta}
    for name, meta in BIOTECH_TOP10.items()
])
LABEL_TO_TICKER = dict(zip(CATALOG["label"], CATALOG["ticker"]))
TICKER_TO_NAME = dict(zip(CATALOG["ticker"], CATALOG["company"]))
TICKER_TO_COUNTRY = dict(zip(CATALOG["ticker"], CATALOG["country"]))
TICKER_TO_PLATFORM = dict(zip(CATALOG["ticker"], CATALOG["platform"]))


def name_of(ticker): return TICKER_TO_NAME.get(ticker, ticker)
def country_of(ticker): return TICKER_TO_COUNTRY.get(ticker, "직접 입력")
def platform_of(ticker): return TICKER_TO_PLATFORM.get(ticker, "직접 입력")


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, period):
    tickers = tuple(dict.fromkeys(tickers))
    raw = yf.download(list(tickers), period=period, interval="1d", group_by="ticker",
                      auto_adjust=False, actions=False, progress=False, threads=True, timeout=20)
    result = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                l0, l1 = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
                frame = raw[ticker].copy() if ticker in l0 else raw.xs(ticker, axis=1, level=1).copy() if ticker in l1 else None
            else:
                frame = raw.copy()
            if frame is None:
                continue
            frame.columns = [str(c) for c in frame.columns]
            frame = frame.dropna(how="all")
            if "Close" in frame:
                frame = frame.dropna(subset=["Close"])
            if not frame.empty:
                result[ticker] = frame
        except (KeyError, TypeError, ValueError):
            continue
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def company_info(ticker):
    try:
        return yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}


def close_series(frame):
    return frame["Adj Close"].dropna() if "Adj Close" in frame and frame["Adj Close"].notna().any() else frame["Close"].dropna()


def rsi(series, window=14):
    delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, math.nan))


def indicators(frame):
    data = frame.copy(); close = data["Close"]
    data["SMA20"] = close.rolling(20).mean(); data["SMA50"] = close.rolling(50).mean(); data["SMA200"] = close.rolling(200).mean()
    data["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    data["SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean(); data["RSI14"] = rsi(close)
    return data


def ret_days(close, days):
    if len(close) <= days or float(close.iloc[-days-1]) == 0: return math.nan
    return (float(close.iloc[-1]) / float(close.iloc[-days-1]) - 1) * 100


def ytd(close):
    values = close[close.index.year == close.index[-1].year]
    return (float(values.iloc[-1]) / float(values.iloc[0]) - 1) * 100 if not values.empty and float(values.iloc[0]) else math.nan


def max_drawdown(close): return float((close / close.cummax() - 1).min() * 100)


def summary_table(prices):
    rows = []
    for ticker, frame in prices.items():
        close = close_series(frame); returns = close.pct_change().dropna()
        if close.empty: continue
        rows.append({
            "기업": name_of(ticker), "티커": ticker, "국가": country_of(ticker), "기술 플랫폼": platform_of(ticker),
            "종가": float(close.iloc[-1]), "1일(%)": ret_days(close, 1), "1개월(%)": ret_days(close, 21),
            "3개월(%)": ret_days(close, 63), "YTD(%)": ytd(close),
            "조회기간(%)": (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100,
            "변동성(%)": returns.std() * math.sqrt(252) * 100, "최대낙폭(%)": max_drawdown(close),
            "최대 1일 상승(%)": returns.max() * 100, "최대 1일 하락(%)": returns.min() * 100,
            "±10% 이벤트 일수": int((returns.abs() >= .10).sum()),
        })
    return pd.DataFrame(rows)


def normalized(prices):
    result = {}
    for ticker, frame in prices.items():
        close = close_series(frame)
        if not close.empty and float(close.iloc[0]): result[f"{name_of(ticker)} ({ticker})"] = close / float(close.iloc[0]) * 100
    return pd.DataFrame(result)


def returns_df(prices):
    return pd.DataFrame({name_of(t): close_series(f).pct_change() for t, f in prices.items()}).dropna(how="all")


def fmt(value, digits=2): return "-" if value is None or pd.isna(value) else f"{float(value):,.{digits}f}"
def pct(value): return "-" if value is None or pd.isna(value) else f"{float(value)*100:.1f}%"


def compact(value, symbol=""):
    if value is None or pd.isna(value): return "-"
    value = float(value)
    for size, suffix in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
        if abs(value) >= size: return f"{symbol}{value/size:,.2f}{suffix}"
    return f"{symbol}{value:,.0f}"


def trend_assessment(frame):
    data = indicators(frame).dropna(subset=["Close"]); last = data.iloc[-1]; score = 0; evidence = []
    for col, label in [("SMA20","20일선"),("SMA50","50일선"),("SMA200","200일선")]:
        if pd.notna(last[col]):
            above = last["Close"] > last[col]; score += 1 if above else -1; evidence.append(f"종가가 {label} {'위' if above else '아래'}")
    if pd.notna(last["MACD"]) and pd.notna(last["SIGNAL"]):
        above = last["MACD"] > last["SIGNAL"]; score += 1 if above else -1; evidence.append(f"MACD가 시그널선 {'위' if above else '아래'}")
    rv = last["RSI14"]
    if pd.notna(rv):
        if rv >= 70: score -= 1; evidence.append(f"RSI {rv:.1f}: 과매수")
        elif rv <= 30: evidence.append(f"RSI {rv:.1f}: 과매도")
        elif rv >= 50: score += 1; evidence.append(f"RSI {rv:.1f}: 양의 모멘텀")
        else: score -= 1; evidence.append(f"RSI {rv:.1f}: 약한 모멘텀")
    if score >= 4: label, color = "강한 상승 추세", "#16a34a"
    elif score >= 2: label, color = "상승 우위", "#65a30d"
    elif score >= -1: label, color = "중립·혼조", "#d97706"
    elif score >= -3: label, color = "하락 우위", "#ea580c"
    else: label, color = "강한 하락 추세", "#dc2626"
    return score, label, color, evidence, data


st.title("🧬 글로벌 바이오 Top 10 전문 분석")
st.caption("대형 상업화 바이오와 혁신 플랫폼 기업을 함께 비교합니다. " + f"조회 시각: {datetime.now():%Y-%m-%d %H:%M}")
st.info("CRISPR Therapeutics(CRSP)를 포함했습니다. CASGEVY는 Vertex와의 협업에서 나온 최초의 승인 CRISPR 기반 치료제입니다.")

with st.sidebar:
    st.header("바이오 분석 설정")
    selected_labels = st.multiselect("글로벌 대표 바이오 10종목", CATALOG["label"].tolist(), default=CATALOG["label"].tolist())
    custom = st.text_input("추가 티커", placeholder="예: BIIB, ALNY, BEAM")
    period_label = st.selectbox("분석 기간", list(PERIODS), index=2)
    theme = st.selectbox("차트 테마", ["plotly_white", "plotly_dark"])
    st.divider(); st.caption("바이오주는 임상·규제·특허·자금조달 이벤트에 따라 급변할 수 있습니다.")

tickers = list(dict.fromkeys([LABEL_TO_TICKER[x] for x in selected_labels] + [x.strip().upper() for x in custom.split(",") if x.strip()]))
if not tickers: st.info("분석할 종목을 선택하세요."); st.stop()
if len(tickers) > 20: st.warning("처음 20개 티커만 사용합니다."); tickers = tickers[:20]

with st.spinner("바이오 기업 데이터를 불러오는 중입니다..."):
    try: all_prices = download_prices(tuple(tickers + list(BENCHMARKS.values())), PERIODS[period_label])
    except Exception as error: st.error(f"데이터 조회 실패: {error}"); st.stop()
prices = {t: all_prices[t] for t in tickers if t in all_prices}
benchmarks = {n: all_prices[t] for n,t in BENCHMARKS.items() if t in all_prices}
missing = [t for t in tickers if t not in prices]
if missing: st.warning("데이터를 찾지 못한 티커: " + ", ".join(missing))
if not prices: st.error("분석 가능한 데이터가 없습니다."); st.stop()

summary = summary_table(prices); norm = normalized(prices); returns = returns_df(prices)
best = summary.loc[summary["조회기간(%)"].idxmax()]; riskiest = summary.loc[summary["변동성(%)"].idxmax()]
c1,c2,c3,c4 = st.columns(4)
c1.metric("분석 종목", f"{len(summary)}개"); c2.metric("기간 수익률 1위", best["기업"], f"{best['조회기간(%)']:+.1f}%")
c3.metric("평균 기간 수익률", f"{summary['조회기간(%)'].mean():+.1f}%"); c4.metric("최고 변동성", riskiest["기업"], f"연 {riskiest['변동성(%)']:.1f}%", delta_color="off")

overview, technical_tab, fundamentals, event_risk, correlation_tab, data_tab = st.tabs([
    "🌐 Top 10 개요", "📈 기술적 분석", "🏢 재무·현금", "⚠️ 이벤트 위험", "🔗 시장 상관관계", "📋 전체 데이터"
])

with overview:
    st.subheader(f"상대 주가 성과 · 시작값 100 ({period_label})")
    fig = px.line(norm, x=norm.index, y=norm.columns, template=theme, labels={"value":"지수화 가격","variable":"기업","Date":"날짜"})
    fig.update_layout(height=550, hovermode="x unified", legend_title_text=""); fig.add_hline(y=100,line_dash="dot",line_color="gray")
    st.plotly_chart(fig, width="stretch")
    left,right = st.columns(2)
    with left:
        rank = summary.sort_values("조회기간(%)")
        f = px.bar(rank,x="조회기간(%)",y="기업",orientation="h",color="기술 플랫폼",color_discrete_map=PLATFORM_COLORS,text="조회기간(%)",template=theme)
        f.update_traces(texttemplate="%{text:.1f}%",textposition="outside"); f.update_layout(height=500,legend_title_text="")
        st.plotly_chart(f,width="stretch")
    with right:
        f = px.scatter(summary,x="변동성(%)",y="조회기간(%)",color="기술 플랫폼",color_discrete_map=PLATFORM_COLORS,text="기업",hover_data=["티커","최대낙폭(%)"],template=theme)
        f.update_traces(textposition="top center",marker=dict(size=12)); f.update_layout(height=500,legend_title_text="")
        st.plotly_chart(f,width="stretch")

with technical_tab:
    ticker = st.selectbox("기술적 분석 기업", list(prices), format_func=lambda x:f"{name_of(x)} ({x})")
    score,label,color,evidence,tech = trend_assessment(prices[ticker]); close = close_series(prices[ticker]); last=tech.iloc[-1]
    a,b,c,d = st.columns(4); a.metric("종가",fmt(close.iloc[-1]),f"{ret_days(close,1):+.2f}%"); b.metric("RSI(14)",fmt(last['RSI14'],1)); c.metric("최대낙폭",f"{max_drawdown(close):.1f}%"); d.metric("추세 판정",label,f"점수 {score:+d}",delta_color="off")
    f=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.04,row_heights=[.62,.19,.19])
    f.add_trace(go.Candlestick(x=tech.index,open=tech.Open,high=tech.High,low=tech.Low,close=tech.Close,name="OHLC"),row=1,col=1)
    for col,co,la in [("SMA20","#f59e0b","20일"),("SMA50","#2563eb","50일"),("SMA200","#7c3aed","200일")]: f.add_trace(go.Scatter(x=tech.index,y=tech[col],name=la,line=dict(color=co)),row=1,col=1)
    f.add_trace(go.Scatter(x=tech.index,y=tech.RSI14,name="RSI"),row=2,col=1); f.add_hline(y=70,line_dash="dot",row=2,col=1); f.add_hline(y=30,line_dash="dot",row=2,col=1)
    f.add_trace(go.Scatter(x=tech.index,y=tech.MACD,name="MACD"),row=3,col=1); f.add_trace(go.Scatter(x=tech.index,y=tech.SIGNAL,name="Signal"),row=3,col=1)
    f.update_layout(height=820,template=theme,xaxis_rangeslider_visible=False,hovermode="x unified",legend=dict(orientation="h"))
    st.plotly_chart(f,width="stretch")
    with st.expander("판정 근거",expanded=True):
        for item in evidence: st.write("• "+item)
        st.caption("과거 가격 기반 요약이며 미래 예측이나 매매 신호가 아닙니다.")

with fundamentals:
    ticker = st.selectbox("재무 분석 기업",list(prices),format_func=lambda x:f"{name_of(x)} ({x})",key="bio_fund")
    with st.spinner("기업 정보를 불러오는 중입니다..."): info=company_info(ticker)
    if not info: st.warning("상세 기업 정보를 불러오지 못했습니다.")
    else:
        st.subheader(info.get("longName") or name_of(ticker)); st.caption(" · ".join(filter(None,[info.get("sector"),info.get("industry"),info.get("country"),info.get("currency")])))
        symbol={"USD":"$","EUR":"€","GBP":"£","AUD":"A$"}.get(info.get("currency"),"")
        x1,x2,x3,x4=st.columns(4); x1.metric("시가총액",compact(info.get("marketCap"),symbol)); x2.metric("후행 PER",fmt(info.get("trailingPE"))); x3.metric("선행 PER",fmt(info.get("forwardPE"))); x4.metric("매출 성장률",pct(info.get("revenueGrowth")))
        x5,x6,x7,x8=st.columns(4); x5.metric("영업이익률",pct(info.get("operatingMargins"))); x6.metric("잉여현금흐름",compact(info.get("freeCashflow"),symbol)); x7.metric("총현금",compact(info.get("totalCash"),symbol)); x8.metric("총부채",compact(info.get("totalDebt"),symbol))
        fcf=info.get("freeCashflow"); cash=info.get("totalCash")
        runway = cash/abs(fcf)*12 if cash and fcf and fcf<0 else None
        if runway: st.metric("단순 현금 런웨이 추정",f"약 {runway:.1f}개월",help="총현금÷연간 음의 잉여현금흐름으로 계산한 단순 추정치입니다.")
        facts=pd.DataFrame([["P/S",fmt(info.get("priceToSalesTrailing12Months"))],["P/B",fmt(info.get("priceToBook"))],["EV/EBITDA",fmt(info.get("enterpriseToEbitda"))],["총이익률",pct(info.get("grossMargins"))],["ROE",pct(info.get("returnOnEquity"))],["52주 최고가",fmt(info.get("fiftyTwoWeekHigh"))],["52주 최저가",fmt(info.get("fiftyTwoWeekLow"))],["목표주가 평균",fmt(info.get("targetMeanPrice"))]],columns=["지표","값"])
        st.dataframe(facts,width="stretch",hide_index=True)
        st.info("임상 단계 기업은 PER보다 현금 보유액, 현금소진 속도, 임상 단계·성공확률, 파트너십과 증자 위험이 더 중요할 수 있습니다.")

with event_risk:
    st.subheader("임상·규제 이벤트에 민감한 급등락 지표")
    cols=["기업","티커","변동성(%)","최대낙폭(%)","최대 1일 상승(%)","최대 1일 하락(%)","±10% 이벤트 일수"]
    st.dataframe(summary[cols].sort_values("변동성(%)",ascending=False),width="stretch",hide_index=True)
    f=px.scatter(summary,x="최대 1일 하락(%)",y="최대 1일 상승(%)",size="변동성(%)",color="기술 플랫폼",text="기업",color_discrete_map=PLATFORM_COLORS,template=theme)
    f.update_traces(textposition="top center"); f.update_layout(height=560,legend_title_text="")
    st.plotly_chart(f,width="stretch"); st.caption("급등락은 임상 결과 때문일 수도 있지만, 이 데이터만으로 원인을 특정할 수 없습니다.")

with correlation_tab:
    st.subheader("바이오 종목 간 일간 수익률 상관관계")
    if returns.shape[1]>=2:
        f=px.imshow(returns.corr(),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto",template=theme)
        f.update_layout(height=650); st.plotly_chart(f,width="stretch")
    ticker=st.selectbox("ETF 민감도 기업",list(prices),format_func=lambda x:f"{name_of(x)} ({x})",key="bio_beta")
    sr=close_series(prices[ticker]).pct_change().rename("Stock"); rows=[]
    for bn,bf in benchmarks.items():
        br=close_series(bf).pct_change().rename("Benchmark"); aligned=pd.concat([sr,br],axis=1).dropna(); var=aligned.Benchmark.var()
        rows.append({"기준지수":bn,"상관계수":aligned.Stock.corr(aligned.Benchmark),"베타":aligned.Stock.cov(aligned.Benchmark)/var if var else math.nan,"관측일수":len(aligned)})
    st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)

with data_tab:
    st.dataframe(summary.sort_values("조회기간(%)",ascending=False),width="stretch",hide_index=True)
    exports=[]
    for ticker,frame in prices.items():
        out=frame.reset_index(); out.insert(0,"Ticker",ticker); out.insert(1,"Company",name_of(ticker)); out.insert(2,"Platform",platform_of(ticker)); exports.append(out)
    st.download_button("바이오 원자료 CSV 다운로드",pd.concat(exports,ignore_index=True).to_csv(index=False).encode("utf-8-sig"),f"global_biotech_{datetime.now():%Y%m%d}.csv","text/csv",width="stretch")

st.divider(); st.caption("정보·교육용 분석 도구이며 투자 권유가 아닙니다. 기업 순위와 Yahoo Finance 데이터는 변동·지연·누락될 수 있습니다.")
