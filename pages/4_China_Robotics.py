import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="중국 로봇 기업 분석", page_icon="🤖", layout="wide")


# 중국 로봇 산업을 휴머노이드·본체·자동화·핵심부품·서비스 로봇으로 나눈 대표 상장사입니다.
# '순수 로봇 기업'과 로봇 밸류체인 노출 기업이 함께 포함됩니다.
CHINA_ROBOTICS = {
    "UBTECH Robotics": {"ticker": "9880.HK", "market": "홍콩", "segment": "휴머노이드", "exposure": "높음"},
    "Inovance Technology": {"ticker": "300124.SZ", "market": "선전", "segment": "자동화·모션제어", "exposure": "높음"},
    "Estun Automation": {"ticker": "002747.SZ", "market": "선전", "segment": "산업용 로봇", "exposure": "높음"},
    "SIASUN Robot": {"ticker": "300024.SZ", "market": "선전", "segment": "산업용 로봇", "exposure": "높음"},
    "EFORT Intelligent Equipment": {"ticker": "688165.SS", "market": "상하이", "segment": "산업용 로봇", "exposure": "높음"},
    "Leader Harmonious Drive": {"ticker": "688017.SS", "market": "상하이", "segment": "감속기·핵심부품", "exposure": "높음"},
    "Zhejiang Shuanghuan": {"ticker": "002472.SZ", "market": "선전", "segment": "정밀기어·핵심부품", "exposure": "중간"},
    "Roborock": {"ticker": "688169.SS", "market": "상하이", "segment": "서비스 로봇", "exposure": "높음"},
    "Ecovacs Robotics": {"ticker": "603486.SS", "market": "상하이", "segment": "서비스 로봇", "exposure": "높음"},
    "Hikvision": {"ticker": "002415.SZ", "market": "선전", "segment": "머신비전·물류로봇", "exposure": "중간"},
}

PRIVATE_LANDSCAPE = pd.DataFrame([
    ["Unitree Robotics", "휴머노이드·사족보행", "비상장/IPO 진행 여부 확인 필요", "저가형 하드웨어·동작 제어"],
    ["AgiBot (Zhiyuan Robotics)", "휴머노이드", "비상장", "범용 휴머노이드·데이터 수집"],
    ["Fourier Intelligence", "휴머노이드·재활", "비상장", "재활로봇 기반 휴머노이드"],
    ["Galbot", "범용 로봇", "비상장", "물류·리테일 조작 작업"],
    ["Leju Robotics", "휴머노이드", "비상장", "교육·산업용 휴머노이드"],
    ["DJI", "드론·로보틱스", "비상장", "비행제어·센서·드론 생태계"],
], columns=["기업", "분야", "시장 상태", "관찰 포인트"])

BENCHMARKS = {"CSI 300": "000300.SS", "항셍지수": "^HSI", "글로벌 로봇 ETF(BOTZ)": "BOTZ", "위안/달러": "CNY=X"}
PERIODS = {"3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y", "3년": "3y", "5년": "5y"}
COLORS = {"휴머노이드":"#dc2626","자동화·모션제어":"#2563eb","산업용 로봇":"#7c3aed","감속기·핵심부품":"#ea580c","정밀기어·핵심부품":"#f59e0b","서비스 로봇":"#16a34a","머신비전·물류로봇":"#0891b2","직접 입력":"#64748b"}

CATALOG = pd.DataFrame([{"label":f"{n} · {m['ticker']} ({m['market']})","company":n,**m} for n,m in CHINA_ROBOTICS.items()])
L2T = dict(zip(CATALOG.label, CATALOG.ticker)); T2N = dict(zip(CATALOG.ticker,CATALOG.company)); T2S = dict(zip(CATALOG.ticker,CATALOG.segment)); T2M = dict(zip(CATALOG.ticker,CATALOG.market)); T2E = dict(zip(CATALOG.ticker,CATALOG.exposure))
def name(t): return T2N.get(t,t)
def segment(t): return T2S.get(t,"직접 입력")


@st.cache_data(ttl=900, show_spinner=False)
def download(tickers, period):
    tickers=tuple(dict.fromkeys(tickers)); raw=yf.download(list(tickers),period=period,interval="1d",group_by="ticker",auto_adjust=False,actions=False,progress=False,threads=True,timeout=20); out={}
    for t in tickers:
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                l0,l1=raw.columns.get_level_values(0),raw.columns.get_level_values(1); f=raw[t].copy() if t in l0 else raw.xs(t,axis=1,level=1).copy() if t in l1 else None
            else: f=raw.copy()
            if f is None: continue
            f.columns=[str(c) for c in f.columns]; f=f.dropna(how="all"); f=f.dropna(subset=["Close"]) if "Close" in f else f
            if not f.empty: out[t]=f
        except (KeyError,TypeError,ValueError): continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def info(ticker):
    try: return yf.Ticker(ticker).get_info() or {}
    except Exception: return {}


def close(f): return f["Adj Close"].dropna() if "Adj Close" in f and f["Adj Close"].notna().any() else f["Close"].dropna()
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0); ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); al=loss.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); return 100-100/(1+ag/al.replace(0,math.nan))
def tech(f):
    x=f.copy(); c=x.Close; x["SMA20"]=c.rolling(20).mean(); x["SMA50"]=c.rolling(50).mean(); x["SMA200"]=c.rolling(200).mean(); x["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); x["SIGNAL"]=x.MACD.ewm(span=9,adjust=False).mean(); x["RSI14"]=rsi(c); return x
def rdays(c,n): return (float(c.iloc[-1])/float(c.iloc[-n-1])-1)*100 if len(c)>n and float(c.iloc[-n-1]) else math.nan
def ytd(c):
    x=c[c.index.year==c.index[-1].year]; return (float(x.iloc[-1])/float(x.iloc[0])-1)*100 if not x.empty and float(x.iloc[0]) else math.nan
def mdd(c): return float((c/c.cummax()-1).min()*100)
def fmt(v,n=2): return "-" if v is None or pd.isna(v) else f"{float(v):,.{n}f}"
def pct(v): return "-" if v is None or pd.isna(v) else f"{float(v)*100:.1f}%"
def compact(v,s=""):
    if v is None or pd.isna(v): return "-"
    for z,u in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
        if abs(float(v))>=z: return f"{s}{float(v)/z:,.2f}{u}"
    return f"{s}{float(v):,.0f}"


def summary(prices):
    rows=[]
    for t,f in prices.items():
        c=close(f); rr=c.pct_change().dropna()
        if c.empty: continue
        rows.append({"기업":name(t),"티커":t,"시장":T2M.get(t,"직접 입력"),"밸류체인":segment(t),"로봇 노출":T2E.get(t,"-"),"종가":float(c.iloc[-1]),"1일(%)":rdays(c,1),"1개월(%)":rdays(c,21),"3개월(%)":rdays(c,63),"YTD(%)":ytd(c),"조회기간(%)":(float(c.iloc[-1])/float(c.iloc[0])-1)*100,"변동성(%)":rr.std()*math.sqrt(252)*100,"최대낙폭(%)":mdd(c),"±7% 변동일":int((rr.abs()>=.07).sum())})
    return pd.DataFrame(rows)


def normalized(prices):
    o={}
    for t,f in prices.items():
        c=close(f)
        if not c.empty and float(c.iloc[0]): o[f"{name(t)} ({t})"]=c/float(c.iloc[0])*100
    return pd.DataFrame(o)


def assess(f):
    x=tech(f).dropna(subset=["Close"]); q=x.iloc[-1]; score=0; evidence=[]
    for col,label in [("SMA20","20일선"),("SMA50","50일선"),("SMA200","200일선")]:
        if pd.notna(q[col]): up=q.Close>q[col]; score+=1 if up else -1; evidence.append(f"종가가 {label} {'위' if up else '아래'}")
    if pd.notna(q.MACD) and pd.notna(q.SIGNAL): up=q.MACD>q.SIGNAL; score+=1 if up else -1; evidence.append(f"MACD가 시그널선 {'위' if up else '아래'}")
    rv=q.RSI14
    if pd.notna(rv):
        if rv>=70: score-=1; evidence.append(f"RSI {rv:.1f}: 과매수")
        elif rv<=30: evidence.append(f"RSI {rv:.1f}: 과매도")
        elif rv>=50: score+=1; evidence.append(f"RSI {rv:.1f}: 양의 모멘텀")
        else: score-=1; evidence.append(f"RSI {rv:.1f}: 약한 모멘텀")
    label="강한 상승" if score>=4 else "상승 우위" if score>=2 else "중립·혼조" if score>=-1 else "하락 우위" if score>=-3 else "강한 하락"
    return score,label,evidence,x


st.title("🤖 중국 로봇 기업 전문 분석")
st.caption("휴머노이드·산업용 로봇·자동화·핵심부품·서비스 로봇 상장사를 밸류체인 관점에서 비교합니다. "+f"조회 시각: {datetime.now():%Y-%m-%d %H:%M}")
st.info("UBTECH(9880.HK)는 직접 상장된 휴머노이드 기업입니다. Unitree·AgiBot 등 주요 비상장사는 별도 산업지도에서 확인할 수 있습니다.")

with st.sidebar:
    st.header("중국 로봇 분석 설정"); labels=st.multiselect("대표 상장사 10개",CATALOG.label.tolist(),default=CATALOG.label.tolist()); custom=st.text_input("추가 티커",placeholder="예: 9868.HK, 688218.SS"); period_label=st.selectbox("분석 기간",list(PERIODS),index=2); theme=st.selectbox("차트 테마",["plotly_white","plotly_dark"]); st.divider(); st.caption("A주·H주는 거래통화, 접근성, 회계·공시 체계가 서로 다릅니다.")
tickers=list(dict.fromkeys([L2T[x] for x in labels]+[x.strip().upper() for x in custom.split(",") if x.strip()]))
if not tickers: st.info("종목을 선택하세요."); st.stop()
if len(tickers)>20: tickers=tickers[:20]; st.warning("처음 20개 티커만 사용합니다.")
with st.spinner("중국 로봇 기업 데이터를 불러오는 중입니다..."):
    try: allp=download(tuple(tickers+list(BENCHMARKS.values())),PERIODS[period_label])
    except Exception as e: st.error(f"데이터 조회 실패: {e}"); st.stop()
prices={t:allp[t] for t in tickers if t in allp}; bench={n:allp[t] for n,t in BENCHMARKS.items() if t in allp}; missing=[t for t in tickers if t not in prices]
if missing: st.warning("Yahoo Finance에서 찾지 못한 티커: "+", ".join(missing))
if not prices: st.error("분석 가능한 데이터가 없습니다."); st.stop()
s=summary(prices); norm=normalized(prices); best=s.loc[s["조회기간(%)"].idxmax()]; risky=s.loc[s["변동성(%)"].idxmax()]
a,b,c,d=st.columns(4); a.metric("분석 종목",f"{len(s)}개"); b.metric("기간 수익률 1위",best.기업,f"{best['조회기간(%)']:+.1f}%"); c.metric("평균 수익률",f"{s['조회기간(%)'].mean():+.1f}%"); d.metric("최고 변동성",risky.기업,f"연 {risky['변동성(%)']:.1f}%",delta_color="off")

overview,technical,fundamental,benchmark_tab,landscape,risk_tab,data_tab=st.tabs(["🌐 상장사 개요","📈 기술적 분석","🏢 기업가치","🇨🇳 중국시장 비교","🦾 비상장 산업지도","⚠️ 위험 분석","📋 전체 데이터"])

with overview:
    f=px.line(norm,x=norm.index,y=norm.columns,template=theme,labels={"value":"지수화 가격","variable":"기업","Date":"날짜"}); f.update_layout(height=550,hovermode="x unified",legend_title_text=""); f.add_hline(y=100,line_dash="dot"); st.plotly_chart(f,width="stretch")
    l,r=st.columns(2)
    with l:
        rank=s.sort_values("조회기간(%)"); f=px.bar(rank,x="조회기간(%)",y="기업",orientation="h",color="밸류체인",color_discrete_map=COLORS,text="조회기간(%)",template=theme); f.update_traces(texttemplate="%{text:.1f}%",textposition="outside"); f.update_layout(height=520,legend_title_text=""); st.plotly_chart(f,width="stretch")
    with r:
        f=px.scatter(s,x="변동성(%)",y="조회기간(%)",color="밸류체인",color_discrete_map=COLORS,text="기업",hover_data=["티커","최대낙폭(%)","로봇 노출"],template=theme); f.update_traces(textposition="top center",marker=dict(size=12)); f.update_layout(height=520,legend_title_text=""); st.plotly_chart(f,width="stretch")

with technical:
    t=st.selectbox("기술적 분석 기업",list(prices),format_func=lambda x:f"{name(x)} ({x})"); score,label,evidence,x=assess(prices[t]); c=close(prices[t]); q=x.iloc[-1]
    p1,p2,p3,p4=st.columns(4); p1.metric("종가",fmt(c.iloc[-1]),f"{rdays(c,1):+.2f}%"); p2.metric("RSI(14)",fmt(q.RSI14,1)); p3.metric("최대낙폭",f"{mdd(c):.1f}%"); p4.metric("추세",label,f"점수 {score:+d}",delta_color="off")
    f=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.04,row_heights=[.62,.19,.19]); f.add_trace(go.Candlestick(x=x.index,open=x.Open,high=x.High,low=x.Low,close=x.Close,name="OHLC"),row=1,col=1)
    for col,co,la in [("SMA20","#f59e0b","20일"),("SMA50","#2563eb","50일"),("SMA200","#7c3aed","200일")]: f.add_trace(go.Scatter(x=x.index,y=x[col],name=la,line=dict(color=co)),row=1,col=1)
    f.add_trace(go.Scatter(x=x.index,y=x.RSI14,name="RSI"),row=2,col=1); f.add_hline(y=70,line_dash="dot",row=2,col=1); f.add_hline(y=30,line_dash="dot",row=2,col=1); f.add_trace(go.Scatter(x=x.index,y=x.MACD,name="MACD"),row=3,col=1); f.add_trace(go.Scatter(x=x.index,y=x.SIGNAL,name="Signal"),row=3,col=1); f.update_layout(height=820,template=theme,xaxis_rangeslider_visible=False,hovermode="x unified",legend=dict(orientation="h")); st.plotly_chart(f,width="stretch")
    with st.expander("판정 근거",expanded=True):
        for item in evidence: st.write("• "+item)
        st.caption("과거 가격의 규칙 기반 요약이며 미래 예측이나 매매 신호가 아닙니다.")

with fundamental:
    t=st.selectbox("기업가치 분석",list(prices),format_func=lambda x:f"{name(x)} ({x})",key="robot_fund")
    with st.spinner("기업 정보를 불러오는 중입니다..."): i=info(t)
    if not i: st.warning("상세 정보를 불러오지 못했습니다. 중국 A주 정보는 일부 누락될 수 있습니다.")
    else:
        st.subheader(i.get("longName") or name(t)); st.caption(" · ".join(filter(None,[i.get("sector"),i.get("industry"),i.get("country"),i.get("currency")])))
        sym={"CNY":"¥","HKD":"HK$","USD":"$"}.get(i.get("currency"),""); q1,q2,q3,q4=st.columns(4); q1.metric("시가총액",compact(i.get("marketCap"),sym)); q2.metric("후행 PER",fmt(i.get("trailingPE"))); q3.metric("선행 PER",fmt(i.get("forwardPE"))); q4.metric("매출 성장률",pct(i.get("revenueGrowth")))
        q5,q6,q7,q8=st.columns(4); q5.metric("영업이익률",pct(i.get("operatingMargins"))); q6.metric("ROE",pct(i.get("returnOnEquity"))); q7.metric("총현금",compact(i.get("totalCash"),sym)); q8.metric("총부채",compact(i.get("totalDebt"),sym))
        facts=pd.DataFrame([["P/S",fmt(i.get("priceToSalesTrailing12Months"))],["P/B",fmt(i.get("priceToBook"))],["EV/EBITDA",fmt(i.get("enterpriseToEbitda"))],["잉여현금흐름",compact(i.get("freeCashflow"),sym)],["52주 최고가",fmt(i.get("fiftyTwoWeekHigh"))],["52주 최저가",fmt(i.get("fiftyTwoWeekLow"))]],columns=["지표","값"]); st.dataframe(facts,width="stretch",hide_index=True)
        st.info("로봇 관련 매출 비중이 낮은 밸류체인 기업은 로봇 테마 상승이 전체 실적 개선으로 바로 이어지지 않을 수 있습니다.")

with benchmark_tab:
    t=st.selectbox("중국시장 비교 기업",list(prices),format_func=lambda x:f"{name(x)} ({x})",key="robot_bench"); series={}; c=close(prices[t]); series[f"{name(t)} ({t})"]=c/float(c.iloc[0])*100
    for n,fm in bench.items():
        cc=close(fm)
        if not cc.empty and float(cc.iloc[0]): series[n]=cc/float(cc.iloc[0])*100
    z=pd.DataFrame(series); f=px.line(z,x=z.index,y=z.columns,template=theme,labels={"value":"지수화 가격","variable":"자산","Date":"날짜"}); f.update_layout(height=570,hovermode="x unified",legend_title_text=""); f.add_hline(y=100,line_dash="dot"); st.plotly_chart(f,width="stretch")

with landscape:
    st.subheader("중국 주요 비상장 로봇 기업"); st.dataframe(PRIVATE_LANDSCAPE,width="stretch",hide_index=True); st.warning("비상장 기업에는 일반 주식 티커가 없습니다. 유사한 회사명이나 테마주를 해당 기업의 주식으로 오인하지 마세요.")

with risk_tab:
    rr=pd.DataFrame({name(t):close(f).pct_change() for t,f in prices.items()}).dropna(how="all")
    if rr.shape[1]>=2:
        f=px.imshow(rr.corr(),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto",template=theme); f.update_layout(height=650); st.plotly_chart(f,width="stretch")
    st.dataframe(s[["기업","티커","변동성(%)","최대낙폭(%)","±7% 변동일"]].sort_values("변동성(%)",ascending=False),width="stretch",hide_index=True)
    st.caption("정책 기대, 증자, 보호예수 해제, 수주 공시, 기술 시연과 미·중 규제 변화가 큰 가격 변동을 만들 수 있습니다.")

with data_tab:
    st.dataframe(s.sort_values("조회기간(%)",ascending=False),width="stretch",hide_index=True); exports=[]
    for t,fm in prices.items(): out=fm.reset_index(); out.insert(0,"Ticker",t); out.insert(1,"Company",name(t)); out.insert(2,"Segment",segment(t)); exports.append(out)
    st.download_button("중국 로봇 원자료 CSV 다운로드",pd.concat(exports,ignore_index=True).to_csv(index=False).encode("utf-8-sig"),f"china_robotics_{datetime.now():%Y%m%d}.csv","text/csv",width="stretch")

st.divider(); st.caption("정보·교육용 분석 도구이며 투자 권유가 아닙니다. 중국 본토·홍콩 데이터는 지연·누락될 수 있고 상장 상태와 기업 순위는 변동될 수 있습니다.")
