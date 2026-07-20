import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="국내 레버리지 ETF Top 10", page_icon="🚀", layout="wide")


# 거래량·인지도·기초지수 다양성을 고려한 국내 상장 대표 10종입니다.
# 실시간 순자산총액·거래대금 순위는 시장 상황에 따라 달라질 수 있습니다.
PRODUCTS = {
    "KODEX 레버리지": {"ticker":"122630.KS","direction":"+2배","multiple":2,"underlying":"KOSPI 200","benchmark":"^KS200","manager":"삼성"},
    "KODEX 200선물인버스2X": {"ticker":"252670.KS","direction":"-2배","multiple":-2,"underlying":"KOSPI 200","benchmark":"^KS200","manager":"삼성"},
    "KODEX 코스닥150레버리지": {"ticker":"233740.KS","direction":"+2배","multiple":2,"underlying":"KOSDAQ 150","benchmark":"^KQ150","manager":"삼성"},
    "TIGER 레버리지": {"ticker":"123320.KS","direction":"+2배","multiple":2,"underlying":"KOSPI 200","benchmark":"^KS200","manager":"미래에셋"},
    "TIGER 200선물인버스2X": {"ticker":"252710.KS","direction":"-2배","multiple":-2,"underlying":"KOSPI 200","benchmark":"^KS200","manager":"미래에셋"},
    "TIGER 코스닥150 레버리지": {"ticker":"233160.KS","direction":"+2배","multiple":2,"underlying":"KOSDAQ 150","benchmark":"^KQ150","manager":"미래에셋"},
    "PLUS 200선물레버리지": {"ticker":"253150.KS","direction":"+2배","multiple":2,"underlying":"KOSPI 200","benchmark":"^KS200","manager":"한화"},
    "PLUS 200선물인버스2X": {"ticker":"253160.KS","direction":"-2배","multiple":-2,"underlying":"KOSPI 200","benchmark":"^KS200","manager":"한화"},
    "KODEX 미국나스닥100레버리지(합성 H)": {"ticker":"409820.KS","direction":"+2배","multiple":2,"underlying":"NASDAQ 100(환헤지)","benchmark":"^NDX","manager":"삼성"},
    "TIGER 미국나스닥100레버리지(합성)": {"ticker":"418660.KS","direction":"+2배","multiple":2,"underlying":"NASDAQ 100(환노출)","benchmark":"^NDX","manager":"미래에셋"},
}

PERIODS={"1개월":"1mo","3개월":"3mo","6개월":"6mo","1년":"1y","2년":"2y","3년":"3y","5년":"5y"}
COLORS={"+2배":"#dc2626","-2배":"#2563eb"}
CATALOG=pd.DataFrame([{"label":f"{n} · {m['ticker'].replace('.KS','')}","name":n,**m} for n,m in PRODUCTS.items()])
L2T=dict(zip(CATALOG.label,CATALOG.ticker)); T2N=dict(zip(CATALOG.ticker,CATALOG.name)); T2D=dict(zip(CATALOG.ticker,CATALOG.direction)); T2U=dict(zip(CATALOG.ticker,CATALOG.underlying)); T2M=dict(zip(CATALOG.ticker,CATALOG.multiple)); T2B=dict(zip(CATALOG.ticker,CATALOG.benchmark)); T2A=dict(zip(CATALOG.ticker,CATALOG.manager))
def name(t): return T2N.get(t,t)


@st.cache_data(ttl=900,show_spinner=False)
def download(tickers,period):
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


def close(f): return f["Adj Close"].dropna() if "Adj Close" in f and f["Adj Close"].notna().any() else f["Close"].dropna()
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0); ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); al=loss.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); return 100-100/(1+ag/al.replace(0,math.nan))
def tech(f):
    x=f.copy(); c=x.Close; x["SMA20"]=c.rolling(20).mean(); x["SMA60"]=c.rolling(60).mean(); x["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); x["SIGNAL"]=x.MACD.ewm(span=9,adjust=False).mean(); x["RSI14"]=rsi(c); return x
def rdays(c,n): return (float(c.iloc[-1])/float(c.iloc[-n-1])-1)*100 if len(c)>n and float(c.iloc[-n-1]) else math.nan
def ytd(c):
    x=c[c.index.year==c.index[-1].year]; return (float(x.iloc[-1])/float(x.iloc[0])-1)*100 if not x.empty and float(x.iloc[0]) else math.nan
def mdd(c): return float((c/c.cummax()-1).min()*100)
def fmt(v,n=2): return "-" if v is None or pd.isna(v) else f"{float(v):,.{n}f}"


def summary(prices):
    rows=[]
    for t,f in prices.items():
        c=close(f); rr=c.pct_change().dropna()
        if c.empty: continue
        rows.append({"상품":name(t),"티커":t.replace(".KS",""),"방향":T2D.get(t,"-"),"기초자산":T2U.get(t,"-"),"운용사":T2A.get(t,"-"),"종가":float(c.iloc[-1]),"1일(%)":rdays(c,1),"1개월(%)":rdays(c,21),"3개월(%)":rdays(c,63),"YTD(%)":ytd(c),"조회기간(%)":(float(c.iloc[-1])/float(c.iloc[0])-1)*100,"변동성(%)":rr.std()*math.sqrt(252)*100,"최대낙폭(%)":mdd(c),"VaR 95%(%)":rr.quantile(.05)*100,"±10% 변동일":int((rr.abs()>=.10).sum())})
    return pd.DataFrame(rows)


def normalized(prices):
    out={}
    for t,f in prices.items():
        c=close(f)
        if not c.empty and float(c.iloc[0]): out[name(t)]=c/float(c.iloc[0])*100
    return pd.DataFrame(out)


st.title("🚀 국내 레버리지·2배 인버스 ETF Top 10")
st.caption("국내 상장 대표 레버리지 상품의 성과·위험·일간 추종오차·변동성 누적효과를 분석합니다. "+f"조회 시각: {datetime.now():%Y-%m-%d %H:%M}")
st.error("레버리지 ETF는 기초자산의 ‘하루 수익률’ ±2배를 목표로 합니다. 장기 수익률은 기초자산 누적수익률의 단순 2배가 아니며 손실도 빠르게 확대될 수 있습니다.")

with st.sidebar:
    st.header("레버리지 분석 설정"); labels=st.multiselect("대표 상품 10개",CATALOG.label.tolist(),default=CATALOG.label.tolist()); custom=st.text_input("추가 국내 ETF 티커",placeholder="예: 267770.KS"); period_label=st.selectbox("분석 기간",list(PERIODS),index=3); theme=st.selectbox("차트 테마",["plotly_white","plotly_dark"]); st.divider(); st.caption("신규 단일종목 레버리지 상품은 Yahoo 데이터가 안정화된 뒤 추가하는 것이 안전합니다.")
tickers=list(dict.fromkeys([L2T[x] for x in labels]+[x.strip().upper() for x in custom.split(",") if x.strip()]))
if not tickers: st.info("상품을 선택하세요."); st.stop()
benchmarks=list(dict.fromkeys(T2B[t] for t in tickers if t in T2B))
with st.spinner("국내 레버리지 ETF 데이터를 불러오는 중입니다..."):
    try: allp=download(tuple(tickers+benchmarks),PERIODS[period_label])
    except Exception as e: st.error(f"데이터 조회 실패: {e}"); st.stop()
prices={t:allp[t] for t in tickers if t in allp}; missing=[t for t in tickers if t not in prices]
if missing: st.warning("Yahoo Finance에서 찾지 못한 티커: "+", ".join(missing))
if not prices: st.error("분석 가능한 데이터가 없습니다."); st.stop()
s=summary(prices); norm=normalized(prices); best=s.loc[s["조회기간(%)"].idxmax()]; worst=s.loc[s["최대낙폭(%)"].idxmin()]
a,b,c,d=st.columns(4); a.metric("분석 상품",f"{len(s)}개"); b.metric("기간 수익률 1위",best.상품,f"{best['조회기간(%)']:+.1f}%"); c.metric("평균 변동성",f"연 {s['변동성(%)'].mean():.1f}%"); d.metric("최대 낙폭 상품",worst.상품,f"{worst['최대낙폭(%)']:.1f}%",delta_color="inverse")

overview,tracking,technical,risk_tab,guide,data_tab=st.tabs(["🌐 Top 10 개요","🧮 2배 추종 분석","📈 기술적 분석","⚠️ 위험 분석","📘 상품 이해","📋 전체 데이터"])

with overview:
    f=px.line(norm,x=norm.index,y=norm.columns,template=theme,labels={"value":"지수화 가격","variable":"상품","Date":"날짜"}); f.update_layout(height=570,hovermode="x unified",legend_title_text=""); f.add_hline(y=100,line_dash="dot"); st.plotly_chart(f,width="stretch")
    l,r=st.columns(2)
    with l:
        rank=s.sort_values("조회기간(%)"); f=px.bar(rank,x="조회기간(%)",y="상품",orientation="h",color="방향",color_discrete_map=COLORS,text="조회기간(%)",template=theme); f.update_traces(texttemplate="%{text:.1f}%",textposition="outside"); f.update_layout(height=520,legend_title_text=""); st.plotly_chart(f,width="stretch")
    with r:
        f=px.scatter(s,x="변동성(%)",y="조회기간(%)",color="방향",color_discrete_map=COLORS,text="상품",hover_data=["티커","기초자산","최대낙폭(%)"],template=theme); f.update_traces(textposition="top center",marker=dict(size=12)); f.update_layout(height=520,legend_title_text=""); st.plotly_chart(f,width="stretch")

with tracking:
    available=[t for t in prices if t in T2B and T2B[t] in allp]
    if not available: st.warning("기초지수 데이터를 불러오지 못했습니다.")
    else:
        t=st.selectbox("추종 분석 상품",available,format_func=lambda x:f"{name(x)} ({x.replace('.KS','')})"); etf=close(prices[t]); base=close(allp[T2B[t]]); aligned=pd.concat([etf.rename("ETF"),base.rename("기초지수")],axis=1).dropna(); er=aligned.ETF.pct_change(); br=aligned.기초지수.pct_change(); multiple=T2M[t]
        daily=pd.DataFrame({"실제 ETF 일간수익률(%)":er*100,"목표 일간수익률(%)":br*multiple*100}).dropna(); daily["추종차이(%p)"]=daily.iloc[:,0]-daily.iloc[:,1]
        actual=aligned.ETF/aligned.ETF.iloc[0]*100; theoretical=(1+(br.fillna(0)*multiple)).cumprod()*100; simple=(1+(aligned.기초지수/aligned.기초지수.iloc[0]-1)*multiple)*100
        comp=pd.DataFrame({"실제 ETF":actual,"일간 재조정 이론값":theoretical,"단순 누적수익률 × 배수":simple})
        f=px.line(comp,x=comp.index,y=comp.columns,template=theme,labels={"value":"지수화 가격","variable":"비교","Date":"날짜"}); f.update_layout(height=570,hovermode="x unified",legend_title_text=""); f.add_hline(y=100,line_dash="dot"); st.plotly_chart(f,width="stretch")
        q1,q2,q3,q4=st.columns(4); q1.metric("목표 배수",f"{multiple:+d}배"); q2.metric("일간 추종차이 평균",f"{daily['추종차이(%p)'].mean():+.3f}%p"); q3.metric("추종차이 표준편차",f"{daily['추종차이(%p)'].std():.3f}%p"); q4.metric("분석 일수",f"{len(daily)}일")
        st.caption("환헤지, 선물 롤오버, 보수, 세금, 거래시간 차이로 실제 수익률과 단순 목표값 사이에 차이가 생길 수 있습니다.")

with technical:
    t=st.selectbox("기술적 분석 상품",list(prices),format_func=lambda x:f"{name(x)} ({x.replace('.KS','')})",key="lev_tech"); x=tech(prices[t]); c=close(prices[t]); q=x.iloc[-1]
    z1,z2,z3,z4=st.columns(4); z1.metric("종가",fmt(c.iloc[-1],0),f"{rdays(c,1):+.2f}%"); z2.metric("RSI(14)",fmt(q.RSI14,1)); z3.metric("연환산 변동성",f"{c.pct_change().std()*math.sqrt(252)*100:.1f}%"); z4.metric("최대낙폭",f"{mdd(c):.1f}%")
    f=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.04,row_heights=[.62,.19,.19]); f.add_trace(go.Candlestick(x=x.index,open=x.Open,high=x.High,low=x.Low,close=x.Close,name="OHLC"),row=1,col=1); f.add_trace(go.Scatter(x=x.index,y=x.SMA20,name="20일선"),row=1,col=1); f.add_trace(go.Scatter(x=x.index,y=x.SMA60,name="60일선"),row=1,col=1); f.add_trace(go.Scatter(x=x.index,y=x.RSI14,name="RSI"),row=2,col=1); f.add_hline(y=70,line_dash="dot",row=2,col=1); f.add_hline(y=30,line_dash="dot",row=2,col=1); f.add_trace(go.Scatter(x=x.index,y=x.MACD,name="MACD"),row=3,col=1); f.add_trace(go.Scatter(x=x.index,y=x.SIGNAL,name="Signal"),row=3,col=1); f.update_layout(height=820,template=theme,xaxis_rangeslider_visible=False,hovermode="x unified",legend=dict(orientation="h")); st.plotly_chart(f,width="stretch")

with risk_tab:
    cols=["상품","티커","방향","기초자산","변동성(%)","최대낙폭(%)","VaR 95%(%)","±10% 변동일"]; st.dataframe(s[cols].sort_values("최대낙폭(%)"),width="stretch",hide_index=True)
    st.warning("VaR 95%는 과거 일간 수익률의 하위 5% 지점일 뿐 최대손실 한도가 아닙니다. 급변장에서는 이보다 큰 손실이 발생할 수 있습니다.")

with guide:
    st.subheader("레버리지 ETF에서 반드시 이해할 다섯 가지")
    st.markdown("""
1. **배수의 기준은 하루입니다.** 한 달·1년 수익률의 정확한 2배를 보장하지 않습니다.
2. **횡보하며 크게 출렁이면 손실이 누적될 수 있습니다.** 이를 변동성 누적효과 또는 음의 복리효과라고 합니다.
3. **인버스 2X도 장기 하락의 단순 두 배가 아닙니다.** 매일 -2배로 재조정됩니다.
4. **괴리율·추적오차·롤오버 비용을 확인해야 합니다.** 특히 선물·합성 상품은 구조가 더 복잡합니다.
5. **신규 단일종목 레버리지는 지수형보다 집중위험이 큽니다.** 기초주식 한 종목의 급락이 거의 두 배로 확대될 수 있습니다.
""")
    st.info("이 페이지의 ‘Top 10’은 실시간 거래대금 순위가 아니라 장기 데이터 분석이 가능한 대표 상품군입니다.")

with data_tab:
    st.dataframe(s.sort_values("조회기간(%)",ascending=False),width="stretch",hide_index=True); exports=[]
    for t,fm in prices.items(): out=fm.reset_index(); out.insert(0,"Ticker",t); out.insert(1,"Product",name(t)); out.insert(2,"Direction",T2D.get(t,"-")); exports.append(out)
    st.download_button("레버리지 ETF 원자료 CSV 다운로드",pd.concat(exports,ignore_index=True).to_csv(index=False).encode("utf-8-sig"),f"korea_leveraged_etf_{datetime.now():%Y%m%d}.csv","text/csv",width="stretch")

st.divider(); st.caption("정보·교육용 분석 도구이며 투자 권유가 아닙니다. 상품명·규제·거래조건·Yahoo Finance 데이터는 변경·지연·누락될 수 있습니다.")
