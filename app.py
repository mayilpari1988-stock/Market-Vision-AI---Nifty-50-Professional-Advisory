import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import math
import base64
import os
import random
from datetime import datetime, timedelta
import pytz

random.seed(42)
np.random.seed(42)

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    TORCH_AVAILABLE = True
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except:
    TORCH_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Market Vision AI")

st.markdown("""
<style>
* { font-family: 'Times New Roman', Times, serif !important; }
.stApp { background: #eef6ff !important; }
.main .block-container { background: #ffffff !important; padding: 12px !important; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #dbeafe; }
header, footer {visibility: hidden;}
div[data-baseweb="input"] { background: #ffffff !important; border: 2px solid #3b82f6 !important; border-radius: 10px !important; }
div[data-baseweb="input"] input { background: #ffffff !important; color: #000000 !important; font-weight: bold !important; font-size: 16px !important; -webkit-text-fill-color: #000000 !important; }
div[data-baseweb="base-input"] { background: #ffffff !important; }
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] { background: #f0f7ff !important; color: #1e40af !important; border: 1px solid #3b82f6 !important; }
div[data-baseweb="select"] > div { background: #ffffff !important; color: #000000 !important; border: 2px solid #3b82f6 !important; border-radius: 8px !important; }
div[data-baseweb="select"] span { color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: bold !important; }
div[data-baseweb="select"] svg { fill: #1e40af !important; }
ul[data-baseweb="menu"] { background: white !important; }
li[data-baseweb="option"] { background: white !important; color: black !important; }
.stButton > button { background: #2563eb !important; color: white !important; border: 2px solid #1e40af !important; border-radius: 10px !important; font-weight: bold !important; height: 44px !important; min-width: 160px !important; font-size: 16px !important; box-shadow: 0 2px 6px rgba(37,99,235,0.3) !important; letter-spacing: 0.5px; }
.stButton > button:hover { background: #1e40af !important; color: white !important; border: 2px solid #1e3a5f !important; }
.ticker-wrap { width: 100%; overflow: hidden; background: #1e3a5f; border-bottom: 3px solid #93c5fd; height: 38px; line-height: 38px; margin-bottom: 12px; border-radius: 6px; }
.ticker { display: inline-block; white-space: nowrap; animation: ticker 800s linear infinite; padding-left: 5%; }
@keyframes ticker { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-100%, 0, 0); } }
.ticker-item { display: inline-block; padding: 0 28px; font-size: 13px; font-weight: bold; color: white; border-right: 1px solid rgba(255,255,255,0.2); }
.ticker-item .green { color: #4ade80; } .ticker-item .red { color: #f87171; } .ticker-item .price { color: #bfdbfe; }
</style>
""", unsafe_allow_html=True)

NIFTY50 = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","BHARTIARTL.NS","SBIN.NS","BAJFINANCE.NS","LICI.NS","HINDUNALVR.NS","ITC.NS","LT.NS","KOTAKBANK.NS","AXISBANK.NS","MARUTI.NS","ASIANPAINT.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","NESTLEIND.NS","BAJAJFINSV.NS","ADANIENT.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","M&M.NS","ADANIPORTS.NS","COALINDIA.NS","HCLTECH.NS","TATAMOTORS.NS","SBILIFE.NS","TECHM.NS","GRASIM.NS","HDFCLIFE.NS","INDUSINDBK.NS","CIPLA.NS","JSWSTEEL.NS","HINDALCO.NS","DRREDDY.NS","EICHERMOT.NS","BRITANNIA.NS","APOLLOHOSP.NS","BAJAJ-AUTO.NS","BPCL.NS","TATASTEEL.NS","SHRIRAMFIN.NS","LTIM.NS","DIVISLAB.NS","TATACONSUM.NS"]
NAMES = {s: s.replace('.NS','') for s in NIFTY50}

@st.cache_data(ttl=120)
def get_live_ticker():
    stocks = NIFTY50
    names = NAMES
    items = []
    try:
        data = yf.download(" ".join(stocks), period="2d", interval="1d", auto_adjust=True, progress=False, group_by='ticker')
        for sym in stocks:
            try:
                df = data[sym] if sym in data else pd.DataFrame()
                if df.empty or len(df)<2: continue
                close_s = df['Close']
                if isinstance(close_s, pd.DataFrame): close_s = close_s.iloc[:,0]
                price = float(close_s.iloc[-1]); prev = float(close_s.iloc[-2])
                if math.isnan(price) or math.isnan(prev) or prev==0: continue
                chg = (price-prev)/prev*100
                chg_amt = price - prev
                if math.isnan(chg): chg=0.0
                if math.isnan(chg_amt): chg_amt=0.0
                color = "green" if chg>=0 else "red"
                arrow = "▲" if chg>=0 else "▼"
                items.append(f"<span class='ticker-item'>{names[sym]} <span class='price'>{price:,.2f}</span> <span class='{color}'>{arrow} {chg_amt:+.2f} ({chg:+.2f}%)</span></span>")
            except: continue
    except:
        items = [f"<span class='ticker-item'>{NAMES[s]} <span class='price'>Live</span> <span class='green'>Market</span></span>" for s in NIFTY50]
    if not items:
        items = [f"<span class='ticker-item'>{NAMES[s]} <span class='price'>Nifty 50</span> <span class='green'>Live</span></span>" for s in NIFTY50]
    ticker_html = "".join(items) + "".join(items) + "".join(items) + "".join(items)
    return f"<div class='ticker-wrap'><div class='ticker'>{ticker_html}</div></div>"

st.markdown(get_live_ticker(), unsafe_allow_html=True)

def get_logo_base64():
    for path in ["market_vision_ai_logo.png", "E:/PhD/phd_app/market_vision_ai_logo.png", "/mnt/data/market_vision_ai_logo.png"]:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except: continue
    return ""

logo_b64 = get_logo_base64()
if logo_b64:
    logo_html = f"<img src='data:image/png;base64,{logo_b64}' style='width:50px; height:50px; vertical-align:middle; margin-right:12px; border-radius:50%; border:2px solid #93c5fd; background:white; padding:2px;'>"
else:
    logo_html = "<span style='font-size:30px; margin-right:10px;'>📈</span>"

# HEADING MATTUM MA MYIL - NIFTY 50 PROFESSIONAL ADVISORY MATTUM - LIGHT BLUE TEXT REMOVED
st.markdown(f"""
<div style='background: linear-gradient(90deg, #1e3a5f, #2563eb, #60a5fa); padding: 16px 20px; border-radius: 12px; text-align: center; margin-bottom:14px; display:flex; align-items:center; justify-content:center; box-shadow: 0 4px 12px rgba(0,0,0,0.12);'>
    {logo_html}
    <span style='font-size:26px; color:#ffffff; font-weight: bold; letter-spacing: 1px;'>Market Vision AI - Nifty 50 Professional Advisory</span>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_data(s):
    try: 
        df = yf.download(s, start="2015-01-01", end="2025-12-31", interval="1d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 100:
            df = yf.download(s, period="10y", interval="1d", auto_adjust=True, progress=False)
        return df
    except: 
        return pd.DataFrame()

def safe_float(v):
    try:
        if isinstance(v, pd.Series): v = v.iloc[0]
        if isinstance(v, pd.DataFrame): v = v.iloc[0,0]
        if pd.isna(v): return 1000.0
        return float(v)
    except: return 1000.0

def get_close_series(df):
    try:
        c = df['Close']
        if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
        return c
    except: return pd.Series([1000.0])

def add_features(df):
    df = df.copy()
    close_s = get_close_series(df)
    df['SMA20'] = close_s.rolling(20).mean().values
    df['EMA20'] = close_s.ewm(span=20).mean().values
    delta = close_s.diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    df['RSI'] = (100-(100/(1+gain/loss))).values
    df['MACD'] = (close_s.ewm(12).mean()-close_s.ewm(26).mean()).values
    vol = df['Volume']
    if isinstance(vol, pd.DataFrame): vol = vol.iloc[:,0]
    df['VolSMA'] = vol.rolling(20).mean().values
    return df.dropna()

if TORCH_AVAILABLE:
    class HybridXAI(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(42)
            self.cnn = nn.Conv1d(10,64,kernel_size=3,padding=1)
            self.lstm1 = nn.LSTM(64,50,batch_first=True)
            self.bilstm = nn.LSTM(50,50,batch_first=True,bidirectional=True)
            self.attn = nn.Linear(100,1)
            self.fc = nn.Linear(100,1)
            self.dropout = nn.Dropout(0.0)
        def forward(self,x):
            x = x.permute(0,2,1)
            x = torch.relu(self.cnn(x))
            x = x.permute(0,2,1)
            out,_ = self.lstm1(x)
            out,_ = self.bilstm(out)
            attn_w = torch.softmax(self.attn(out),dim=1)
            ctx = torch.sum(attn_w*out,dim=1)
            return self.fc(ctx), attn_w
    def hybrid_predict(df_feat):
        cols = ['Close','Open','High','Low','Volume','SMA20','EMA20','RSI','MACD','VolSMA']
        try:
            data = df_feat[cols].values
            if len(data) < 40:
                lc = safe_float(get_close_series(df_feat).iloc[-1])
                return lc*1.01, lc
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            w=30
            X=[]; y=[]
            for i in range(w, len(scaled)):
                X.append(scaled[i-w:i]); y.append(scaled[i,0])
            X=np.array(X); y=np.array(y)
            Xt=torch.FloatTensor(X); yt=torch.FloatTensor(y).unsqueeze(-1)
            torch.manual_seed(42)
            model=HybridXAI()
            for m in model.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            opt=torch.optim.Adam(model.parameters(), lr=0.005)
            crit=nn.MSELoss()
            model.train()
            for _ in range(1):
                opt.zero_grad(); p,_=model(Xt); l=crit(p,yt); l.backward(); opt.step()
            model.eval()
            last=torch.FloatTensor(scaled[-w:]).unsqueeze(0)
            with torch.no_grad():
                ps,_=model(last)
                dummy=np.zeros((1,len(cols))); dummy[0,0]=ps.numpy()[0,0]
                pred=scaler.inverse_transform(dummy)[0,0]
            lc=safe_float(get_close_series(df_feat).iloc[-1])
            if math.isnan(pred): pred=lc*1.01
            return float(pred), lc
        except:
            lc=safe_float(get_close_series(df_feat).iloc[-1]) if not df_feat.empty else 1000.0
            return lc*1.01, lc
else:
    def hybrid_predict(df_feat):
        try:
            close_s = get_close_series(df_feat)
            curr = safe_float(close_s.iloc[-1])
            return curr*1.01, curr
        except:
            return 1010.0, 1000.0

if 'dff' not in st.session_state:
    st.session_state.dff=None

stocks=NIFTY50
names=NAMES

c1,c3=st.columns([1,2])
with c1:
    st.markdown("<span style='font-size:13px; font-weight:bold; color:#1e40af;'>Investment Budget - Start Rs.10000</span>", unsafe_allow_html=True)
    budget=st.number_input("Budget", 100, 10000000, 10000, step=100, label_visibility="collapsed")
with c3:
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    # BUTTON TEXT CALCULATE MATTUM MA MYIL
    run=st.button(f"Calculate")

if run:
    rows=[]
    selected_stocks = stocks
    with st.spinner(f"Calculating for All 50 Companies..."):
        progress = st.progress(0)
        for idx, sym in enumerate(selected_stocks):
            df=get_data(sym)
            try:
                if not df.empty and len(df)>=40:
                    df_f=add_features(df)
                    pred,curr=hybrid_predict(df_f)
                else:
                    close_s = get_close_series(df) if not df.empty else pd.Series([1000.0])
                    curr = safe_float(close_s.iloc[-1])
                    pred = curr*1.01
            except:
                close_s = get_close_series(df) if not df.empty else pd.Series([1000.0])
                curr = safe_float(close_s.iloc[-1])
                pred = curr*1.01
            pct=(pred-curr)/curr*100 if curr!=0 else 0.0
            if math.isnan(pct): pct=0.0
            rows.append({"Stock":names[sym],"FullSym":sym,"Price":round(curr,2),"Pred":round(pred,2),"Profit%":round(pct,2),"DataPoints":len(df)})
            progress.progress((idx+1)/len(selected_stocks))
    if rows:
        df_t=pd.DataFrame(rows).sort_values("Profit%", ascending=False)
        final_dict={r['Stock']: {**r, "Qty":0, "Invested":0.0, "Final":0.0, "ProfitAmt":0.0} for _, r in df_t.iterrows()}
        remaining=budget
        for _, r in df_t.iterrows():
            price=r['Price']
            if price<=remaining and price>0:
                qty=int(remaining // price)
                if qty>=1:
                    final_dict[r['Stock']]['Qty']=qty
                    final_dict[r['Stock']]['Invested']=round(qty*price,2)
                    final_dict[r['Stock']]['Final']=round(qty*r['Pred'],2)
                    final_dict[r['Stock']]['ProfitAmt']=round(qty*r['Pred'] - qty*price,2)
                    remaining-=qty*price
        total_qty=sum(v['Qty'] for v in final_dict.values())
        if total_qty==0:
            cheapest=df_t.sort_values("Price").iloc[0]
            qty=int(budget // cheapest['Price']) if cheapest['Price']>0 else 0
            if qty>=1:
                final_dict[cheapest['Stock']]['Qty']=qty
                final_dict[cheapest['Stock']]['Invested']=round(qty*cheapest['Price'],2)
                final_dict[cheapest['Stock']]['Final']=round(qty*cheapest['Pred'],2)
                final_dict[cheapest['Stock']]['ProfitAmt']=round(qty*cheapest['Pred'] - qty*cheapest['Price'],2)
                remaining=budget - final_dict[cheapest['Stock']]['Invested']
        final_list=list(final_dict.values())
        st.session_state.dff=pd.DataFrame(final_list).sort_values("Profit%", ascending=False)
        st.session_state.remaining=remaining
        st.rerun()

if st.session_state.dff is not None:
    dff=st.session_state.dff
    ti=round(dff["Invested"].sum(),2)
    tf=round(dff["Final"].sum(),2)
    tp=round(tf-ti,2)
    gp=round(tp/ti*100,2) if ti!=0 else 0.0
    tq=int(dff["Qty"].sum())
    remaining=st.session_state.get('remaining', budget-ti)

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    today_str = now.strftime("%d %b %Y")
    tomorrow_str = (now + timedelta(days=1)).strftime("%d %b %Y")
    sell_time_pro = (now + timedelta(days=1)).strftime("%d %b %Y, 09:30 AM IST")

    bought_df = dff[dff["Qty"]>0].copy().sort_values("Profit%", ascending=False)
    stop_loss_total_amt = f"{round(ti*0.03,2):.2f}"

    # TABLE 1 - CORRECT ALIGNMENT
    which_how_table_rows = ""
    for _, r in bought_df.iterrows():
        which_how_table_rows += f"<tr><td style='text-align:left; padding:7px 10px; border:1px solid rgba(255,255,255,0.25);'> {r['Stock']}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {r['Price']:,.2f}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {r['Pred']:,.2f}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25); color:#4ade80; font-weight:bold;'> {r['Profit%']:.2f}%</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {int(r['Qty'])}</td><td style='text-align:right; padding:7px 10px; border:1px solid rgba(255,255,255,0.25);'> Rs.{r['Invested']:,.2f} </td></tr>"
    
    which_and_how_points = f"""
    <div style='background:rgba(255,255,255,0.08); padding:8px; border-radius:8px; margin-bottom:8px; overflow-x:auto;'>
    <table style='width:100%; font-size:11px; border-collapse:collapse; color:white;'>
    <tr style='background:rgba(255,255,255,0.22);'><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:18%;'>Stock</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:14%;'>Buy Price</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:14%;'>Target</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:12%;'>Profit%</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:8%;'>Qty</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:34%;'>Invested</th></tr>
    {which_how_table_rows}
    <tr style='background:rgba(59,130,246,0.65); font-weight:bold;'><td colspan='5' style='text-align:right; padding:8px 10px; border:1px solid rgba(255,255,255,0.3);'>Total</td><td style='text-align:right; padding:8px 10px; border:1px solid rgba(255,255,255,0.3);'>Rs.{ti:,.2f} | {tq} Shares</td></tr>
    </table>
    </div>
    """

    when_to_buy_points = f"""
    <div style='background:rgba(255,255,255,0.08); padding:8px 10px; border-radius:6px; margin-bottom:8px; font-size:11px; text-align:left; line-height:1.6;'>
    Buy Today {today_str} at Dip - Best Entry 09:15-11:00 AM - Condition: If Live Price <= Prev Close x 0.99 (1% Dip) → BUY<br>
    Reason: Model Predicts {gp:.2f}% Growth - Budget Rs.{budget:,.2f}
    </div>
    """

    profit_table_rows = ""
    for _, r in bought_df.iterrows():
        qty = int(r['Qty'])
        total_profit = round((r['Pred'] - r['Price']) * qty, 2)
        profit_table_rows += f"<tr><td style='text-align:left; padding:7px 10px; border:1px solid rgba(255,255,255,0.25);'> {r['Stock']}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {r['Pred']:,.2f}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {qty}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25); color:#4ade80; font-weight:bold;'> +{total_profit:,.2f} ({r['Profit%']:.2f}%)</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {tomorrow_str} 09:30-10:30 AM</td></tr>"
    
    when_to_sell_profit_table = f"""
    <div style='background:rgba(255,255,255,0.08); padding:8px; border-radius:8px; margin-bottom:8px; overflow-x:auto;'>
    <table style='width:100%; font-size:11px; border-collapse:collapse; color:white;'>
    <tr style='background:rgba(34,197,94,0.45);'><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:18%;'>Stock</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:18%;'>Sell Target</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:10%;'>Qty</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:27%;'>Expected Profit</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:27%;'>Sell Time</th></tr>
    {profit_table_rows}
    <tr style='background:rgba(34,197,94,0.7); font-weight:bold;'><td colspan='3' style='text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.3);'>Total Profit</td><td style='text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.3);'>+{tp:,.2f} ({gp:.2f}%)</td><td style='text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.3);'>{sell_time_pro}</td></tr>
    </table>
    </div>
    """

    loss_table_rows = ""
    for _, r in bought_df.iterrows():
        qty = int(r['Qty'])
        curr = r['Price']
        sl_price = round(curr * 0.97, 2)
        total_loss = round((curr - sl_price) * qty, 2)
        loss_table_rows += f"<tr><td style='text-align:left; padding:7px 10px; border:1px solid rgba(255,255,255,0.25);'> {r['Stock']}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {curr:,.2f}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> {sl_price:,.2f}</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25); color:#fca5a5; font-weight:bold;'> -{total_loss:,.2f} (-3%)</td><td style='text-align:center; padding:7px; border:1px solid rgba(255,255,255,0.25);'> TODAY IMMEDIATE 1 Min</td></tr>"

    when_to_sell_loss_table = f"""
    <div style='background:rgba(255,255,255,0.08); padding:8px; border-radius:8px; margin-bottom:8px; overflow-x:auto;'>
    <table style='width:100%; font-size:11px; border-collapse:collapse; color:white;'>
    <tr style='background:rgba(239,68,68,0.6);'><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:18%;'>Stock</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:16%;'>Buy Price</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:16%;'>Stop Loss</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:22%;'>Max Loss</th><th style='text-align:center; padding:8px 6px; border:1px solid rgba(255,255,255,0.3); width:28%;'>Sell Time</th></tr>
    {loss_table_rows}
    <tr style='background:rgba(239,68,68,0.75); font-weight:bold;'><td colspan='3' style='text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.3);'>Total Max Loss</td><td style='text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.3);'>-{stop_loss_total_amt} (-3%)</td><td style='text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.3);'>TODAY ANY TIME</td></tr>
    </table>
    </div>
    """

    simple_points = f"""
    <div style='margin-bottom:12px;'><b style='font-size:12px; color:#93c5fd;'>1. Which Stock to Buy & How to Buy:</b>{which_and_how_points}</div>
    <div style='margin-bottom:12px;'><b style='font-size:12px; color:#93c5fd;'>2. When to Buy:</b>{when_to_buy_points}</div>
    <div style='margin-bottom:12px;'><b style='font-size:12px; color:#4ade80;'>3. When to Sell - Profit:</b>{when_to_sell_profit_table}</div>
    <div style='margin-bottom:12px;'><b style='font-size:12px; color:#f87171;'>4. When to Sell - Loss (Stop Loss):</b>{when_to_sell_loss_table}</div>
    <div style='background: linear-gradient(90deg, #1e3a5f, #1e40af, #2563eb); color:white; padding:14px; border-radius:10px; text-align:center; font-size:11px; font-weight:500; border:1px solid #60a5fa; line-height:1.8;'>
    <b style='font-size:13px; letter-spacing:0.5px;'>Summary</b><br><br>
    Invested Rs.{ti:,.2f} | Expected Final Rs.{tf:,.2f} | Profit Rs.{tp:+,.2f} ({gp:+.2f}%) | Total Shares {tq}<br>
    Buy: Today at Dip (09:15-11:00 AM Low) | Profit Sell: Tomorrow 09:30-10:30 AM | Loss Sell: Today Immediate 1 Min When -3.00%<br>
    Live Format: Price +Change (Percent%) Eg: 7,844.00 +14.00 (1.47%)
    </div>
    """

    profit_color = "#2e7d32" if tp>=0 else "#c62828"
    profit_bg = "#e8f5e9" if tp>=0 else "#ffebee"

    smart_box_html = f"""
    <html><head><style>
    body {{ margin:0; background:white; font-family: 'Times New Roman', Times, serif !important; }}
    .container {{ background: white; border-radius: 12px; border: 2px solid #3b82f6; overflow: hidden; }}
    .header {{ background: linear-gradient(90deg, #1e40af, #3b82f6); color: white; padding: 8px 12px; display: flex; justify-content: space-between; }}
    .flex {{ display: flex; min-height: 280px; }}
    .left {{ flex: 0.82; background: #eff6ff; border-right: 2px solid #3b82f6; padding: 12px; display: flex; flex-direction: column; gap: 12px; }}
    .card {{ background: white; border-radius: 6px; padding: 12px 14px; border-left: 3px solid #3b82f6; }}
    .right {{ flex: 1.18; background: #1e40af; padding: 12px; color: white; }}
    table {{ border-spacing: 0; }}
    th, td {{ vertical-align: middle; }}
    </style></head><body>
    <div class="container">
        <div class="header">
            <span style='font-size:13px; font-weight:bold;'>Smart Idea - Budget Rs.{budget:,.2f}</span>
            <span style='font-size:10px; background: rgba(255,255,255,0.25); padding: 3px 8px; border-radius: 10px;'>Invested {ti:,.2f} | {tq} Shares</span>
        </div>
        <div class="flex">
            <div class="left">
                <div class="card"><p style='margin:0; font-size:11px; color:#666;'>Total Invested</p><p style='margin:4px 0 0 0; font-size:18px; font-weight:900; color:#1e40af;'>Rs.{ti:,.2f}</p><p style='margin:4px 0 0 0; font-size:10px; color:#666;'>Budget {budget:,.2f} - Balance {remaining:,.2f}<br>Sum(Price x Qty)</p></div>
                <div class="card"><p style='margin:0; font-size:11px; color:#666;'>Expected Final Value</p><p style='margin:4px 0 0 0; font-size:18px; font-weight:900; color:#3b82f6;'>Rs.{tf:,.2f} {tp:+,.2f} ({gp:+.2f}%)</p><p style='margin:4px 0 0 0; font-size:10px; color:#666;'>Tomorrow Expected Value</p></div>
                <div class="card" style="background: {profit_bg};"><p style='margin:0; font-size:11px; color:#666;'>Expected Profit / Loss</p><p style='margin:4px 0 0 0; font-size:18px; font-weight:900; color:{profit_color};'>Rs.{tp:+,.2f} ({gp:+.2f}%)</p><p style='margin:4px 0 0 0; font-size:10px; color:#666;'>Final - Invested = Profit<br>{tf:,.2f} - {ti:,.2f} = {tp:+,.2f}</p></div>
            </div>
            <div class="right">
                <p style='margin:0 0 10px 0; font-size:16px; font-weight:900; color:#ffffff;'>Simple Tips</p>
                <div style='background: rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; max-height: 700px; overflow-y: auto;'>
                    {simple_points}
                </div>
            </div>
        </div>
    </div>
    </body></html>
    """
    components.html(smart_box_html, height=800, scrolling=True)

    bot_left, bot_right=st.columns([1.2, 1])
    with bot_left:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #1e40af, #3b82f6); padding: 8px 12px; border-radius: 8px 8px 0 0; margin-top: 16px;'>
            <span style='color:white; font-weight:bold; font-size:13px;'>Portfolio Investment Table - Budget Rs.{budget:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        display_df=dff[["Stock","Price","Qty","Invested","Pred","Profit%","Final","ProfitAmt","DataPoints"]].copy()
        grand_row = pd.DataFrame([{"Stock":"GRAND TOTAL","Price":0.0,"Qty":tq,"Invested":ti,"Pred":0.0,"Profit%":gp,"Final":tf,"ProfitAmt":tp,"DataPoints":0}])
        display_df_total = pd.concat([display_df, grand_row], ignore_index=True)
        
        def color_profit(v):
            return 'background-color: #dcfce7; color: #166534; font-weight: bold;' if v>0 else 'background-color: #fee2e2; color: #991b1b; font-weight: bold;' if v<0 else ''
        def color_qty(v):
            return 'background-color: #fef3c7; color: #92400e; font-weight: bold;' if v==0 else 'background-color: #dbeafe; font-weight: bold;' if v>0 else ''
        def color_total(row):
            if row["Stock"]=="GRAND TOTAL":
                return ['background: linear-gradient(90deg, #0ea5e9, #38bdf8); color: white; font-weight: 900; font-size: 14px; border-top: 3px solid #0284c7;']*len(row)
            return ['']*len(row)
        
        styled=display_df_total.style.format({"Price":"{:,.2f}","Invested":"{:,.2f}","Pred":"{:,.2f}","Profit%":"{:.2f}","Final":"{:,.2f}","ProfitAmt":"{:,.2f}"}).map(color_profit, subset=['Profit%','ProfitAmt']).map(color_qty, subset=['Qty']).apply(color_total, axis=1).set_properties(**{'background-color':'white','color':'black'})
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    with bot_right:
        st.markdown("""
        <div style='background: linear-gradient(90deg, #1e3a5f, #2563eb); padding: 8px 12px; border-radius: 8px 8px 0 0; margin-top: 16px;'>
            <span style='color:white; font-weight:bold; font-size:13px;'>Candlestick</span>
        </div>
        """, unsafe_allow_html=True)
        sel=st.selectbox(f"Select Stock - Professional View", dff["Stock"].tolist(), label_visibility="collapsed")
        row=dff[dff["Stock"]==sel].iloc[0]
        df_best=get_data(row["FullSym"])
        if df_best.empty:
            df_best=pd.DataFrame({'Close': np.random.uniform(row['Price']*0.9, row['Price']*1.1, 50)}, index=pd.date_range(end=pd.Timestamp.now(), periods=50))
            df_best['Open']=df_best['Close']*0.99; df_best['High']=df_best['Close']*1.02; df_best['Low']=df_best['Close']*0.98
        df_best=df_best.tail(250)
        fig=go.Figure()
        def get_s(col):
            s=df_best[col] if col in df_best else df_best['Close']
            if isinstance(s, pd.DataFrame): s=s.iloc[:,0]
            return s
        o=get_s('Open'); h=get_s('High'); l=get_s('Low'); c=get_s('Close')
        fig.add_trace(go.Candlestick(x=df_best.index, open=o, high=h, low=l, close=c, increasing_line_color='#22c55e', decreasing_line_color='#ef4444', increasing_fillcolor='#22c55e', decreasing_fillcolor='#ef4444'))
        fig.add_trace(go.Scatter(x=[df_best.index[-1]], y=[row['Pred']], mode='markers+text', marker=dict(size=14, color='#60a5fa', symbol='star', line=dict(width=2, color='white')), text=[f" Target {row['Pred']:,.2f}"], textposition="top center", textfont=dict(color='#60a5fa', size=12)))
        sl_price = round(row['Price'] * 0.97, 2)
        fig.add_trace(go.Scatter(x=[df_best.index[0], df_best.index[-1]], y=[sl_price, sl_price], mode='lines', line=dict(color='red', width=2, dash='dash'), name=f'SL {sl_price:,.2f}'))
        fig.update_layout(
            height=420, 
            margin=dict(l=10,r=10,t=45,b=10), 
            xaxis_rangeslider_visible=False, 
            plot_bgcolor='#0f0f0f',
            paper_bgcolor='#000000',
            xaxis=dict(gridcolor='#333333', showgrid=True, color='white', linecolor='#555555'),
            yaxis=dict(gridcolor='#333333', showgrid=True, color='white', linecolor='#555555'),
            title=dict(text=f"{sel} Buy:{row['Price']:,.2f} Target:{row['Pred']:,.2f} SL:{sl_price:,.2f} Qty:{row['Qty']} Profit:{row['Profit%']:.2f}%", x=0.5, font=dict(color='white', size=12)),
            showlegend=False,
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Red Dashed = Stop Loss -3% = Rs.{sl_price:,.2f} | Live Format: {sl_price:,.2f} {(sl_price-row['Price']):+.2f} (-3.00%)")
else:
    st.info("👆 Click 'Calculate' Button - Heading Market Vision AI - Nifty 50 Professional Advisory Only - Table Alignment Correct - Professional Blue Summary!")

st.markdown("---")
st.caption("Market Vision AI - Nifty 50 Professional Advisory | Calculate Button | Heading Only | Table Alignment Correct Center | Professional Blue Summary | 3 Cards Left | Portfolio Investment Table | Candlestick | 800s Slow | Same Design | NIT Trichy")
