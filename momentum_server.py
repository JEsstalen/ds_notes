"""
美股多策略排序工具 · 本地 Web 服务
策略: 动量 / 动量+质量 / 低波动 / Piotroski F-Score
依赖: pip install flask yfinance pandas
启动: python momentum_server.py  →  http://localhost:5001
"""

import json, calendar, warnings, threading, os, math
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
from flask import Flask, Response, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
app.config["JSON_AS_ASCII"] = False

# ─── SSE ──────────────────────────────────────────────────────────────────────

_sse_queues: list = []
_sse_lock = threading.Lock()

def _broadcast(event: str, data: object):
    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    msg = f"event: {event}\ndata: {json.dumps(clean(data), ensure_ascii=False)}\n\n"
    with _sse_lock:
        dead = [q for q in _sse_queues if not _try_put(q, msg)]
        for q in dead:
            _sse_queues.remove(q)

def _try_put(q, msg):
    try: q.put_nowait(msg); return True
    except: return False

def log(text: str, pct: int = -1):
    _broadcast("log", {"msg": text, "pct": pct})

# ─── Wikipedia 股票池 ──────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _read_html_wiki(url):
    import requests, pandas as pd
    from io import StringIO
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))

def fetch_sp500():
    import pandas as pd
    log("获取 S&P 500 成分股…", 5)
    df = _read_html_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df = df[["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["Ticker", "Name", "Sector"]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df["Index"] = "S&P 500"
    return df.dropna(subset=["Ticker"])

def fetch_nasdaq100():
    import pandas as pd
    log("获取 NASDAQ-100 成分股…", 5)
    for t in _read_html_wiki("https://en.wikipedia.org/wiki/Nasdaq-100"):
        cols = list(t.columns)
        low  = [str(c).lower() for c in cols]
        tc = next((cols[i] for i,c in enumerate(low) if c in ("ticker","symbol")), None)
        if not tc: continue
        nc = next((cols[i] for i,c in enumerate(low) if any(k in c for k in ("company","security","name"))), None)
        sc = next((cols[i] for i,c in enumerate(low) if "sector" in c), None)
        r = pd.DataFrame({
            "Ticker": t[tc].astype(str).str.strip().str.replace(".","-",regex=False),
            "Name":   t[nc].astype(str).str.strip() if nc else "",
            "Sector": t[sc].astype(str).str.strip() if sc else "N/A",
            "Index":  "NASDAQ-100",
        })
        r = r[r["Ticker"].str.match(r"^[A-Z]")]
        if len(r) > 50: return r.reset_index(drop=True)
    raise ValueError("无法解析 NASDAQ-100 页面")

def build_universe(universe, sector_filter):
    import pandas as pd
    if universe == "sp500":       meta = fetch_sp500()
    elif universe == "nasdaq100": meta = fetch_nasdaq100()
    else:
        meta = pd.concat([fetch_sp500(), fetch_nasdaq100()], ignore_index=True)
        meta = (meta.groupby("Ticker")
                .agg(Name=("Name","first"), Sector=("Sector","first"),
                     Index=("Index", lambda x: " & ".join(sorted(set(x)))))
                .reset_index())
    if sector_filter and sector_filter != "all":
        meta = meta[meta["Sector"].str.lower().str.contains(sector_filter.lower())]
    log(f"股票池就绪：{len(meta)} 只", 10)
    return meta.reset_index(drop=True)

# ─── 价格下载 ──────────────────────────────────────────────────────────────────

def load_prices(tickers, start, end):
    import pandas as pd, yfinance as yf
    BATCH, frames, total = 100, [], len(tickers)
    for i in range(0, total, BATCH):
        batch = tickers[i:i+BATCH]
        log(f"下载行情 {min(i+BATCH,total)}/{total}…", 10+int(min(i+BATCH,total)/total*45))
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns else raw
        if close is None or (hasattr(close,"empty") and close.empty): continue
        if hasattr(close,"ndim") and close.ndim==1: close = close.to_frame(name=batch[0])
        frames.append(close)
    if not frames: raise RuntimeError("行情下载失败，请检查网络")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.dropna(how="all")

# ─── 全局基本面缓存 ────────────────────────────────────────────────────────────
# key: ticker  value: dict of raw fields
# 缓存有效期内（同一服务进程）切换策略不重复拉取

import time as _time
_info_cache: dict = {}          # ticker -> (ts, data)
_pio_cache:  dict = {}          # ticker -> (ts, score, detail)
_CACHE_TTL = 3600               # 1小时过期

def _cache_get_info(ticker):
    entry = _info_cache.get(ticker)
    if entry and _time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None

def _cache_set_info(ticker, data):
    _info_cache[ticker] = (_time.time(), data)

def _cache_get_pio(ticker):
    entry = _pio_cache.get(ticker)
    if entry and _time.time() - entry[0] < _CACHE_TTL:
        return entry[1], entry[2]
    return None, None

def _cache_set_pio(ticker, score, detail):
    _pio_cache[ticker] = (_time.time(), score, detail)

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None: return None
    if isinstance(v, float) and (v != v or abs(v) == float("inf")): return None
    return v

def _monthly_stats(prices, ticker):
    """返回月度动量相关指标，失败返回 None"""
    import numpy as np
    s = prices[ticker].resample("ME").last().dropna()
    if len(s) < 3: return None
    mr = s.pct_change().dropna()
    mom_mean = mr.mean()
    mom_std  = mr.std()
    sharpe   = mom_mean/mom_std if mom_std>1e-9 else float("nan")
    cum = (1+mr).cumprod()
    max_dd = ((cum-cum.cummax())/cum.cummax()).min()
    return {
        "price":     round(float(prices[ticker].dropna().iloc[-1]), 2),
        "total_ret": round(float(s.iloc[-1]/s.iloc[0]-1)*100, 2),
        "mom_mean":  round(float(mom_mean)*100, 3),
        "sharpe":    round(float(sharpe),3) if not (sharpe!=sharpe) else None,
        "max_dd":    round(float(max_dd)*100, 2),
    }

def _meta_info(meta_idx, ticker):
    info = meta_idx.loc[ticker] if ticker in meta_idx.index else {}
    return {
        "name":   str(info.get("Name","")),
        "sector": str(info.get("Sector","N/A")),
        "index":  str(info.get("Index","")),
    }

# ─── 策略 1：纯动量 ────────────────────────────────────────────────────────────

def calc_momentum(prices, meta, start_str, end_str):
    import pandas as pd
    meta_idx = meta.set_index("Ticker")
    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx%50==0: log(f"计算动量 {idx}/{total}…", 58+int(idx/total*35))
        ps = _monthly_stats(prices, ticker)
        if ps is None: continue
        results.append({**_meta_info(meta_idx, ticker), "ticker": ticker, **ps})
    df = pd.DataFrame(results).sort_values("mom_mean", ascending=False).reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 2：动量 + 质量 ───────────────────────────────────────────────────────

def _fetch_info(ticker, fields):
    """拉取 yfinance .info，命中缓存直接返回；单次请求超时 8 秒"""
    cached = _cache_get_info(ticker)
    if cached is not None:
        return ticker, {k: cached.get(k) for k in fields}
    try:
        import yfinance as yf
        import threading as _th
        result = [None]
        def _do():
            try: result[0] = yf.Ticker(ticker).info
            except: pass
        t = _th.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout=8)
        if result[0] is None:
            return ticker, {k: None for k in fields}
        data = {k: _safe(result[0].get(k)) for k in (
            "returnOnEquity","profitMargins","priceToSalesTrailing12Months","beta",
            "trailingPE","forwardPE","priceToBook","enterpriseToEbitda","marketCap",
            "revenueGrowth","earningsGrowth","debtToEquity","dividendYield",
            "fiftyTwoWeekHigh","fiftyTwoWeekLow","targetMeanPrice","recommendationKey",
            "shortName","sector","industry",
        )}
        _cache_set_info(ticker, data)
        return ticker, {k: data.get(k) for k in fields}
    except:
        return ticker, {k: None for k in fields}

def _parallel_info(tickers, fields, log_prefix, pct_start, pct_end, workers=20):
    """并发拉取一批 ticker 的 info 字段，实时推进度，返回 dict[ticker->dict]"""
    results = {}
    done = [0]
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_info, t, fields): t for t in tickers}
        for f in as_completed(futs):
            done[0] += 1
            pct = pct_start + int(done[0] / total * (pct_end - pct_start))
            log(f"{log_prefix} {done[0]}/{total}…", pct)
            try:
                t, d = f.result()
                results[t] = d
            except:
                pass
    return results

def calc_momentum_quality(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np
    meta_idx = meta.set_index("Ticker")

    price_stats = {}
    total = len(prices.columns)
    for idx, t in enumerate(prices.columns):
        if idx % 50 == 0: log(f"计算动量 {idx}/{total}…", 58 + int(idx/total*12))
        ps = _monthly_stats(prices, t)
        if ps: price_stats[t] = ps

    valid = list(price_stats.keys())
    FIELDS = ["returnOnEquity", "profitMargins"]
    fundamentals = _parallel_info(valid, FIELDS, "基本面", 71, 95, workers=20)

    rows = []
    for t in valid:
        fs = fundamentals.get(t, {})
        roe = round(fs["returnOnEquity"]*100,1) if fs.get("returnOnEquity") is not None else None
        pm  = round(fs["profitMargins"]*100,1)  if fs.get("profitMargins")  is not None else None
        rows.append({**_meta_info(meta_idx, t), "ticker": t,
                     **price_stats[t], "roe": roe, "profit_margin": pm})

    df = pd.DataFrame(rows)
    def norm(col):
        s = df[col].dropna(); mn,mx = s.min(),s.max()
        if mx==mn: return pd.Series(0.5, index=df.index)
        return (df[col].fillna(mn)-mn)/(mx-mn)
    df["composite"] = (0.5*norm("mom_mean") + 0.3*norm("roe") + 0.2*norm("profit_margin")).round(3)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 3：低波动率 ──────────────────────────────────────────────────────────

def calc_low_vol(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np, yfinance as yf
    meta_idx = meta.set_index("Ticker")

    log("下载 SPY 基准…", 58)
    spy_raw = yf.download("SPY", start=start_str, end=end_str, auto_adjust=True, progress=False)
    spy_ret = spy_raw["Close"].pct_change().dropna() if not spy_raw.empty else pd.Series(dtype=float)

    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx%50==0: log(f"计算波动率 {idx}/{total}…", 60+int(idx/total*35))
        ps = _monthly_stats(prices, ticker)
        if ps is None: continue

        dr = prices[ticker].pct_change().dropna()
        ann_vol = round(float(dr.std()*np.sqrt(252)*100), 2)

        beta = None
        if not spy_ret.empty:
            aln = pd.concat([dr, spy_ret], axis=1).dropna()
            aln.columns = ["s","m"]
            if len(aln) >= 20:
                varm = aln["m"].var()
                if varm > 1e-12:
                    beta = round(float(aln.cov().iloc[0,1]/varm), 3)

        results.append({**_meta_info(meta_idx, ticker), "ticker": ticker,
                        **ps, "beta": beta, "ann_vol": ann_vol})

    df = pd.DataFrame(results)
    df = df.sort_values(["beta","ann_vol"], ascending=[True,True], na_position="last").reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 策略 4：Piotroski F-Score ─────────────────────────────────────────────────

def _piotroski_one(ticker):
    import yfinance as yf, numpy as np

    # 命中缓存直接返回
    cached_score, cached_detail = _cache_get_pio(ticker)
    if cached_score is not None or cached_detail is not None:
        return ticker, cached_score, cached_detail or {}

    def get(df, keys, col=0):
        for k in (keys if isinstance(keys,list) else [keys]):
            if k in df.index:
                v = df.loc[k].iloc[col] if col < df.shape[1] else np.nan
                if v is not None and not (isinstance(v,float) and np.isnan(v)):
                    return float(v)
        return np.nan

    def sdiv(a, b):
        if np.isnan(a) or np.isnan(b) or b==0: return np.nan
        return a/b

    def sig(cond):
        try: return 1 if bool(cond) else 0
        except: return 0

    try:
        t = yf.Ticker(ticker)
        # 三张表并发拉取，每张最多等 10 秒
        with ThreadPoolExecutor(max_workers=3) as ex:
            fi = ex.submit(lambda: t.income_stmt)
            fb = ex.submit(lambda: t.balance_sheet)
            fc = ex.submit(lambda: t.cashflow)
            inc = fi.result(timeout=10)
            bal = fb.result(timeout=10)
            cf  = fc.result(timeout=10)

        if any(x is None or (hasattr(x,"empty") and x.empty) for x in [inc,bal,cf]):
            _cache_set_pio(ticker, None, {})
            return ticker, None, {}
        if inc.shape[1]<2 or bal.shape[1]<2:
            _cache_set_pio(ticker, None, {})
            return ticker, None, {}

        ni0  = get(inc, ["Net Income","NetIncome"], 0)
        ni1  = get(inc, ["Net Income","NetIncome"], 1)
        ta0  = get(bal, ["Total Assets","TotalAssets"], 0)
        ta1  = get(bal, ["Total Assets","TotalAssets"], 1)
        ocf0 = get(cf,  ["Operating Cash Flow","OperatingCashFlow"], 0)
        ltd0 = get(bal, ["Long Term Debt","LongTermDebt"], 0) if any(k in bal.index for k in ["Long Term Debt","LongTermDebt"]) else 0.0
        ltd1 = get(bal, ["Long Term Debt","LongTermDebt"], 1) if any(k in bal.index for k in ["Long Term Debt","LongTermDebt"]) else 0.0
        ca0  = get(bal, ["Current Assets","CurrentAssets"], 0)
        ca1  = get(bal, ["Current Assets","CurrentAssets"], 1)
        cl0  = get(bal, ["Current Liabilities","CurrentLiabilities"], 0)
        cl1  = get(bal, ["Current Liabilities","CurrentLiabilities"], 1)
        sh0  = get(bal, ["Ordinary Shares Number","Share Issued","CommonStock"], 0)
        sh1  = get(bal, ["Ordinary Shares Number","Share Issued","CommonStock"], 1)
        gp0  = get(inc, ["Gross Profit","GrossProfit"], 0)
        gp1  = get(inc, ["Gross Profit","GrossProfit"], 1)
        rv0  = get(inc, ["Total Revenue","Revenue"], 0)
        rv1  = get(inc, ["Total Revenue","Revenue"], 1)

        roa0=sdiv(ni0,ta0); roa1=sdiv(ni1,ta1); ocfa=sdiv(ocf0,ta0)
        cr0=sdiv(ca0,cl0);  cr1=sdiv(ca1,cl1)
        lev0=sdiv(ltd0,ta0);lev1=sdiv(ltd1,ta1)
        gm0=sdiv(gp0,rv0);  gm1=sdiv(gp1,rv1)
        at0=sdiv(rv0,ta0);  at1=sdiv(rv1,ta1)

        f1=sig(roa0>0); f2=sig(ocfa>0)
        f3=sig(not np.isnan(roa0) and not np.isnan(roa1) and roa0>roa1)
        f4=sig(not np.isnan(ocfa) and not np.isnan(roa0) and ocfa>roa0)
        f5=sig(not np.isnan(lev0) and not np.isnan(lev1) and lev0<lev1)
        f6=sig(not np.isnan(cr0)  and not np.isnan(cr1)  and cr0>cr1)
        f7=sig(not np.isnan(sh0)  and not np.isnan(sh1)  and sh0<=sh1)
        f8=sig(not np.isnan(gm0)  and not np.isnan(gm1)  and gm0>gm1)
        f9=sig(not np.isnan(at0)  and not np.isnan(at1)  and at0>at1)

        score  = f1+f2+f3+f4+f5+f6+f7+f8+f9
        detail = {
            "f_profit":    f1+f2+f3+f4,
            "f_leverage":  f5+f6+f7,
            "f_efficiency":f8+f9,
            "roa":         round(roa0*100,2) if not np.isnan(roa0) else None,
            "gross_margin":round(gm0*100,1)  if not np.isnan(gm0)  else None,
        }
        _cache_set_pio(ticker, score, detail)
        return ticker, score, detail
    except Exception:
        _cache_set_pio(ticker, None, {})
        return ticker, None, {}

def calc_piotroski(prices, meta, start_str, end_str):
    import pandas as pd
    meta_idx = meta.set_index("Ticker")
    tickers  = list(prices.columns)
    total    = len(tickers)
    log(f"获取财务报表（{total} 只，并行中…）", 58)

    pio_data = {}
    done = [0]
    # workers=12：每只内部再起3线程，实际并发约36，不过雅虎有限流
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_piotroski_one, t): t for t in tickers}
        for f in as_completed(futs):
            done[0] += 1
            pct = 58 + int(done[0] / total * 38)
            log(f"F-Score {done[0]}/{total}…", pct)
            try:
                t, score, detail = f.result()
                pio_data[t] = (score, detail)
            except: pass

    results = []
    for ticker in tickers:
        s = prices[ticker].dropna()
        if s.empty: continue
        score, detail = pio_data.get(ticker, (None, {}))
        monthly = prices[ticker].resample("ME").last().dropna()
        total_ret = round(float(monthly.iloc[-1]/monthly.iloc[0]-1)*100,2) if len(monthly)>=2 else None
        results.append({
            **_meta_info(meta_idx, ticker),
            "ticker":       ticker,
            "price":        round(float(s.iloc[-1]),2),
            "fscore":       score,
            "f_profit":     detail.get("f_profit"),
            "f_leverage":   detail.get("f_leverage"),
            "f_efficiency": detail.get("f_efficiency"),
            "roa":          detail.get("roa"),
            "gross_margin": detail.get("gross_margin"),
            "total_ret":    total_ret,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("fscore", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df

# ─── 日期解析 ──────────────────────────────────────────────────────────────────

def resolve_dates(start_ym, end_ym, months):
    def mlast(ym):
        dt = datetime.strptime(ym,"%Y-%m")
        return dt.replace(day=calendar.monthrange(dt.year,dt.month)[1]).date()
    def mfirst(ym): return datetime.strptime(ym,"%Y-%m").date()
    def subm(d,n):
        m,y = d.month-n, d.year
        while m<=0: m+=12; y-=1
        return d.replace(year=y,month=m,day=1)
    end_dt   = mlast(end_ym)   if end_ym   else date.today()
    start_dt = mfirst(start_ym) if start_ym else subm(end_dt, months)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

# ─── 策略 5：双动量（Dual Momentum）──────────────────────────────────────────
# 第一层：绝对动量 — 收益率 < 无风险利率（3M T-Bill）的股票排除
# 第二层：剩余股票按相对动量排序
# 无风险利率用 ^IRX（13周国库券年化收益率）近似

def calc_dual_momentum(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np, yfinance as yf

    meta_idx = meta.set_index("Ticker")

    # 拉取无风险利率（^IRX = 13-week T-bill annualized %）
    log("获取无风险利率（^IRX）…", 58)
    try:
        irx = yf.download("^IRX", start=start_str, end=end_str,
                          auto_adjust=True, progress=False)["Close"].dropna()
        # IRX 是年化百分比，换算到区间总收益率
        days = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days or 1
        rf_total = float(irx.mean()) / 100 * days / 365
    except Exception:
        rf_total = 0.02  # 无法拉取时用 2% 兜底

    log(f"无风险利率（区间累计）= {rf_total*100:.2f}%，计算双动量…", 62)

    results, total = [], len(prices.columns)
    for idx, ticker in enumerate(prices.columns):
        if idx % 30 == 0:
            log(f"计算双动量 {idx}/{total}…", 62 + int(idx / total * 33))
        ps = _monthly_stats(prices, ticker)
        if ps is None:
            continue

        # 绝对动量过滤：期间总收益 > 无风险利率才保留
        abs_pass = (ps["total_ret"] / 100) > rf_total
        results.append({
            **_meta_info(meta_idx, ticker),
            "ticker":    ticker,
            **ps,
            "rf_hurdle": round(rf_total * 100, 2),
            "abs_pass":  bool(abs_pass),
        })

    df = pd.DataFrame(results)
    # 绝对动量未通过的排到最后，通过的内部按相对动量排序
    df["_sort"] = df["mom_mean"].where(df["abs_pass"], other=-9999)
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    log("计算完成", 100)
    return df


# ─── 策略 6：多因子 Z-Score 合成 ──────────────────────────────────────────────
# 动量(35%) + 质量-ROE(25%) + 质量-利润率(15%) + 低波动-Beta(15%) + 价值-PS(10%)
# 每个因子先标准化为 Z-Score，再加权合成，最终按综合 Z 降序排列

# ─── 策略 6：多因子 Z-Score 合成 ──────────────────────────────────────────────

def calc_multifactor(prices, meta, start_str, end_str):
    import pandas as pd, numpy as np

    meta_idx = meta.set_index("Ticker")

    price_stats = {}
    total = len(prices.columns)
    for idx, t in enumerate(prices.columns):
        if idx % 30 == 0:
            log(f"计算价格因子 {idx}/{total}…", 58 + int(idx/total*12))
        ps = _monthly_stats(prices, t)
        if not ps: continue
        dr = prices[t].pct_change().dropna()
        price_stats[t] = {**ps, "ann_vol": round(float(dr.std()*np.sqrt(252)*100), 2)}

    valid  = list(price_stats.keys())
    FIELDS = ["returnOnEquity","profitMargins","priceToSalesTrailing12Months","beta"]
    fundamentals = _parallel_info(valid, FIELDS, "基本面", 71, 95, workers=20)

    rows = []
    for t in valid:
        fs = fundamentals.get(t, {})
        rows.append({
            **_meta_info(meta_idx, t), "ticker": t,
            **price_stats[t],
            "roe":           round(fs["returnOnEquity"]*100, 2)           if fs.get("returnOnEquity")                is not None else None,
            "profit_margin": round(fs["profitMargins"]*100, 2)            if fs.get("profitMargins")                 is not None else None,
            "ps":            round(fs["priceToSalesTrailing12Months"], 2) if fs.get("priceToSalesTrailing12Months")  is not None else None,
            "beta":          round(fs["beta"], 3)                         if fs.get("beta")                          is not None else None,
        })

    df = pd.DataFrame(rows)

    def zscore(col, invert=False):
        s = df[col].dropna()
        if len(s) < 2: return pd.Series(0.0, index=df.index)
        mu, sd = s.mean(), s.std()
        if sd < 1e-9: return pd.Series(0.0, index=df.index)
        z = (df[col] - mu) / sd
        return (-z if invert else z).fillna(0)

    df["mf_score"] = (
        0.35 * zscore("mom_mean") +
        0.25 * zscore("roe") +
        0.15 * zscore("profit_margin") +
        0.15 * zscore("beta", invert=True) +
        0.10 * zscore("ps",   invert=True)
    ).round(3)

    df = df.sort_values("mf_score", ascending=False).reset_index(drop=True)
    df.insert(0,"rank",range(1,len(df)+1))
    log("计算完成", 100)
    return df


# ─── 市场温度计 ───────────────────────────────────────────────────────────────

def _safe_series(s):
    """把 pd.Series 转成普通 list，NaN → None"""
    import numpy as np
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)
            for v in s]

def _percentile_of(series, value):
    """value 在 series 历史中的百分位（0-100）"""
    import numpy as np
    if value is None: return None
    arr = [v for v in series if v is not None]
    if not arr: return None
    return round(float(np.sum(np.array(arr) <= value) / len(arr) * 100), 1)

def _fetch_market_data():
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import requests as req

    result = {}
    today = pd.Timestamp.today()
    start_2y  = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    start_60d = (today - pd.DateOffset(days=60)).strftime("%Y-%m-%d")

    def dl1(ticker, start=None):
        """下载单个 ticker，返回 Close Series"""
        raw = yf.download(ticker, start=start or start_2y, auto_adjust=True, progress=False)
        c = raw["Close"] if "Close" in raw.columns else raw
        if hasattr(c, "ndim") and c.ndim == 2: c = c.iloc[:, 0]
        return c.dropna()

    def mk(val, chg, hist2y, hist60d, rnd=2, dates60d=None):
        v = round(float(val), rnd) if val is not None else None
        h2 = [round(float(x), rnd) for x in hist2y if x is not None and not np.isnan(x)]
        h60 = [round(float(x), rnd) for x in hist60d if x is not None and not np.isnan(x)]
        c = round(float(chg), rnd) if chg is not None else None
        out = {"value": v, "change": c,
               "percentile": _percentile_of(h2, v),
               "history": h60}
        if dates60d is not None:
            out["dates"] = list(dates60d)
        return out

    def fred(series_id):
        df = pd.read_csv(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
        df.columns = ["date", "val"]
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna(subset=["val"]).sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        s2y  = df[df["date"] >= pd.Timestamp(start_2y)]
        s60d = df[df["date"] >= pd.Timestamp(start_60d)]
        dates = [str(d.date()) for d in s60d["date"]]
        return s2y["val"], s60d["val"], dates

    # ── VIX ──────────────────────────────────────────────────────────────────
    try:
        s = dl1("^VIX")
        s2y = s; s60d = s[s.index >= start_60d]
        chg = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None
        dates = [str(d.date()) for d in s60d.index]
        result["vix"] = mk(s.iloc[-1], chg, s2y, s60d, dates60d=dates)
    except: result["vix"] = {}

    # ── Put/Call Ratio — CBOE 官网实时（Total / Equity）────────────────────
    try:
        import re as _re
        pc_r = req.get("https://www.cboe.com/us/options/market_statistics/daily/",
                       headers=_HEADERS, timeout=10)
        # 页面里是转义 JSON: {\"name\":\"TOTAL PUT/CALL RATIO\",\"value\":\"0.74\"}
        pc_matches = dict(_re.findall(
            r'\{\\"name\\":\\"([^"\\\\]+PUT/CALL[^"\\\\]*)\\"[^}]*\\"value\\":\\"([^"\\\\]+)\\"',
            pc_r.text, _re.IGNORECASE))
        total_pc  = _safe(float(pc_matches["TOTAL PUT/CALL RATIO"]))  if "TOTAL PUT/CALL RATIO"  in pc_matches else None
        equity_pc = _safe(float(pc_matches["EQUITY PUT/CALL RATIO"])) if "EQUITY PUT/CALL RATIO" in pc_matches else None
        index_pc  = _safe(float(pc_matches["INDEX PUT/CALL RATIO"]))  if "INDEX PUT/CALL RATIO"  in pc_matches else None

        val = equity_pc or total_pc
        # 分位参考：equity P/C 历史区间（手工校准）
        hist_ref = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                    0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20]
        result["put_call"] = {
            "value": round(val, 2) if val else None,
            "change": None,
            "percentile": _percentile_of(hist_ref, val) if val else None,
            "history": [], "dates": [],
            "note_extra": f"Total {total_pc}  ·  Equity {equity_pc}  ·  Index {index_pc}",
        }
    except: result["put_call"] = {}

    # ── 广度：RSP/SPY 比值（等权/市值权重，上升=广度扩张）────────────────────
    try:
        rsp = dl1("RSP"); spy = dl1("SPY")
        idx = rsp.index.intersection(spy.index)
        ratio = (rsp[idx] / spy[idx]).dropna()
        ma5 = ratio.rolling(5).mean().dropna()
        s60d = ma5[ma5.index >= start_60d]
        chg = float(ma5.iloc[-1] - ma5.iloc[-2]) if len(ma5) > 1 else None
        result["ad_ratio"] = mk(ma5.iloc[-1], chg, ma5, s60d, rnd=4)
    except: result["ad_ratio"] = {}

    # ── McClellan Oscillator — 用 RSP/SPY net momentum 近似 ─────────────────
    try:
        rsp = dl1("RSP"); spy = dl1("SPY")
        idx = rsp.index.intersection(spy.index)
        net = (rsp[idx] - spy[idx] * (rsp[idx].iloc[0] / spy[idx].iloc[0])).dropna()
        ema19 = net.ewm(span=19, adjust=False).mean()
        ema39 = net.ewm(span=39, adjust=False).mean()
        mcl = (ema19 - ema39).dropna()
        s60d = mcl[mcl.index >= start_60d]
        chg = float(mcl.iloc[-1] - mcl.iloc[-2]) if len(mcl) > 1 else None
        result["mcclell"] = mk(mcl.iloc[-1], chg, mcl, s60d, rnd=2)
    except: result["mcclell"] = {}

    # ── 52周新高/新低比 — QQQ vs IWM 相对强弱近似 ───────────────────────────
    try:
        qqq = dl1("QQQ"); iwm = dl1("IWM")
        idx = qqq.index.intersection(iwm.index)
        # 计算各自距52周高点的比例
        hi52_qqq = qqq[idx].rolling(252).max()
        hi52_iwm = iwm[idx].rolling(252).max()
        ratio = ((qqq[idx]/hi52_qqq) / (iwm[idx]/hi52_iwm)).dropna()
        s60d = ratio[ratio.index >= start_60d]
        chg = float(ratio.iloc[-1] - ratio.iloc[-2]) if len(ratio) > 1 else None
        result["nh_nl"] = mk(ratio.iloc[-1], chg, ratio, s60d, rnd=3)
    except: result["nh_nl"] = {}

    # ── 纳指 PE ──────────────────────────────────────────────────────────────
    try:
        pe = _safe(yf.Ticker("QQQ").info.get("trailingPE"))
        # 近10年纳指 PE 历史区间作为分位参考
        hist_ref = [15,18,20,22,24,26,28,30,32,35,38,40,42,45]
        result["ndx_pe"] = {
            "value": round(float(pe), 1) if pe else None, "change": None,
            "percentile": _percentile_of(hist_ref, pe) if pe else None,
            "history": [],
        }
    except: result["ndx_pe"] = {}

    # ── CAPE (Shiller PE) — Yale 数据 ────────────────────────────────────────
    try:
        df = pd.read_excel("http://www.econ.yale.edu/~shiller/data/ie_data.xls",
                           sheet_name="Data", skiprows=7, engine="xlrd")
        cape = pd.to_numeric(df["CAPE"], errors="coerce").dropna().tail(300).tolist()
        val = cape[-1] if cape else None
        result["cape"] = {
            "value": round(float(val), 1) if val else None, "change": None,
            "percentile": _percentile_of([round(v,1) for v in cape], val) if val else None,
            "history": [round(v,1) for v in cape[-24:]],
        }
    except: result["cape"] = {}

    # ── 10Y-2Y 利差 — FRED T10Y2Y ────────────────────────────────────────────
    try:
        s2y, s60d, dates = fred("T10Y2Y")
        chg = float(s2y.iloc[-1] - s2y.iloc[-2]) if len(s2y) > 1 else None
        result["yield_spread"] = mk(s2y.iloc[-1], chg, s2y, s60d, rnd=2, dates60d=dates)
    except: result["yield_spread"] = {}

    # ── 标普500>200均线占比 — SPY+QQQ+IWM 均自身均线判断 ────────────────────
    try:
        scores = {}
        for t in ["SPY","QQQ","IWM","DIA","MDY"]:
            try:
                s = dl1(t)
                ma200 = s.rolling(200).mean()
                above = (s > ma200).astype(float).rolling(20).mean().dropna()
                scores[t] = above
            except: pass
        if scores:
            combined = pd.concat(scores.values(), axis=1).mean(axis=1).dropna() * 100
            s60d = combined[combined.index >= start_60d]
            chg = float(combined.iloc[-1] - combined.iloc[-2]) if len(combined) > 1 else None
            result["pct200"] = mk(combined.iloc[-1], chg, combined, s60d, rnd=1)
        else: result["pct200"] = {}
    except: result["pct200"] = {}

    # ── QQQ 资金流向（3日成交额滚动）────────────────────────────────────────
    try:
        raw = yf.download("QQQ", start=start_60d, auto_adjust=True, progress=False)
        prices = raw["Close"].dropna()
        volume = raw["Volume"].dropna()
        if hasattr(prices,"ndim") and prices.ndim==2: prices=prices.iloc[:,0]
        if hasattr(volume,"ndim") and volume.ndim==2: volume=volume.iloc[:,0]
        flow = (prices * volume / 1e8).dropna()
        flow3 = flow.rolling(3).sum().dropna()
        # 2年历史分位用60天数据近似
        chg = float(flow3.iloc[-1] - flow3.iloc[-2]) if len(flow3) > 1 else None
        result["qqq_flow"] = mk(flow3.iloc[-1], chg, flow3, flow3, rnd=1)
    except: result["qqq_flow"] = {}

    # ── 垃圾债信用利差 — FRED BAMLH0A0HYM2（单位：%，展示为bps）────────────
    try:
        s2y, s60d, dates = fred("BAMLH0A0HYM2")
        val_bps = float(s2y.iloc[-1]) * 100
        chg_bps = float(s2y.iloc[-1] - s2y.iloc[-2]) * 100 if len(s2y) > 1 else None
        h2_bps = [v*100 for v in s2y.tolist()]
        h60_bps = [v*100 for v in s60d.tolist()]
        result["hy_spread"] = {
            "value": round(val_bps, 0), "change": round(chg_bps, 0) if chg_bps else None,
            "percentile": _percentile_of(h2_bps, val_bps),
            "history": [round(v, 0) for v in h60_bps],
            "dates": dates,
        }
    except: result["hy_spread"] = {}

    # ── 保证金债务 — FRED BOGMBBM（商业银行保证金贷款，月度，单位：十亿→亿）
    try:
        s2y, s60d, dates = fred("BOGMBBM")
        val = float(s2y.iloc[-1]) * 10
        chg = float(s2y.iloc[-1] - s2y.iloc[-2]) * 10 if len(s2y) > 1 else None
        h2 = [v*10 for v in s2y.tolist()]
        h60 = [v*10 for v in s60d.tolist()]
        result["margin_debt"] = {
            "value": round(val, 0), "change": round(chg, 0) if chg else None,
            "percentile": _percentile_of(h2, val),
            "history": [round(v, 0) for v in h60],
            "dates": dates,
        }
    except: result["margin_debt"] = {}

    # ── AAII 散户情绪 — 用 IWM/SPY 比值近似（散户偏好小盘）───────────────────
    try:
        iwm = dl1("IWM"); spy = dl1("SPY")
        idx = iwm.index.intersection(spy.index)
        ratio = (iwm[idx] / spy[idx]).dropna()
        pct = ratio.rank(pct=True) * 100
        s60d = pct[pct.index >= start_60d]
        val = float(pct.iloc[-1])
        chg = float(pct.iloc[-1] - pct.iloc[-2]) if len(pct) > 1 else None
        result["aaii_bull"] = mk(val, chg, pct, s60d, rnd=1)
    except: result["aaii_bull"] = {}

    return result


_market_cache = {"ts": 0, "data": None}
_MARKET_TTL = 3600  # 1小时缓存

@app.route("/api/market")
def api_market():
    import time
    now = time.time()
    if now - _market_cache["ts"] < _MARKET_TTL and _market_cache["data"]:
        return jsonify(_market_cache["data"])
    try:
        data = _fetch_market_data()
        _market_cache["ts"] = now
        _market_cache["data"] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




STRATEGIES = {
    "momentum":         calc_momentum,
    "momentum_quality": calc_momentum_quality,
    "low_vol":          calc_low_vol,
    "piotroski":        calc_piotroski,
    "dual_momentum":    calc_dual_momentum,
    "multifactor":      calc_multifactor,
}

# ─── 详情接口 ─────────────────────────────────────────────────────────────────

@app.route("/api/detail/<ticker>")
def api_detail(ticker):
    import yfinance as yf
    months = int(request.args.get("months", 6))
    start, end = resolve_dates(request.args.get("start"), request.args.get("end"), months)
    t    = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, auto_adjust=True)
    prices = [{"date":str(d.date()),"close":round(float(c),2),"volume":int(v)}
              for d,c,v in zip(hist.index,hist["Close"],hist["Volume"]) if c==c]
    info = {}
    try:
        raw = t.info
        info = {k: _safe(raw.get(v)) for k,v in {
            "pe":"trailingPE","forward_pe":"forwardPE","pb":"priceToBook",
            "ps":"priceToSalesTrailing12Months","ev_ebitda":"enterpriseToEbitda",
            "market_cap":"marketCap","revenue_growth":"revenueGrowth",
            "earnings_growth":"earningsGrowth","profit_margin":"profitMargins",
            "roe":"returnOnEquity","debt_equity":"debtToEquity",
            "dividend_yield":"dividendYield","beta":"beta",
            "52w_high":"fiftyTwoWeekHigh","52w_low":"fiftyTwoWeekLow",
            "analyst_target":"targetMeanPrice","recommendation":"recommendationKey",
            "short_name":"shortName","sector":"sector","industry":"industry",
        }.items()}
    except: pass
    return jsonify({"ticker": ticker, "prices": prices, "info": info})

# ─── 主计算接口 ───────────────────────────────────────────────────────────────

_calc_lock = threading.Lock()

@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.json or {}

    def worker():
        try:
            import pandas as pd
            meta = build_universe(body.get("universe","nasdaq100"), body.get("sector_filter",""))
            start_str, end_str = resolve_dates(body.get("start"), body.get("end"), int(body.get("months",6)))
            log(f"区间：{start_str} → {end_str}", 12)
            prices = load_prices(meta["Ticker"].tolist(), start_str, end_str)
            strategy = body.get("strategy", "momentum")
            fn = STRATEGIES.get(strategy, calc_momentum)
            df = fn(prices, meta, start_str, end_str)
            rows = df.head(100).to_dict(orient="records")
            _broadcast("done", {"rows": rows, "start": start_str, "end": end_str, "strategy": strategy})
        except Exception as e:
            _broadcast("error", {"msg": str(e)})

    if not _calc_lock.acquire(blocking=False):
        return jsonify({"error": "已有任务运行中，请稍候"}), 429

    def run_release():
        try: worker()
        finally: _calc_lock.release()

    threading.Thread(target=run_release, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/sectors")
def api_sectors():
    import pandas as pd
    universe = request.args.get("universe","sp500")
    try:
        if universe=="sp500":       meta=fetch_sp500()
        elif universe=="nasdaq100": meta=fetch_nasdaq100()
        else: meta=pd.concat([fetch_sp500(),fetch_nasdaq100()],ignore_index=True)
        return jsonify({"sectors": sorted(meta["Sector"].dropna().unique().tolist())})
    except Exception as e:
        return jsonify({"sectors":[],"error":str(e)})

@app.route("/events")
def events():
    import queue
    q = queue.Queue(maxsize=200)
    with _sse_lock: _sse_queues.append(q)
    def stream():
        try:
            while True:
                try: yield q.get(timeout=30)
                except: yield ": ping\n\n"
        finally:
            with _sse_lock:
                if q in _sse_queues: _sse_queues.remove(q)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "momentum_ui.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  美股多策略排序工具")
    print(f"  访问: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
