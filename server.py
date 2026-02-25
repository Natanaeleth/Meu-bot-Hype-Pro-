"""
HyperBot Pro v2 — server.py
Trading autônomo Hyperliquid • Futuros Perpétuos
Recursos: preços ao vivo, saldo real, lotes de até 5 posições,
          capital alocado por lote, histórico de trades, notícias crypto
"""

import json, time, threading, logging, sys, math, re
from collections import deque
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response, stream_with_context

try:
    from flask_cors import CORS; _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

from hl_crypto import sign_l1_action, float_to_wire, private_key_to_address

# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)7s │ %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hyperbot.log', encoding='utf-8')
    ]
)
L = logging.getLogger('HyperBot')

# ══════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)
if _HAS_CORS:
    CORS(app)

@app.after_request
def _cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS,DELETE'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return r

@app.before_request
def _preflight():
    if request.method == 'OPTIONS':
        r = Response()
        r.headers['Access-Control-Allow-Origin'] = '*'
        r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS,DELETE'
        r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return r, 200

# ══════════════════════════════════════════════════════════════════
# HYPERLIQUID API
# ══════════════════════════════════════════════════════════════════
HL_INFO = 'https://api.hyperliquid.xyz/info'
HL_EXCH = 'https://api.hyperliquid.xyz/exchange'

_meta_cache = {'data': None, 'ts': 0}

def hl_post(url, payload, timeout=12):
    try:
        data = json.dumps(payload).encode()
        req  = Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        L.warning(f'HL HTTP {e.code}: {e.read().decode()[:120]}')
    except URLError as e:
        L.debug(f'HL URL error: {e.reason}')
    except Exception as e:
        L.debug(f'HL error: {e}')
    return None

def get_meta():
    now = time.time()
    if now - _meta_cache['ts'] < 300 and _meta_cache['data']:
        return _meta_cache['data']
    d = hl_post(HL_INFO, {'type': 'meta'})
    if d:
        _meta_cache['data'] = d
        _meta_cache['ts']   = now
    return d

def get_all_mids():
    return hl_post(HL_INFO, {'type': 'allMids'})

def get_user_state(addr):
    return hl_post(HL_INFO, {'type': 'clearinghouseState', 'user': addr})

def get_candles(coin, interval='15m', days=10):
    start = int((time.time() - days * 86400) * 1000)
    return hl_post(HL_INFO, {
        'type': 'candleSnapshot',
        'req':  {'coin': coin, 'interval': interval, 'startTime': start}
    })

def get_open_orders(addr):
    return hl_post(HL_INFO, {'type': 'openOrders', 'user': addr}) or []

def get_fills(addr):
    return hl_post(HL_INFO, {'type': 'userFills', 'user': addr}) or []

def get_asset_index(coin):
    meta = get_meta()
    if not meta: return None
    for i, a in enumerate(meta.get('universe', [])):
        if a['name'] == coin: return i
    return None

def get_asset_szdecimals(coin):
    meta = get_meta()
    if not meta: return 3
    for a in meta.get('universe', []):
        if a['name'] == coin: return a.get('szDecimals', 3)
    return 3

# ══════════════════════════════════════════════════════════════════
# SIMULATED CANDLES (fallback when API offline)
# ══════════════════════════════════════════════════════════════════
COIN_BASES = {
    'BTC':65000,'ETH':3200,'SOL':150,'ARB':1.2,'OP':2.5,'AVAX':37,
    'LINK':15,'APT':8,'SUI':1.1,'INJ':24,'DYDX':2.1,'WIF':2.5,
    'PEPE':0.000012,'BONK':0.000025,'TIA':9,'GMX':28,'UNI':9,
    'CBBTC':65000,'MAGIC':0.6,'RDNT':0.08,'PENDLE':4.5,'GNS':1.8,'SEI':0.35,
}

def sim_candles(coin, n=220):
    base  = COIN_BASES.get(coin, 10)
    seed  = (hash(coin) % 2**31 + int(time.time() / 3600)) % 2**31
    rng   = np.random.default_rng(seed)
    pl    = n // 5
    prices = [base]
    for drift, vol in [(0.0006,0.006),(0.0020,0.010),(0.0000,0.012),(-0.0015,0.010),(0.0018,0.009)]:
        for _ in range(pl):
            prices.append(prices[-1] * (1 + drift + rng.normal(0, vol)))
    prices = np.array(prices[1:n+1])
    if len(prices) < n:
        prices = np.append(prices, [prices[-1]] * (n - len(prices)))
    vb = base * 15000
    vols = rng.uniform(vb*0.5, vb*1.5, n)
    for b in [pl*2, pl*3, pl*4]:
        if b < n: vols[b-2:b+3] *= rng.uniform(2.5, 4.0)
    out = []
    for i in range(n):
        p = prices[i]; a = p * rng.uniform(0.004, 0.018)
        out.append({'T':int((time.time()-(n-i)*900)*1000),
                    'o':str(round(prices[i-1] if i>0 else p,8)),
                    'h':str(round(p+a*rng.uniform(0.3,1),8)),
                    'l':str(round(p-a*rng.uniform(0.3,1),8)),
                    'c':str(round(p,8)),'v':str(round(float(vols[i]),2))})
    return out

def to_df(raw):
    if not raw: return None
    try:
        rows = [{'ts': c.get('T', c.get('t', 0)),
                 'open':  float(c.get('o', 0)),
                 'high':  float(c.get('h', 0)),
                 'low':   float(c.get('l', 0)),
                 'close': float(c.get('c', 0)),
                 'volume':float(c.get('v', 0))} for c in raw]
        df = pd.DataFrame(rows).dropna()
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.reset_index(drop=True) if len(df) >= 60 else None
    except Exception as e:
        L.error(f'to_df: {e}'); return None

# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════
def ema(s, n):    return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); g=d.where(d>0,0.).rolling(n).mean(); l=(-d.where(d<0,0.)).rolling(n).mean()
    return 100-(100/(1+g/l.replace(0,np.inf)))
def atr(df, n=14):
    hl=df.high-df.low; hc=(df.high-df.close.shift()).abs(); lc=(df.low-df.close.shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(n).mean()
def macd(s, f=12, sl=26, sg=9):
    m=ema(s,f)-ema(s,sl); sig=ema(m,sg); return m,sig,m-sig
def bb(s, n=20, k=2.0):
    mid=s.rolling(n).mean(); std=s.rolling(n).std()
    return mid, mid+k*std, mid-k*std

# ══════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════
NETWORKS = {
    'hl':   ['BTC','ETH','SOL','ARB','OP','AVAX','LINK','APT','SUI','INJ',
             'DYDX','WIF','PEPE','BONK','TIA','SEI'],
    'base': ['ETH','CBBTC','LINK','UNI','AAVE','CRV','LDO','SNX','COMP'],
    'arb':  ['ETH','BTC','ARB','LINK','UNI','GMX','MAGIC','RDNT','PENDLE','GNS'],
}

RISK_PROFILES = {
    'conservative': {'risk': 1.0, 'max_lev': 5,  'min_str': 62, 'rr': 2.5},
    'moderate':     {'risk': 2.0, 'max_lev': 10, 'min_str': 55, 'rr': 2.0},
    'aggressive':   {'risk': 3.0, 'max_lev': 20, 'min_str': 45, 'rr': 1.8},
}

def analyze(df, coin, risk_mode='moderate', strategies=None):
    if df is None or len(df) < 80: return None
    strategies = strategies or ['ema_cross','rsi_div','trend_follow','macd','bb_bounce','breakout','scalping']

    c=df.close; v=df.volume
    e9=ema(c,9); e21=ema(c,21); e50=ema(c,50); e200=ema(c,min(200,len(df)-5))
    r=rsi(c); at=atr(df); vma=v.rolling(20).mean()
    m,ms,mh=macd(c); bm,bu,bl=bb(c)

    i=len(df)-1

    def v_(s,j):
        val = s.iloc[j]
        return float(val) if not (isinstance(val, float) and math.isnan(val)) else 0.0

    cur  = {k: v_(s,i) for k,s in [('c',c),('e9',e9),('e21',e21),('e50',e50),('e200',e200),
             ('r',r),('at',at),('v',v),('vma',vma),('m',m),('ms',ms),
             ('bu',bu),('bl',bl),('bm',bm)]}

    if cur['vma'] == 0: cur['vma'] = 1.0
    vol_ok   = cur['v'] > cur['vma'] * 1.2
    vol_soft = cur['v'] > cur['vma'] * 0.65
    trend_up = cur['e50'] > cur['e200']
    trend_dn = cur['e50'] < cur['e200']

    scores = []

    # 1. EMA Cross (scan 3 bars back)
    if 'ema_cross' in strategies:
        for look in range(1, 4):
            ji=i-look+1; jp=ji-1
            if jp < 0: break
            cu9=v_(e9,ji); cu21=v_(e21,ji); pr9=v_(e9,jp); pr21=v_(e21,jp)
            vj = v_(v,ji); vmj = v_(vma,ji)
            v_ok = vj > vmj*1.0 if vmj > 0 else True
            if pr9<=pr21 and cu9>cu21 and trend_up and 33<v_(r,ji)<76 and (v_ok or look>1):
                scores.append(('LONG',  42-look*2, 'EMA Cross ↑')); break
            if pr9>=pr21 and cu9<cu21 and trend_dn and 24<v_(r,ji)<67 and (v_ok or look>1):
                scores.append(('SHORT', 42-look*2, 'EMA Cross ↓')); break

    # 2. RSI Divergence (scan 5 bars)
    if 'rsi_div' in strategies:
        for look in range(0, 5):
            ji=i-look
            if ji < 5: break
            rj=v_(r,ji); cj=v_(c,ji); r4=v_(r,max(0,ji-4)); c4=v_(c,max(0,ji-4))
            if rj<37 and cj<c4 and rj>r4:
                scores.append(('LONG',  35-look, 'RSI Bull Div')); break
            if rj>63 and cj>c4 and rj<r4:
                scores.append(('SHORT', 35-look, 'RSI Bear Div')); break

    # 3. MACD Cross (scan 3 bars)
    if 'macd' in strategies:
        for look in range(1, 4):
            ji=i-look+1; jp=ji-1
            if jp < 0: break
            cm=v_(m,ji); cms=v_(ms,ji); pm=v_(m,jp); pms=v_(ms,jp)
            if pm<=pms and cm>cms and cm<0 and trend_up:
                scores.append(('LONG',  30-look, 'MACD Cross ↑')); break
            if pm>=pms and cm<cms and cm>0 and trend_dn:
                scores.append(('SHORT', 30-look, 'MACD Cross ↓')); break

    # 4. Bollinger Bounce (scan 3 bars)
    if 'bb_bounce' in strategies:
        for look in range(0, 3):
            ji=i-look; jp=max(0,ji-1)
            pc=v_(c,jp); pbl=v_(bl,jp); pbu=v_(bu,jp)
            cc=v_(c,ji); cbl=v_(bl,ji); cbu=v_(bu,ji); cr=v_(r,ji)
            if pc<=pbl and cc>cbl and cr<49:
                scores.append(('LONG',  28-look, 'BB Bounce ↑')); break
            if pc>=pbu and cc<cbu and cr>51:
                scores.append(('SHORT', 28-look, 'BB Bounce ↓')); break

    # 5. Trend Pullback
    if 'trend_follow' in strategies:
        if cur['e21']>cur['e50']>cur['e200'] and cur['e21']*0.996<cur['c']<cur['e21']*1.005 and 40<cur['r']<70 and vol_soft:
            scores.append(('LONG', 38, 'Trend Pullback'))
        if cur['e21']<cur['e50']<cur['e200'] and cur['e21']*0.995<cur['c']<cur['e21']*1.004 and 30<cur['r']<60 and vol_soft:
            scores.append(('SHORT', 38, 'Trend Pullback Short'))

    # 6. Breakout
    if 'breakout' in strategies:
        for look in range(0, 3):
            ji=i-look
            if ji < 21: break
            rh=float(df.high.iloc[ji-20:ji].max()); rl=float(df.low.iloc[ji-20:ji].min())
            cc=v_(c,ji); cr=v_(r,ji); vv=v_(v,ji); vvm=v_(vma,ji)
            vb2 = vv > vvm*1.1 if vvm>0 else True
            if cc>rh*0.997 and vb2 and cr>50:
                scores.append(('LONG',  28-look, 'Breakout High')); break
            if cc<rl*1.003 and vb2 and cr<50:
                scores.append(('SHORT', 28-look, 'Breakout Low')); break


    # 7. Scalping — EMA3/EMA8 micro-cross com volume spike (ideal para 5m)
    if 'scalping' in strategies:
        e3 = ema(c, 3); e8 = ema(c, 8)
        for look in range(1, 3):
            ji = i - look + 1; jp = ji - 1
            if jp < 0: break
            cu3=v_(e3,ji); cu8=v_(e8,ji); pr3=v_(e3,jp); pr8=v_(e8,jp)
            vv=v_(v,ji); vvm=v_(vma,ji)
            vol_spike = (vv > vvm * 1.4) if vvm > 0 else False
            rj = v_(r, ji)
            if pr3 <= pr8 and cu3 > cu8 and vol_spike and 38 < rj < 72 and trend_up:
                scores.append(('LONG',  30 - look, 'Scalp EMA3/8 ↑')); break
            if pr3 >= pr8 and cu3 < cu8 and vol_spike and 28 < rj < 62 and trend_dn:
                scores.append(('SHORT', 30 - look, 'Scalp EMA3/8 ↓')); break

    if not scores: return None

    ls = sum(s[1] for s in scores if s[0]=='LONG')
    ss2= sum(s[1] for s in scores if s[0]=='SHORT')
    direction = 'LONG' if ls >= ss2 else 'SHORT'
    raw_str   = max(ls, ss2)
    strength  = min(97, raw_str + (8 if vol_ok else 0) + (5 if len(scores)>1 else 0))

    prof = RISK_PROFILES.get(risk_mode, RISK_PROFILES['moderate'])
    if strength < prof['min_str']: return None

    entry = cur['c']
    if entry <= 0: return None
    atr_pct  = (cur['at'] / entry) * 100
    stop_pct = min(3.2, max(0.8, atr_pct * 1.3))
    sl = entry*(1-stop_pct/100) if direction=='LONG' else entry*(1+stop_pct/100)
    tp = entry*(1+stop_pct*prof['rr']/100) if direction=='LONG' else entry*(1-stop_pct*prof['rr']/100)
    lev = min(prof['max_lev'], max(3, int(3+strength/15)))
    reasons = ' + '.join(dict.fromkeys(s[2] for s in scores if s[0]==direction))

    return {
        'coin': coin, 'symbol': coin+'/USDT', 'direction': direction,
        'strength': round(strength, 1),
        'entryPrice': round(entry, 8), 'stopLoss': round(sl, 8),
        'takeProfit': round(tp, 8), 'stopPct': round(stop_pct, 3),
        'leverage': lev, 'rsi': round(cur['r'], 2),
        'reasons': reasons, 'timestamp': int(time.time()*1000),
    }

# ══════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════
def _round_size(size, szDecimals):
    factor = 10 ** szDecimals
    return math.floor(size * factor) / factor

def place_order(pk, addr, coin, is_buy, size, price, reduce_only=False, market=False):
    idx = get_asset_index(coin)
    if idx is None:
        L.error(f'Asset not found: {coin}'); return None
    szd = get_asset_szdecimals(coin)
    size = _round_size(size, szd)
    if size <= 0:
        L.error(f'Size too small for {coin}'); return None
    if market:
        px = float_to_wire(price * (1.05 if is_buy else 0.95))
        tif = 'Ioc'
    else:
        px  = float_to_wire(price)
        tif = 'Gtc'
    sz = float_to_wire(size)
    action = {
        'type': 'order',
        'orders': [{'a': idx, 'b': is_buy, 'p': px, 's': sz,
                    'r': reduce_only, 't': {'limit': {'tif': tif}}}],
        'grouping': 'na',
    }
    try:
        payload = sign_l1_action(action, pk)
        result  = hl_post(HL_EXCH, payload)
        if result:
            L.info(f'Order {coin} {"BUY" if is_buy else "SELL"} {size}@{px}: {result}')
        return result
    except Exception as e:
        L.error(f'place_order error: {e}', exc_info=True)
        return None

def cancel_all_orders(pk, addr, coin):
    orders = get_open_orders(addr)
    idx    = get_asset_index(coin)
    if idx is None: return
    to_cancel = [o for o in orders if o.get('coin') == coin]
    if not to_cancel: return
    action = {
        'type': 'cancel',
        'cancels': [{'a': idx, 'o': int(o['oid'])} for o in to_cancel]
    }
    try:
        payload = sign_l1_action(action, pk)
        hl_post(HL_EXCH, payload)
    except Exception as e:
        L.error(f'cancel error: {e}')

# ══════════════════════════════════════════════════════════════════
# BOT STATE
# ══════════════════════════════════════════════════════════════════
BS = {
    # Connection
    'wallet':       None,
    'pk':           None,
    'network':      'hl',
    # Bot config
    'running':      False,
    'mode':         'scan',       # scan | auto | aggressive
    'risk_mode':    'moderate',
    'timeframe':    '15m',
    'strategies':   ['ema_cross','rsi_div','trend_follow','macd','bb_bounce'],
    # Capital config
    'total_capital': 0.0,         # USD to allocate (set by user)
    'max_positions': 5,           # max simultaneous positions (1-5)
    'capital_per_slot': 0.0,      # total_capital / max_positions
    # State
    'positions':    {},           # coin → position dict
    'signals':      [],
    'scan_count':   0,
    'offline_mode': False,
    # Stats
    'stats': {
        'balance':        0.0,
        'pnl_today':      0.0,
        'pnl_total':      0.0,
        'wins':           0,
        'losses':         0,
        'open_pos':       0,
        'unrealized_pnl': 0.0,
    },
    # History
    'trade_history': [],
    'pnl_history':   [],
    # Logs
    'logs':          [],
    # Price cache
    'price_cache':   {},
    'price_ts':      0,
    # News cache
    'news_cache':    [],
    'news_ts':       0,
}

_lock        = threading.Lock()
_sse_clients = []   # list of queues for SSE price stream

# ══════════════════════════════════════════════════════════════════
# LOGGING HELPER
# ══════════════════════════════════════════════════════════════════
def add_log(msg, level='info'):
    entry = {'time': time.strftime('%H:%M:%S'), 'msg': msg, 'level': level}
    with _lock:
        BS['logs'].append(entry)
        if len(BS['logs']) > 500: BS['logs'] = BS['logs'][-500:]
    L.info(f'[{level.upper()}] {msg}')

# ══════════════════════════════════════════════════════════════════
# LIVE PRICE FEED (background thread, every 3s)
# ══════════════════════════════════════════════════════════════════
def _price_feed():
    while True:
        try:
            mids = get_all_mids()
            if mids:
                with _lock:
                    BS['price_cache'].update(mids)
                    BS['price_ts'] = int(time.time() * 1000)
                msg = 'data:' + json.dumps({'prices': mids, 'ts': BS['price_ts']}) + '\n\n'
                dead = []
                for q in _sse_clients:
                    try: q.put_nowait(msg)
                    except: dead.append(q)
                for q in dead:
                    try: _sse_clients.remove(q)
                    except: pass
        except Exception as e:
            L.debug(f'price_feed: {e}')
        time.sleep(3)

threading.Thread(target=_price_feed, daemon=True, name='PriceFeed').start()

# ══════════════════════════════════════════════════════════════════
# NEWS FETCHER (CoinGecko + Crypto RSS)
# ══════════════════════════════════════════════════════════════════

# Keywords → impact level
HIGH_IMPACT   = ['etf','fed','fomc','sec','hack','exploit','crash','ban','halving',
                 'blackrock','bitcoin reserve','shutdown','whale','regulation','lawsuit',
                 'collapse','bankruptcy','freeze','sarham','emergency']
MEDIUM_IMPACT = ['upgrade','launch','partnership','integration','listing','mainnet',
                 'airdrop','staking','defi','nft','token','protocol','bridge','update',
                 'partnership','merger','funding','raise']

def classify_impact(title):
    t = title.lower()
    if any(k in t for k in HIGH_IMPACT):   return 'high'
    if any(k in t for k in MEDIUM_IMPACT): return 'medium'
    return 'low'

def _fetch_news():
    """Fetch crypto news from multiple RSS feeds"""
    FEEDS = [
        'https://cointelegraph.com/rss',
        'https://coindesk.com/arc/outboundfeeds/rss/',
        'https://decrypt.co/feed',
        'https://bitcoinmagazine.com/.rss/full/',
    ]
    import html
    news = []
    for feed_url in FEEDS:
        try:
            req  = Request(feed_url, headers={'User-Agent':'Mozilla/5.0'})
            with urlopen(req, timeout=8) as r:
                content = r.read().decode('utf-8', errors='replace')
            # Parse RSS with regex (no xml lib needed)
            items = re.findall(r'<item[^>]*>(.*?)</item>', content, re.DOTALL)
            for item in items[:8]:
                title = re.search(r'<title[^>]*><!\[CDATA\[(.*?)\]\]>', item, re.DOTALL)
                if not title:
                    title = re.search(r'<title[^>]*>(.*?)</title>', item, re.DOTALL)
                link  = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
                pubDate = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)
                if not title: continue
                t = html.unescape(title.group(1).strip())
                l = link.group(1).strip() if link else '#'
                d = pubDate.group(1).strip() if pubDate else ''
                impact = classify_impact(t)
                # Parse source name from URL
                src = re.search(r'https?://([^/]+)', feed_url)
                source = src.group(1).replace('www.','') if src else feed_url
                news.append({
                    'title':  t, 'url': l, 'source': source,
                    'date':   d, 'impact': impact,
                    'ts':     int(time.time() * 1000)
                })
        except Exception as e:
            L.debug(f'News fetch {feed_url}: {e}')
    # Sort by impact: high first
    order = {'high': 0, 'medium': 1, 'low': 2}
    news.sort(key=lambda n: order.get(n['impact'], 2))
    return news[:50]

def _news_loop():
    while True:
        try:
            news = _fetch_news()
            if news:
                with _lock:
                    BS['news_cache'] = news
                    BS['news_ts']    = int(time.time() * 1000)
                add_log(f'📰 {len(news)} notícias atualizadas', 'info')
        except Exception as e:
            L.debug(f'news_loop: {e}')
        time.sleep(300)  # refresh every 5 minutes

threading.Thread(target=_news_loop, daemon=True, name='NewsFeed').start()

# ══════════════════════════════════════════════════════════════════
# BALANCE & POSITIONS (real-time from HL)
# ══════════════════════════════════════════════════════════════════
def fetch_balance_and_positions():
    """Single call fetches both balance AND real positions from HL"""
    if not BS['wallet']: return 0.0
    state = get_user_state(BS['wallet'])
    if not state: return BS['stats']['balance']

    # Balance
    summary = state.get('crossMarginSummary', {})
    balance = float(summary.get('accountValue', 0))
    BS['stats']['balance'] = balance

    # Unrealized PnL from HL
    upnl = float(summary.get('totalUnrealizedPnl', 0))
    BS['stats']['unrealized_pnl'] = round(upnl, 2)

    # Real positions
    real = {}
    for ap in state.get('assetPositions', []):
        pos_data = ap.get('position', {})
        sz = float(pos_data.get('szi', 0))
        if sz == 0: continue
        coin = pos_data.get('coin', '')
        entry_px  = float(pos_data.get('entryPx', 0) or 0)
        lev_data  = pos_data.get('leverage', {})
        lev_val   = float(lev_data.get('value', 1)) if isinstance(lev_data, dict) else float(lev_data or 1)
        upnl_pos  = float(pos_data.get('unrealizedPnl', 0) or 0)
        margin_used = float(pos_data.get('marginUsed', 0) or 0)
        size_usd  = abs(sz) * entry_px

        # If we already tracked this position, preserve our TP/SL
        existing = BS['positions'].get(coin, {})
        real[coin] = {
            'coin':         coin,
            'symbol':       coin + '/USDT',
            'direction':    'LONG' if sz > 0 else 'SHORT',
            'size':         abs(sz),
            'entryPrice':   entry_px,
            'currentPrice': float(BS['price_cache'].get(coin, entry_px) or entry_px),
            'stopLoss':     existing.get('stopLoss', 0.0),
            'takeProfit':   existing.get('takeProfit', 0.0),
            'stopPct':      existing.get('stopPct', 1.5),
            'sizeUsd':      round(size_usd, 2),
            'leverage':     lev_val,
            'pnl':          round(upnl_pos, 2),
            'pnlPct':       round(upnl_pos / max(margin_used, 0.01) * 100, 2) if margin_used else 0,
            'breakeven':    existing.get('breakeven', False),
            'trailing':     existing.get('trailing', False),
            'real':         True,
            'openTime':     existing.get('openTime', time.strftime('%H:%M:%S')),
            'hlLink':       f'https://app.hyperliquid.xyz/trade/{coin}',
        }

    with _lock:
        # Add new real positions
        for coin, pos in real.items():
            BS['positions'][coin] = pos
        # Remove positions closed on HL side
        closed_coins = [k for k in BS['positions']
                        if BS['positions'][k].get('real') and k not in real]
        for coin in closed_coins:
            pos = BS['positions'].pop(coin)
            # Record as closed trade
            cp = float(BS['price_cache'].get(coin, pos['entryPrice']) or pos['entryPrice'])
            record_closed_trade(pos, cp, 'Fechado na HL')
    BS['stats']['open_pos'] = len(BS['positions'])
    return balance

def update_position_prices():
    """Update currentPrice + PnL for all tracked positions using cached prices"""
    prices = BS['price_cache']
    if not prices: return
    with _lock:
        for coin, pos in BS['positions'].items():
            px = float(prices.get(coin, pos['currentPrice']) or pos['currentPrice'])
            entry = pos['entryPrice']
            if entry <= 0: continue
            dirn  = pos['direction']
            mult  = 1 if dirn == 'LONG' else -1
            lev   = pos.get('leverage', 1)
            sz_usd= pos.get('sizeUsd', 0)
            pnl_pct = (px - entry) / entry * 100 * mult * lev
            pnl_usd = sz_usd * pnl_pct / 100
            pos['currentPrice'] = px
            pos['pnl']          = round(pnl_usd, 2)
            pos['pnlPct']       = round(pnl_pct, 2)
    BS['stats']['unrealized_pnl'] = round(
        sum(p.get('pnl', 0) for p in BS['positions'].values()), 2)

# ══════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT (break-even + trailing)
# ══════════════════════════════════════════════════════════════════
def manage_positions():
    update_position_prices()
    with _lock: positions = dict(BS['positions'])

    for coin, pos in positions.items():
        entry = pos['entryPrice']
        tp    = pos['stopLoss']    # NOTE: variable naming kept for consistency
        sl    = pos['stopLoss']
        tp_v  = pos['takeProfit']
        dirn  = pos['direction']
        price = pos['currentPrice']
        if tp_v == 0 or sl == 0: continue

        mult     = 1 if dirn == 'LONG' else -1
        to_tp    = abs(tp_v - entry)
        traveled = mult * (price - entry)

        # Break-even: SL → entry+0.1% when 50% to TP
        if not pos.get('breakeven') and to_tp > 0 and traveled >= to_tp * 0.5:
            new_sl = entry * 1.001 if dirn == 'LONG' else entry * 0.999
            if BS['pk'] and BS['mode'] != 'scan' and pos.get('real'):
                place_order(BS['pk'], BS['wallet'], coin,
                            dirn == 'SHORT', pos['size'], new_sl, reduce_only=True)
            with _lock:
                if coin in BS['positions']:
                    BS['positions'][coin]['stopLoss']  = new_sl
                    BS['positions'][coin]['breakeven'] = True
            add_log(f'🛡️ Break-even {coin}: SL → ${new_sl:.4f}', 'warn')

        # Trailing: activate at 75% to TP
        elif pos.get('breakeven') and not pos.get('trailing') and to_tp > 0 and traveled >= to_tp * 0.75:
            with _lock:
                if coin in BS['positions']:
                    BS['positions'][coin]['trailing'] = True
            add_log(f'🔄 Trailing Stop ativado: {coin}', 'warn')

        elif pos.get('trailing'):
            trail_pct = pos.get('stopPct', 1.5) * 0.5
            curr_sl   = pos['stopLoss']
            if dirn == 'LONG':
                new_trail = price * (1 - trail_pct / 100)
                if new_trail > curr_sl:
                    with _lock:
                        if coin in BS['positions']: BS['positions'][coin]['stopLoss'] = new_trail
            else:
                new_trail = price * (1 + trail_pct / 100)
                if new_trail < curr_sl:
                    with _lock:
                        if coin in BS['positions']: BS['positions'][coin]['stopLoss'] = new_trail

# ══════════════════════════════════════════════════════════════════
# TRADE HISTORY
# ══════════════════════════════════════════════════════════════════
def record_closed_trade(pos, close_price, reason='Manual'):
    dirn    = pos['direction']
    entry   = pos['entryPrice']
    mult    = 1 if dirn == 'LONG' else -1
    lev     = pos.get('leverage', 1)
    sz_usd  = pos.get('sizeUsd', 0)
    pnl_pct = (close_price - entry) / max(entry, 0.00001) * 100 * mult * lev
    pnl_usd = sz_usd * pnl_pct / 100

    trade = {
        'id':         int(time.time() * 1000),
        'coin':       pos['coin'],
        'symbol':     pos.get('symbol', pos['coin'] + '/USDT'),
        'direction':  dirn,
        'entryPrice': entry,
        'closePrice': round(close_price, 8),
        'pnl':        round(pnl_usd, 2),
        'pnlPct':     round(pnl_pct, 2),
        'sizeUsd':    round(sz_usd, 2),
        'leverage':   lev,
        'reason':     reason,
        'openTime':   pos.get('openTime', ''),
        'closeTime':  time.strftime('%H:%M:%S'),
        'date':       time.strftime('%Y-%m-%d'),
        'hlLink':     f'https://app.hyperliquid.xyz/trade/{pos["coin"]}',
    }
    won = pnl_usd > 0
    with _lock:
        BS['trade_history'].insert(0, trade)
        if len(BS['trade_history']) > 500: BS['trade_history'] = BS['trade_history'][:500]
        BS['pnl_history'].append(round(
            sum(t['pnl'] for t in BS['trade_history']), 2))
    BS['stats']['pnl_today'] = round(BS['stats']['pnl_today'] + pnl_usd, 2)
    BS['stats']['pnl_total'] = round(BS['stats']['pnl_total'] + pnl_usd, 2)
    if won: BS['stats']['wins'] += 1
    else:   BS['stats']['losses'] += 1
    return trade

# ══════════════════════════════════════════════════════════════════
# TRADE EXECUTION (with lot-based capital allocation)
# ══════════════════════════════════════════════════════════════════
def execute_trade(sig):
    """
    Execute a trade on Hyperliquid.
    Capital per trade = total_capital / max_positions
    Opens up to max_positions (5) simultaneously.
    """
    coin = sig['coin']
    dirn = sig['direction']

    # Max simultaneous positions gate
    max_pos = BS.get('max_positions', 5)
    if len(BS['positions']) >= max_pos:
        add_log(f'Máx {max_pos} posições atingido — {coin} ignorado', 'warn')
        return

    # No duplicate coin
    if coin in BS['positions']: return

    # Daily drawdown gate
    bal = BS['stats']['balance'] or BS.get('total_capital', 500) or 500
    if bal > 0 and BS['stats']['pnl_today'] < 0:
        drawdown_pct = abs(BS['stats']['pnl_today']) / bal * 100
        if drawdown_pct >= 5.0:
            add_log('⛔ Drawdown diário -5% atingido. Bot pausado.', 'error')
            return

    # Capital per lot
    capital = BS.get('total_capital', 0)
    if capital <= 0:
        capital = bal * 0.8   # default: use 80% of balance

    capital_per_slot = capital / max(max_pos, 1)

    # Size in USD (capped by capital_per_slot)
    size_usd  = min(capital_per_slot, bal * 0.25)
    entry     = sig['entryPrice']
    if entry <= 0: return
    size_coin = size_usd / entry

    is_buy = dirn == 'LONG'
    add_log(
        f"⚡ {dirn} {coin} | Entrada: ${entry:.4f} | "
        f"SL: ${sig['stopLoss']:.4f} | TP: ${sig['takeProfit']:.4f} | "
        f"${size_usd:.0f} @ {sig['leverage']}x",
        'success'
    )

    # SCAN mode: log signal, no execution
    if BS['mode'] == 'scan' or not BS['pk']:
        ep = sig['entryPrice']; rs = sig['reasons']
        add_log(
            f"[SCAN] {dirn} {coin} @ ${ep:.4f} | {rs}",
            'signal-long' if is_buy else 'signal-short'
        )
        return

    # AUTO / AGGRESSIVE: execute on HL
    result = place_order(BS['pk'], BS['wallet'], coin, is_buy, size_coin, entry)
    if result and result.get('status') != 'err':
        # Place protective SL order
        place_order(BS['pk'], BS['wallet'], coin, not is_buy,
                    size_coin, sig['stopLoss'], reduce_only=True)
        # Place TP order
        place_order(BS['pk'], BS['wallet'], coin, not is_buy,
                    size_coin, sig['takeProfit'], reduce_only=True)

        pos = {
            'coin':         coin,
            'symbol':       sig['symbol'],
            'direction':    dirn,
            'entryPrice':   entry,
            'currentPrice': entry,
            'stopLoss':     sig['stopLoss'],
            'takeProfit':   sig['takeProfit'],
            'stopPct':      sig['stopPct'],
            'size':         size_coin,
            'sizeUsd':      round(size_usd, 2),
            'leverage':     sig['leverage'],
            'pnl':          0.0,
            'pnlPct':       0.0,
            'breakeven':    False,
            'trailing':     False,
            'real':         True,
            'openTime':     time.strftime('%H:%M:%S'),
            'hlLink':       f'https://app.hyperliquid.xyz/trade/{coin}',
        }
        with _lock: BS['positions'][coin] = pos
        add_log(f'✅ Posição aberta: {dirn} {coin} | ${size_usd:.0f}', 'success')
    else:
        add_log(f'❌ Falha ao abrir {coin}: {result}', 'error')

# ══════════════════════════════════════════════════════════════════
# MAIN SCAN LOOP
# ══════════════════════════════════════════════════════════════════
INTERVALS = {'5m': 300, '15m': 900, '1h': 3600, '4h': 14400}

def scan_loop():
    while BS['running']:
        try:
            BS['scan_count'] += 1
            add_log(
                f"🔍 Scan #{BS['scan_count']} — "
                f"{BS['network'].upper()} — {BS['timeframe']} — "
                f"Posições: {len(BS['positions'])}/{BS['max_positions']}",
                'info'
            )

            # Fetch real balance + positions from HL
            fetch_balance_and_positions()

            pairs  = NETWORKS.get(BS['network'], NETWORKS['hl'])
            found  = []
            online = True

            for coin in pairs:
                if not BS['running']: break
                raw = get_candles(coin, BS['timeframe'])
                if not raw:
                    online = False
                    raw    = sim_candles(coin)
                df  = to_df(raw)
                sig = analyze(df, coin, BS['risk_mode'], BS['strategies'])
                if sig:
                    found.append(sig)
                    lvl = 'signal-long' if sig['direction'] == 'LONG' else 'signal-short'
                    add_log(
                        f"🎯 {sig['direction']} {coin} | "
                        f"Força: {sig['strength']}/100 | {sig['reasons']}",
                        lvl
                    )
                    if BS['mode'] in ('auto', 'aggressive'):
                        execute_trade(sig)
                time.sleep(0.25)

            BS['offline_mode'] = not online
            found.sort(key=lambda s: s['strength'], reverse=True)
            with _lock: BS['signals'] = found

            add_log(
                f"✅ Scan #{BS['scan_count']} — {len(pairs)} pares | "
                f"{len(found)} sinais | "
                f"Saldo: ${BS['stats']['balance']:.2f}",
                'info'
            )

            # Wait for next scan interval, updating prices every 10s
            interval = INTERVALS.get(BS['timeframe'], 900)
            elapsed  = 0
            while elapsed < interval and BS['running']:
                time.sleep(10)
                elapsed += 10
                update_position_prices()
                manage_positions()
                # Re-sync real positions every 60s
                if elapsed % 60 == 0 and BS['wallet']:
                    fetch_balance_and_positions()

        except Exception as e:
            L.error(f'scan_loop error: {e}', exc_info=True)
            time.sleep(30)

# ══════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({
        'status':  'ok',
        'time':    int(time.time()),
        'offline': BS['offline_mode'],
        'version': '2.0',
    })

@app.route('/api/connect', methods=['POST'])
def connect():
    d      = request.get_json() or {}
    wallet = d.get('wallet', '').strip()
    pk     = d.get('privateKey', '').strip()
    net    = d.get('network', 'hl')

    if not wallet:
        return jsonify({'error': 'Wallet address required'}), 400

    derived = None
    if pk:
        try:
            derived = private_key_to_address(pk)
        except Exception as e:
            return jsonify({'error': f'Private key inválida: {e}'}), 400

    BS['wallet']  = wallet
    BS['pk']      = pk or None
    BS['network'] = net

    balance = fetch_balance_and_positions()
    add_log(f'✅ Wallet: {wallet[:10]}... | Saldo: ${balance:.2f} | Rede: {net.upper()}', 'success')
    add_log(f'Modo: {"🔑 AUTO TRADE" if pk else "🔍 SCAN ONLY"}', 'info')

    return jsonify({
        'success':     True,
        'balance':     balance,
        'mode':        'auto' if pk else 'scan',
        'derivedAddr': derived,
        'positions':   list(BS['positions'].values()),
    })

@app.route('/api/config', methods=['POST'])
def set_config():
    """Set capital allocation and max positions"""
    d = request.get_json() or {}
    if 'totalCapital' in d:
        BS['total_capital'] = float(d['totalCapital'])
    if 'maxPositions' in d:
        BS['max_positions'] = max(1, min(5, int(d['maxPositions'])))
    add_log(
        f'Config: Capital=${BS["total_capital"]:.2f} | '
        f'Max={BS["max_positions"]} posições | '
        f'Por lote=${BS["total_capital"]/max(BS["max_positions"],1):.2f}',
        'info'
    )
    return jsonify({
        'success':        True,
        'totalCapital':   BS['total_capital'],
        'maxPositions':   BS['max_positions'],
        'capitalPerSlot': BS['total_capital'] / max(BS['max_positions'], 1),
    })

@app.route('/api/start', methods=['POST'])
def start():
    d = request.get_json() or {}
    if BS['running']:
        return jsonify({'error': 'Já está rodando'}), 400
    BS.update({
        'mode':        d.get('mode', 'scan'),
        'risk_mode':   d.get('riskMode', 'moderate'),
        'network':     d.get('network', BS['network']),
        'timeframe':   d.get('timeframe', '15m'),
        'strategies':  d.get('strategies', BS['strategies']),
        'running':     True,
    })
    if 'totalCapital' in d:
        BS['total_capital'] = float(d['totalCapital'])
    if 'maxPositions' in d:
        BS['max_positions'] = max(1, min(5, int(d['maxPositions'])))

    threading.Thread(target=scan_loop, daemon=True, name='ScanLoop').start()
    add_log(
        f'🚀 Bot iniciado | Modo: {BS["mode"].upper()} | '
        f'{BS["network"].upper()} | {BS["timeframe"]} | '
        f'Capital: ${BS["total_capital"]:.2f} | Max: {BS["max_positions"]} posições',
        'success'
    )
    return jsonify({'success': True})

@app.route('/api/stop', methods=['POST'])
def stop():
    BS['running'] = False
    add_log('⏹ Bot parado', 'warn')
    return jsonify({'success': True})

@app.route('/api/status')
def status():
    update_position_prices()
    with _lock:
        pos  = list(BS['positions'].values())
        sigs = BS['signals'][:15]
        logs = BS['logs'][-80:]
    return jsonify({
        'running':      BS['running'],
        'mode':         BS['mode'],
        'network':      BS['network'],
        'timeframe':    BS['timeframe'],
        'scan_count':   BS['scan_count'],
        'offline':      BS['offline_mode'],
        'totalCapital': BS['total_capital'],
        'maxPositions': BS['max_positions'],
        'capitalPerSlot': BS['total_capital'] / max(BS['max_positions'], 1),
        'stats':        {**BS['stats'], 'open_pos': len(pos)},
        'positions':    pos,
        'signals':      sigs,
        'logs':         logs,
    })

@app.route('/api/balance')
def balance_route():
    bal = fetch_balance_and_positions()
    return jsonify({
        'balance':    bal,
        'unrealized': BS['stats']['unrealized_pnl'],
        'pnl_today':  BS['stats']['pnl_today'],
    })

@app.route('/api/prices')
def prices_route():
    return jsonify({
        'prices': BS['price_cache'],
        'ts':     BS['price_ts'],
    })

@app.route('/api/prices/stream')
def prices_stream():
    """Server-Sent Events: live price updates every 3s"""
    import queue

    q = queue.Queue(maxsize=20)
    _sse_clients.append(q)

    # Send current prices immediately
    initial = 'data:' + json.dumps({
        'prices': BS['price_cache'], 'ts': BS['price_ts']
    }) + '\n\n'

    def generate():
        yield initial
        while True:
            try:
                msg = q.get(timeout=35)
                yield msg
            except Exception:
                yield 'data:{}\n\n'  # keepalive

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

@app.route('/api/candles')
def candles_route():
    coin  = request.args.get('coin', 'BTC')
    tf    = request.args.get('tf', '15m')
    raw   = get_candles(coin, tf)
    simul = raw is None
    if simul: raw = sim_candles(coin)
    return jsonify({'candles': raw, 'simulated': simul, 'coin': coin})

@app.route('/api/close_position', methods=['POST'])
def close_position():
    d    = request.get_json() or {}
    coin = d.get('coin', '').upper()
    if not coin or coin not in BS['positions']:
        return jsonify({'error': 'Posição não encontrada'}), 404

    pos = BS['positions'][coin]
    cp  = float(BS['price_cache'].get(coin, pos['currentPrice']) or pos['currentPrice'])

    # Execute market close on HL if real position
    if BS['pk'] and pos.get('real'):
        is_buy = pos['direction'] == 'SHORT'  # reverse to close
        cancel_all_orders(BS['pk'], BS['wallet'], coin)
        result = place_order(
            BS['pk'], BS['wallet'], coin,
            is_buy, pos['size'], cp,
            reduce_only=True, market=True
        )
        add_log(f'📤 Fechando {coin} a mercado: {result}', 'warn')

    # Record trade
    trade = record_closed_trade(pos, cp, 'Fechado Manual')
    with _lock:
        if coin in BS['positions']: del BS['positions'][coin]

    add_log(
        f'🔒 Posição fechada: {coin} | PnL: ${trade["pnl"]:.2f} ({trade["pnlPct"]:.2f}%)',
        'success' if trade['pnl'] >= 0 else 'error'
    )
    return jsonify({'success': True, 'trade': trade})

@app.route('/api/signals')
def signals_route():
    return jsonify({'signals': BS['signals']})

@app.route('/api/trades')
def trades_route():
    page    = int(request.args.get('page', 1))
    per     = int(request.args.get('per', 50))
    start   = (page - 1) * per
    trades  = BS['trade_history'][start:start+per]
    total   = len(BS['trade_history'])
    wins    = sum(1 for t in BS['trade_history'] if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in BS['trade_history'])
    return jsonify({
        'trades':    trades,
        'total':     total,
        'page':      page,
        'pages':     math.ceil(total / per) if per > 0 else 1,
        'wins':      wins,
        'losses':    total - wins,
        'win_rate':  round(wins / total * 100, 1) if total > 0 else 0,
        'total_pnl': round(total_pnl, 2),
    })

@app.route('/api/trades/clear', methods=['POST'])
def clear_trades():
    with _lock:
        BS['trade_history'] = []
        BS['pnl_history']   = []
    add_log('Histórico de trades limpo', 'warn')
    return jsonify({'success': True})

@app.route('/api/pnl_history')
def pnl_history_route():
    return jsonify({'history': BS['pnl_history'][-200:]})

@app.route('/api/news')
def news_route():
    return jsonify({
        'news': BS['news_cache'],
        'ts':   BS['news_ts'],
        'count': len(BS['news_cache']),
    })

@app.route('/api/open_orders')
def open_orders_route():
    if not BS['wallet']:
        return jsonify({'orders': []})
    orders = get_open_orders(BS['wallet'])
    return jsonify({'orders': orders or []})

@app.route('/api/fills')
def fills_route():
    if not BS['wallet']:
        return jsonify({'fills': []})
    fills = get_fills(BS['wallet'])
    fills = (fills or [])[:100]
    return jsonify({'fills': fills})

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print()
    print('╔════════════════════════════════════════════════╗')
    print('║     HyperBot Pro v2 — Trading Bot              ║')
    print('║     http://localhost:5000                      ║')
    print('║     API: http://localhost:5000/api             ║')
    print('╚════════════════════════════════════════════════╝')
    print()
    add_log('🚀 Servidor HyperBot Pro v2 iniciado na porta 5000', 'success')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# ── Serve frontend (index.html at root) ──────────────────────────
import os
from flask import send_from_directory

@app.route('/')
def frontend():
    here = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(here, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(here, filename)
