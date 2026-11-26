"""每日產生 美元指數(DXY) 與 美元/台幣(USDTWD) 技術圖並推播到 LINE。
指標：布林(20,2)、KD(9,3,3)、威廉%R(9)、MACD(1,9,收盤,9)
用法：python dxy_line.py chart   -> 產生圖
      python dxy_line.py send    -> 推播到 LINE（圖需已上傳到公開網址）
"""
import os, sys, glob, time, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests, yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 標的 =====（代號, Yahoo 代碼, 小數位數）
ASSETS = [
    ("DXY", "DX-Y.NYB", 2),
    ("USDTWD", "TWD=X", 3),
]

# ===== 指標參數（可自行調整）=====
BB_N, BB_K = 20, 2
KD_N, KD_M1, KD_M2 = 9, 3, 3
WR_N = 9
MACD_FAST, MACD_SLOW, MACD_SIG = 1, 9, 9   # 快線 1 = 收盤價本身
BARS = 120          # 圖上顯示最近幾根日K
KEEP_DAYS = 14      # repo 內只保留最近幾天的圖

TODAY = dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d")
OUT_DIR = "charts"


def out_file(name):
    return f"{OUT_DIR}/{name}_{TODAY}.jpg"


def get_data(ticker):
    df = None
    for _ in range(3):  # Yahoo 偶爾失敗，重試
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
        if df is not None and len(df) > 0:
            break
        time.sleep(5)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def add_indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    df["MA"] = c.rolling(BB_N).mean()
    sd = c.rolling(BB_N).std(ddof=0)
    df["UP"], df["DN"] = df["MA"] + BB_K * sd, df["MA"] - BB_K * sd
    rsv = (c - l.rolling(KD_N).min()) / (h.rolling(KD_N).max() - l.rolling(KD_N).min()) * 100
    df["K"] = rsv.ewm(alpha=1 / KD_M1, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / KD_M2, adjust=False).mean()
    hh, ll = h.rolling(WR_N).max(), l.rolling(WR_N).min()
    df["WR"] = (hh - c) / (hh - ll) * -100
    dif = c.ewm(span=MACD_FAST, adjust=False).mean() - c.ewm(span=MACD_SLOW, adjust=False).mean()
    df["DIF"] = dif
    df["DEA"] = dif.ewm(span=MACD_SIG, adjust=False).mean()
    df["HIST"] = df["DIF"] - df["DEA"]
    return df


def draw(df, name, dec):
    d = df.tail(BARS)
    x = np.arange(len(d))
    fig, ax = plt.subplots(4, 1, figsize=(10, 13), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 1.3, 1.5]})
    # ---- 主圖：K線（紅漲綠跌）+ 布林通道 ----
    o, hi, lo, cl = d["Open"], d["High"], d["Low"], d["Close"]
    col = ["red" if c >= op else "green" for c, op in zip(cl, o)]
    span = float(hi.max() - lo.min())
    ax[0].vlines(x, lo, hi, colors=col, lw=0.8)
    ax[0].bar(x, (cl - o).abs().clip(lower=span * 0.002),
              bottom=np.minimum(o, cl), color=col, width=0.6)
    ax[0].plot(x, d["MA"], color="orange", lw=1.2)
    ax[0].plot(x, d["UP"], color="royalblue", lw=0.9, ls="--")
    ax[0].plot(x, d["DN"], color="royalblue", lw=0.9, ls="--")
    ax[0].fill_between(x, d["UP"], d["DN"], alpha=0.06, color="royalblue")
    ax[0].set_title(f"{name}  {d.index[-1].date()}  Close {cl.iloc[-1]:.{dec}f}   "
                    f"BB({BB_N},{BB_K})", loc="left")
    # 最新一根的數值標示（右側，避免互相重疊）
    last = len(d) - 1
    items = [("Close", cl.iloc[-1], "black"),
             ("Upper", d["UP"].iloc[-1], "royalblue"),
             (f"MA{BB_N}", d["MA"].iloc[-1], "orange"),
             ("Lower", d["DN"].iloc[-1], "royalblue")]
    items.sort(key=lambda t: -t[1])
    gap, prev = span * 0.06, None
    for label, val, colr in items:
        y = val if prev is None or prev - val >= gap else prev - gap
        prev = y
        ax[0].annotate(f"{label} {val:.{dec}f}", xy=(last, val), xytext=(last + 2, y),
                       color=colr, fontsize=9, fontweight="bold", va="center",
                       arrowprops=dict(arrowstyle="-", color=colr, lw=0.6),
                       annotation_clip=False)
    ax[0].axhline(cl.iloc[-1], color="black", lw=0.5, ls=":")
    ax[0].set_xlim(-1, len(d) + 17)
    ax[1].plot(x, d["K"], color="blue", label="K"); ax[1].plot(x, d["D"], color="red", label="D")
    ax[1].axhline(80, ls="--", c="gray", lw=.6); ax[1].axhline(20, ls="--", c="gray", lw=.6)
    ax[1].set_title(f"KD({KD_N},{KD_M1},{KD_M2})  K={d['K'].iloc[-1]:.1f} D={d['D'].iloc[-1]:.1f}",
                    loc="left", fontsize=9)
    ax[1].legend(loc="upper left", ncol=2, fontsize=8)
    ax[2].plot(x, d["WR"], color="purple")
    ax[2].axhline(-20, ls="--", c="gray", lw=.6); ax[2].axhline(-80, ls="--", c="gray", lw=.6)
    ax[2].set_title(f"Williams %R({WR_N})  {d['WR'].iloc[-1]:.1f}", loc="left", fontsize=9)
    colors = ["red" if v >= 0 else "green" for v in d["HIST"]]
    ax[3].bar(x, d["HIST"], color=colors, width=0.8)
    ax[3].plot(x, d["DIF"], color="blue", lw=1, label="DIF")
    ax[3].plot(x, d["DEA"], color="orange", lw=1, label="DEA")
    ax[3].axhline(0, c="gray", lw=.6)
    ax[3].set_title(f"MACD({MACD_FAST},{MACD_SLOW},Close,{MACD_SIG})", loc="left", fontsize=9)
    ax[3].legend(loc="upper left", ncol=2, fontsize=8)
    step = max(len(d) // 8, 1)
    ax[3].set_xticks(list(x)[::step])
    ax[3].set_xticklabels([t.strftime("%m/%d") for t in d.index[::step]])
    for a in ax:
        a.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_file(name), dpi=80, format="jpg", pil_kwargs={"quality": 85})
    plt.close(fig)
    print("saved", out_file(name))


def cleanup():
    cutoff = time.time() - KEEP_DAYS * 86400
    for f in glob.glob(f"{OUT_DIR}/*.jpg"):
        if os.path.getmtime(f) < cutoff:
            os.remove(f)


def send_line():
    token = os.environ["LINE_CHANNEL_TOKEN"]
    user_id = os.environ["LINE_USER_ID"]
    repo = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("GITHUB_REF_NAME", "main")
    msgs = [{"type": "text", "text": f"📊 美元指數 / 美元台幣 技術圖 {TODAY}"}]
    for name, _, _ in ASSETS:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{out_file(name)}"
        msgs.append({"type": "image", "originalContentUrl": url, "previewImageUrl": url})
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {token}"},
        json={"to": user_id, "messages": msgs},
        timeout=30,
    )
    print(r.status_code, r.text)
    r.raise_for_status()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chart"
    if mode == "chart":
        for name, ticker, dec in ASSETS:
            draw(add_indicators(get_data(ticker)), name, dec)
        cleanup()
    elif mode == "send":
        send_line()
