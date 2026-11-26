"""每日產生美元指數技術圖（布林/KD/威廉/MACD）並推播到 LINE。
用法：python dxy_line.py chart   -> 產生圖
      python dxy_line.py send    -> 推播到 LINE（圖需已上傳到公開網址）
"""
import os, sys, datetime as dt
import pandas as pd, requests, yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 參數（可自行調整）=====
TICKER = "DX-Y.NYB"
BB_N, BB_K = 20, 2
KD_N, KD_M1, KD_M2 = 9, 3, 3
WR_N = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 1, 9, 9
BARS = 120  # 圖上顯示最近幾根日K

TODAY = dt.date.today().strftime("%Y%m%d")
OUT_DIR = "charts"
OUT_FILE = f"{OUT_DIR}/dxy_{TODAY}.jpg"


def get_data():
    df = yf.download(TICKER, period="2y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def add_indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    # 布林通道
    df["MA"] = c.rolling(BB_N).mean()
    sd = c.rolling(BB_N).std(ddof=0)
    df["UP"], df["DN"] = df["MA"] + BB_K * sd, df["MA"] - BB_K * sd
    # KD
    rsv = (c - l.rolling(KD_N).min()) / (h.rolling(KD_N).max() - l.rolling(KD_N).min()) * 100
    df["K"] = rsv.ewm(alpha=1 / KD_M1, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / KD_M2, adjust=False).mean()
    # 威廉 %R
    hh, ll = h.rolling(WR_N).max(), l.rolling(WR_N).min()
    df["WR"] = (hh - c) / (hh - ll) * -100
    # MACD
    dif = c.ewm(span=MACD_FAST, adjust=False).mean() - c.ewm(span=MACD_SLOW, adjust=False).mean()
    df["DIF"] = dif
    df["DEA"] = dif.ewm(span=MACD_SIG, adjust=False).mean()
    df["HIST"] = df["DIF"] - df["DEA"]
    return df


def draw(df):
    d = df.tail(BARS)
    x = range(len(d))
    fig, ax = plt.subplots(4, 1, figsize=(10, 13), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 1.3, 1.5]})
    # 價格 + 布林
    ax[0].plot(x, d["Close"], color="black", lw=1.2, label="Close")
    ax[0].plot(x, d["MA"], color="orange", lw=1, label=f"MA{BB_N}")
    ax[0].plot(x, d["UP"], color="red", lw=0.8, label="Upper")
    ax[0].plot(x, d["DN"], color="green", lw=0.8, label="Lower")
    ax[0].fill_between(x, d["UP"], d["DN"], alpha=0.08)
    ax[0].set_title(f"DXY  {d.index[-1].date()}  Close {d['Close'].iloc[-1]:.2f}   "
                    f"BB({BB_N},{BB_K})", loc="left")
    ax[0].legend(loc="upper left", ncol=4, fontsize=8)
    # KD
    ax[1].plot(x, d["K"], color="blue", label="K"); ax[1].plot(x, d["D"], color="red", label="D")
    ax[1].axhline(80, ls="--", c="gray", lw=.6); ax[1].axhline(20, ls="--", c="gray", lw=.6)
    ax[1].set_title(f"KD({KD_N},{KD_M1},{KD_M2})  K={d['K'].iloc[-1]:.1f} D={d['D'].iloc[-1]:.1f}", loc="left", fontsize=9)
    ax[1].legend(loc="upper left", ncol=2, fontsize=8)
    # 威廉
    ax[2].plot(x, d["WR"], color="purple")
    ax[2].axhline(-20, ls="--", c="gray", lw=.6); ax[2].axhline(-80, ls="--", c="gray", lw=.6)
    ax[2].set_title(f"Williams %R({WR_N})  {d['WR'].iloc[-1]:.1f}", loc="left", fontsize=9)
    # MACD
    colors = ["red" if v >= 0 else "green" for v in d["HIST"]]
    ax[3].bar(x, d["HIST"], color=colors, width=0.8)
    ax[3].plot(x, d["DIF"], color="blue", lw=1, label="DIF")
    ax[3].plot(x, d["DEA"], color="orange", lw=1, label="DEA")
    ax[3].axhline(0, c="gray", lw=.6)
    ax[3].set_title(f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIG})", loc="left", fontsize=9)
    ax[3].legend(loc="upper left", ncol=2, fontsize=8)
    # X 軸日期
    step = max(len(d) // 8, 1)
    ax[3].set_xticks(list(x)[::step])
    ax[3].set_xticklabels([t.strftime("%m/%d") for t in d.index[::step]])
    for a in ax: a.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=80, format="jpg", pil_kwargs={"quality": 85})
    print("saved", OUT_FILE)


def send_line():
    token = os.environ["LINE_CHANNEL_TOKEN"]
    user_id = os.environ["LINE_USER_ID"]
    repo = os.environ["GITHUB_REPOSITORY"]  # 例如 yourname/dxy-line（需為公開 repo）
    url = f"https://raw.githubusercontent.com/{repo}/main/{OUT_FILE}"
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {token}"},
        json={"to": user_id, "messages": [
            {"type": "text", "text": f"📊 美元指數技術圖 {TODAY}"},
            {"type": "image", "originalContentUrl": url, "previewImageUrl": url},
        ]},
        timeout=30,
    )
    print(r.status_code, r.text)
    r.raise_for_status()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chart"
    if mode == "chart":
        draw(add_indicators(get_data()))
    elif mode == "send":
        send_line()
