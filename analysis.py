"""重現論文中的統計檢定與回測數字。

用法：python analysis.py
需要：numpy、scipy
資料：data/taiex_daily.csv（臺灣證券交易所 TAIEX 每日開盤、收盤，2026/10/1–11/12）
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

rows = list(csv.DictReader(open(Path(__file__).parent / "data" / "taiex_daily.csv", encoding="utf-8-sig")))
for r in rows:
    r["d"] = date.fromisoformat(r["date"])
    for k in ("open", "close", "prev_close", "return_pct", "return_points"):
        r[k] = float(r[k])
    r["thu"] = r["is_thursday"] == "1"

thu = np.array([r["return_pct"] for r in rows if r["thu"]])
oth = np.array([r["return_pct"] for r in rows if not r["thu"]])
thu_pt = np.array([r["return_points"] for r in rows if r["thu"]])
oth_pt = np.array([r["return_points"] for r in rows if not r["thu"]])

print("== 敘述統計 ==")
for name, a in (("週四播映日", thu), ("其他交易日", oth)):
    print(f"{name}: n={len(a)} 平均={a.mean():+.2f}% 中位數={np.median(a):+.2f}% "
          f"標準差={a.std(ddof=1):.2f}% 下跌日數={(a < 0).sum()} 合計={a.sum():+.2f}")
print(f"點數平均：週四 {thu_pt.mean():+.2f}，其他 {oth_pt.mean():+.2f}")

print("\n== 假說檢定 ==")
t, p = stats.ttest_ind(thu, oth, equal_var=False)
va, vb = thu.var(ddof=1) / len(thu), oth.var(ddof=1) / len(oth)
df = (va + vb) ** 2 / (va ** 2 / (len(thu) - 1) + vb ** 2 / (len(oth) - 1))
diff, se = thu.mean() - oth.mean(), np.sqrt(va + vb)
tc = stats.t.ppf(0.975, df)
print(f"Welch t（百分比）: t={t:.2f} df={df:.1f} p={p:.3f}")
t2, p2 = stats.ttest_ind(thu_pt, oth_pt, equal_var=False)
print(f"Welch t（點數）: t={t2:.2f} p={p2:.3f}")
t3, p3 = stats.ttest_1samp(thu, 0)
print(f"單樣本 t: t={t3:.2f} p={p3:.3f}")
u = stats.mannwhitneyu(thu, oth, alternative="two-sided")
print(f"Mann–Whitney U: U={u.statistic:.0f} p={u.pvalue:.3f} "
      f"P(週四<其他)={1 - u.statistic / (len(thu) * len(oth)):.2f}")

rng = np.random.default_rng(0)
allv = np.concatenate([thu, oth])
n, N, cnt = len(thu), 200_000, 0
for _ in range(N):
    s = rng.permutation(allv)
    if abs(s[:n].mean() - s[n:].mean()) >= abs(diff):
        cnt += 1
print(f"排列檢定（{N:,} 次）: 平均差={diff:.2f} p={cnt / N:.3f}")

oth_ex = np.array([r["return_pct"] for r in rows if not r["thu"] and r["d"] != date(2026, 11, 6)])
t4, p4 = stats.ttest_ind(thu, oth_ex, equal_var=False)
print(f"Welch t，剔除 11/6: t={t4:.2f} p={p4:.3f}")

print("\n== 效果量 ==")
sp = np.sqrt(((len(thu) - 1) * thu.var(ddof=1) + (len(oth) - 1) * oth.var(ddof=1)) / (len(thu) + len(oth) - 2))
print(f"平均差={diff:.2f} 95% CI=[{diff - tc * se:.2f}, {diff + tc * se:.2f}] Cohen's d={diff / sp:.2f}")
rng = np.random.default_rng(3)
bs = [rng.choice(thu, len(thu)).mean() - rng.choice(oth, len(oth)).mean() for _ in range(100_000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"拔靴法 95% CI=[{lo:.2f}, {hi:.2f}]")

print("\n== 逐一剔除單一週四 ==")
thu_days = [r["d"] for r in rows if r["thu"]]
for i, d in enumerate(thu_days):
    a = np.delete(thu, i)
    tt, pp = stats.ttest_ind(a, oth, equal_var=False)
    print(f"剔除 {d.month}/{d.day}: t={tt:.2f} p={pp:.4f}")

print("\n== 回測 ==")
O = {r["d"]: r["open"] for r in rows}
C = {r["d"]: r["close"] for r in rows}
D = lambda m, d: date(2026, m, d)
# 做多：週五開盤買進（週五休市則週四收盤），次週三收盤賣出
longs = [((D(10, 2), "o"), D(10, 7)), ((D(10, 8), "c"), D(10, 14)), ((D(10, 16), "o"), D(10, 21)),
         ((D(10, 22), "c"), D(10, 28)), ((D(10, 30), "o"), D(11, 4)), ((D(11, 6), "o"), D(11, 11))]
# 反手放空：週三收盤放空，下次做多進場時回補；最後一筆以 11/12 收盤暫計
shorts = [(D(10, 7), (D(10, 8), "c")), (D(10, 14), (D(10, 16), "o")), (D(10, 21), (D(10, 22), "c")),
          (D(10, 28), (D(10, 30), "o")), (D(11, 4), (D(11, 6), "o")), (D(11, 11), (D(11, 12), "c"))]
px = lambda x: (O if x[1] == "o" else C)[x[0]]
lr = [C[s] / px(b) - 1 for b, s in longs]
sr = [(C[e] - px(x)) / C[e] for e, x in shorts]
for (b, s), r_ in zip(longs, lr):
    print(f"做多 {b[0].month}/{b[0].day}{'開' if b[1] == 'o' else '收'} → {s.month}/{s.day}收: {r_ * 100:+.2f}%")
for (e, x), r_ in zip(shorts, sr):
    print(f"放空 {e.month}/{e.day}收 → {x[0].month}/{x[0].day}{'開' if x[1] == 'o' else '收'}: {r_ * 100:+.2f}%")
lt, st = np.prod([1 + x for x in lr]) - 1, np.prod([1 + x for x in sr]) - 1
bh = rows[-1]["close"] / rows[0]["prev_close"] - 1
print(f"做多累計={lt * 100:+.2f}% 放空累計={st * 100:+.2f}% 合併={((1 + lt) * (1 + st) - 1) * 100:+.2f}% "
      f"買進持有={bh * 100:+.2f}%")
gap = np.mean([(r["open"] / r["prev_close"] - 1) * 100 for r in rows if r["thu"]])
intr = np.mean([(r["close"] / r["open"] - 1) * 100 for r in rows if r["thu"]])
print(f"週四隔夜跳空平均={gap:+.2f}% 盤中平均={intr:+.2f}%")
