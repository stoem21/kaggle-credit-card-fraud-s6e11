"""CV AUC vs public leaderboard AUC for each submission (numbers from EXPERIMENTS.md) -> figures/cv_vs_lb.png
Run: .venv/bin/python -m eda.plot_cv_vs_lb
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fraud.config import ROOT

SUBS = [("1  LightGBM + TE", 0.94719, 0.94837), ("2  shallow trees + MLP blend", 0.94990, 0.94997),
        ("3  + EBM, 7 CV splits", 0.95030, 0.95027), ("4  stack (final)", 0.95121, 0.95121)]
SURFACE, INK, INK2, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
lo, hi = 0.9465, 0.9520
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.annotate("dashed line: CV = leaderboard", (0.9490, 0.9490), xytext=(10, -14), textcoords="offset points",
            color=MUTED, fontsize=8)
for name, cv, lb in SUBS:
    ax.scatter(cv, lb, s=60, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
    right = cv < 0.948  # first point: label to the right, away from the y axis
    ax.annotate(name, (cv, lb), xytext=(8, 6) if right else (-8, 6), textcoords="offset points",
                ha="left" if right else "right", fontsize=8.5, color=INK2)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("cross-validation AUC (our own estimate)", color=INK2, fontsize=9)
ax.set_ylabel("public leaderboard AUC (Kaggle)", color=INK2, fontsize=9)
ax.set_title("Cross-validation predicted every leaderboard score\nwithin 0.0012 AUC",
             color=INK, fontsize=11, loc="left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
out = ROOT / "figures" / "cv_vs_lb.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out.relative_to(ROOT))
