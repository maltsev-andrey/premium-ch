#!/usr/bin/env python3
"""
s05_backtest.py — premium-ch stage 5: rolling-origin backtest.

For each origin year T in ORIGINS, fit the s04 model on years <= T only
and forecast every canton's year T+1 standard premium. Score against the
actual. This answers the only question that matters for a forecasting
model: how often was it right, before it knew the answer.

Intervention handling (honest by design):
  - the ZG 2026/2027 intervention dummy is active ONLY for intervention
    years inside the training window; when the training window ends
    before 2026, the model knows nothing about the intervention
  - therefore the origin-2025 forecast of ZG 2026 is made WITHOUT
    intervention knowledge and is expected to miss badly — that miss is
    reported, not hidden: it quantifies what a pure data model cannot
    see coming (a policy decision), and motivates the dummy
  - from origin 2026 the dummy is identified from training data, and the
    ZG 2027 forecast applies it (2027 is a known intervention year)

Metrics per origin: absolute percentage error per canton, median APE,
and empirical coverage of the 50% and 90% predictive intervals.

Outputs:
    data/processed/backtest_results.csv   origin, canton, actual, quantiles, APE
    data/processed/backtest_summary.csv   per-origin MAPE + coverage
    figures/f06_backtest.png              APE distribution + coverage per origin

Usage (srv-anaconda-01, conda env premium-ch), ~10-15 min on CPU:
    python pipeline/s05_backtest.py
"""

import sys
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS

# reuse the s04 model definition — one model, fitted many ways
from s04_model_trend import model, INTERVENTION  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/nfs/shared/premium-ch")
IN_MED = BASE_DIR / "data" / "processed" / "canton_median.csv"
OUT_RES = BASE_DIR / "data" / "processed" / "backtest_results.csv"
OUT_SUM = BASE_DIR / "data" / "processed" / "backtest_summary.csv"
FIG_DIR = BASE_DIR / "figures"

ORIGINS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]  # train <= T, predict T+1
SEED = 20261110
N_WARMUP = 1000
N_SAMPLES = 1500
N_CHAINS = 4

GREY = "#B9BFC7"
INK = "#1F2430"
MUTED = "#6B7280"
BLUE = "#2563EB"
RED = "#DC2626"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# One origin
# ---------------------------------------------------------------------------


def fit_and_forecast(med: pd.DataFrame, origin: int, seed: int) -> pd.DataFrame:
    train = med[med["year"] <= origin].copy()
    target_year = origin + 1
    actual = med[med["year"] == target_year].set_index("canton")["median_premium"]
    if actual.empty:
        raise ValueError(f"no actuals for {target_year}")

    cantons = sorted(train["canton"].unique())
    years = sorted(train["year"].unique())
    c_map = {c: i for i, c in enumerate(cantons)}
    y_map = {y: i for i, y in enumerate(years)}
    t_center = float(np.mean(years))

    zg_mask = np.array([
        1.0 if (c in INTERVENTION and y in INTERVENTION[c]) else 0.0
        for c, y in zip(train["canton"], train["year"])
    ])
    intervention_in_train = bool(zg_mask.any())

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, progress_bar=False)
    mcmc.run(
        jrandom.PRNGKey(seed),
        canton_idx=jnp.array(train["canton"].map(c_map).to_numpy()),
        year_idx=jnp.array(train["year"].map(y_map).to_numpy()),
        t_centered=jnp.array((train["year"] - t_center).to_numpy(float)),
        zg_mask=jnp.array(zg_mask),
        n_cantons=len(cantons), n_years=len(years),
        y=jnp.array(np.log(train["median_premium"].to_numpy(float))),
    )
    post = mcmc.get_samples()
    n_draws = post["a"].shape[0]
    rng = np.random.default_rng(seed)

    f_next = np.asarray(post["f"][:, -1]) \
        + rng.normal(0.0, 1.0, n_draws) * np.asarray(post["sigma_f"])
    t_next = target_year - t_center
    nu = np.asarray(post["nu"])
    sigma = np.asarray(post["sigma"])
    delta_zg = np.asarray(post["delta_zg"])

    rows = []
    for ci, canton in enumerate(cantons):
        if canton not in actual.index:
            continue
        mu = np.asarray(post["a"][:, ci]) \
            + np.asarray(post["b"][:, ci]) * t_next + f_next
        # apply the intervention dummy to the forecast ONLY if the model
        # could have learned it from training data and T+1 is affected
        dummy_applied = (
            intervention_in_train
            and canton in INTERVENTION
            and target_year in INTERVENTION[canton]
        )
        if dummy_applied:
            mu = mu + delta_zg
        draw = np.exp(mu + rng.standard_t(nu) * sigma)
        q = np.percentile(draw, [5, 25, 50, 75, 95])
        act = float(actual[canton])
        rows.append({
            "origin": origin, "target_year": target_year, "canton": canton,
            "actual": act,
            "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
            "ape_pct": abs(q[2] / act - 1) * 100,
            "in50": int(q[1] <= act <= q[3]),
            "in90": int(q[0] <= act <= q[4]),
            "dummy_applied": int(dummy_applied),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    med = pd.read_csv(IN_MED)
    results = []
    for i, origin in enumerate(ORIGINS):
        log(f"origin {origin}: training on <= {origin}, predicting {origin + 1}")
        res = fit_and_forecast(med, origin, SEED + i)
        mape = res["ape_pct"].median()
        log(f"  median APE {mape:.2f}%, coverage50 {res['in50'].mean():.0%}, "
            f"coverage90 {res['in90'].mean():.0%}")
        results.append(res)

    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(OUT_RES, index=False)

    summary = (
        allres.groupby("origin")
        .agg(target_year=("target_year", "first"),
             n=("canton", "size"),
             median_ape_pct=("ape_pct", "median"),
             max_ape_pct=("ape_pct", "max"),
             worst_canton=("ape_pct", lambda s: allres.loc[s.idxmax(), "canton"]),
             coverage50=("in50", "mean"),
             coverage90=("in90", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT_SUM, index=False)
    log("summary:")
    for _, r in summary.iterrows():
        log(f"  {r.origin}->{int(r.target_year)}: "
            f"median APE {r.median_ape_pct:.2f}%, "
            f"max {r.max_ape_pct:.2f}% ({r.worst_canton}), "
            f"cov50 {r.coverage50:.0%}, cov90 {r.coverage90:.0%}")
    overall = allres[allres["dummy_applied"] == 0]
    log(f"overall (excl. dummy-informed ZG rows): "
        f"median APE {overall['ape_pct'].median():.2f}%, "
        f"cov90 {overall['in90'].mean():.0%} over {len(overall)} forecasts")

    # ZG special rows for the research findings
    zg = allres[allres["canton"] == "ZG"]
    for _, r in zg[zg["target_year"].isin([2026, 2027])].iterrows():
        log(f"  ZG {int(r.target_year)} (dummy_applied={r.dummy_applied}): "
            f"forecast {r.p50:.0f}, actual {r.actual:.0f}, APE {r.ape_pct:.1f}%")

    # ---- figure --------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    targets = [o + 1 for o in ORIGINS]
    data = [allres[allres["origin"] == o]["ape_pct"].values for o in ORIGINS]
    bp = ax1.boxplot(data, tick_labels=[str(t) for t in targets], widths=0.55,
                     patch_artist=True, medianprops=dict(color=RED, lw=2))
    for box in bp["boxes"]:
        box.set(facecolor="#DBEAFE", edgecolor=MUTED, lw=0.8)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set(color=MUTED, lw=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color=GREY, alpha=0.35, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.tick_params(colors=MUTED, labelsize=9)
    ax1.set_ylabel("abs. error of median forecast, %", color=MUTED, fontsize=9)
    ax1.set_title(
        "Rolling-origin backtest: forecast error per target year (26 cantons)",
        fontsize=11, color=INK, loc="left",
    )

    cov50 = [allres[allres["origin"] == o]["in50"].mean() * 100 for o in ORIGINS]
    cov90 = [allres[allres["origin"] == o]["in90"].mean() * 100 for o in ORIGINS]
    x = np.arange(1, len(ORIGINS) + 1)
    ax2.plot(x, cov90, "o-", color=BLUE, lw=1.8, label="90% interval")
    ax2.plot(x, cov50, "o-", color=MUTED, lw=1.8, label="50% interval")
    ax2.axhline(90, color=BLUE, lw=0.8, ls=":")
    ax2.axhline(50, color=MUTED, lw=0.8, ls=":")
    ax2.set_ylim(0, 105)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GREY, alpha=0.35, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.set_ylabel("empirical coverage, %", color=MUTED, fontsize=9)
    ax2.set_xlabel("target year", color=MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f06_backtest.png", dpi=150)
    plt.close(fig)
    log(f"figure written: {FIG_DIR / 'f06_backtest.png'}")
    log(f"results: {OUT_RES}, summary: {OUT_SUM}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
