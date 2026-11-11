#!/usr/bin/env python3
"""
s04_model_trend.py — premium-ch stage 4: hierarchical Bayesian trend model
and 2028 cantonal forecast (NumPyro / NUTS, CPU).

Model (log scale, y[c,t] = log median standard premium):

    y[c,t] ~ StudentT(nu, mu[c,t], sigma)
    mu[c,t] = a[c] + b[c] * (t - t_center) + f[t] + delta_zg * I[c=ZG, t in 2026/2027]

    a[c]  canton intercept           ~ Normal(a0, tau_a)   (partial pooling)
    b[c]  canton-specific log-slope  ~ Normal(b0, tau_b)   (partial pooling)
    f[t]  national year effect, random walk:
          f[0] = 0;  f[t] = f[t-1] + eps[t],  eps ~ Normal(0, sigma_f)
          (sum of drift is carried by b0; f captures common year shocks:
           2022 freeze, 2024 spike)
    delta_zg  Zug 2026/2027 intervention (canton covers 99% of inpatient
          hospital costs, ~220M CHF; announced to END in 2028)

Forecast 2028: f[T+1] = f[T] + eps_new, intervention dummy = 0
    -> the ZG snap-back is predicted by model structure, not hand-tuning.

Outputs:
    data/processed/forecast_2028.csv       per canton: mean, 5/25/50/75/95%
    data/processed/posterior_summary.csv   parameter summaries + diagnostics
    models/trend_posterior.nc              full posterior (ArviZ netcdf)
    figures/f05_forecast_2028.png          forecast intervals vs 2027 actuals

Usage (srv-anaconda-01, conda env premium-ch — needs numpyro, arviz):
    python pipeline/s04_model_trend.py
"""

import sys
from datetime import datetime
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, Predictive

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/nfs/shared/premium-ch")
IN_MED = BASE_DIR / "data" / "processed" / "canton_median.csv"
OUT_FC = BASE_DIR / "data" / "processed" / "forecast_2028.csv"
OUT_SUM = BASE_DIR / "data" / "processed" / "posterior_summary.csv"
OUT_NC = BASE_DIR / "models" / "trend_posterior.nc"
FIG_DIR = BASE_DIR / "figures"

SEED = 20261109
N_WARMUP = 1500
N_SAMPLES = 2000
N_CHAINS = 4

INTERVENTION = {"ZG": [2026, 2027]}  # canton -> affected premium years
FORECAST_YEAR = 2028

numpyro.set_host_device_count(N_CHAINS)

GREY = "#B9BFC7"
INK = "#1F2430"
MUTED = "#6B7280"
BLUE = "#2563EB"
RED = "#DC2626"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def model(canton_idx, year_idx, t_centered, zg_mask, n_cantons, n_years, y=None):
    a0 = numpyro.sample("a0", dist.Normal(6.0, 1.0))
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(0.5))
    b0 = numpyro.sample("b0", dist.Normal(0.04, 0.05))     # ~4%/yr prior
    tau_b = numpyro.sample("tau_b", dist.HalfNormal(0.02))

    # non-centered parameterization (avoids the funnel at small tau)
    with numpyro.plate("cantons", n_cantons):
        z_a = numpyro.sample("z_a", dist.Normal(0.0, 1.0))
        z_b = numpyro.sample("z_b", dist.Normal(0.0, 1.0))
    a = numpyro.deterministic("a", a0 + z_a * tau_a)
    b = numpyro.deterministic("b", b0 + z_b * tau_b)

    sigma_f = numpyro.sample("sigma_f", dist.HalfNormal(0.03))
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([n_years - 1]))
    f = jnp.concatenate([jnp.zeros(1), jnp.cumsum(eps * sigma_f)])
    numpyro.deterministic("f", f)

    delta_zg = numpyro.sample("delta_zg", dist.Normal(0.0, 0.3))

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.05))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    mu = a[canton_idx] + b[canton_idx] * t_centered + f[year_idx] \
        + delta_zg * zg_mask
    numpyro.sample("obs", dist.StudentT(nu, mu, sigma), obs=y)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def prepare():
    med = pd.read_csv(IN_MED)
    med = med.sort_values(["canton", "year"]).reset_index(drop=True)
    cantons = sorted(med["canton"].unique())
    years = sorted(med["year"].unique())
    c_map = {c: i for i, c in enumerate(cantons)}
    y_map = {y: i for i, y in enumerate(years)}

    t_center = float(np.mean(years))
    data = {
        "canton_idx": med["canton"].map(c_map).to_numpy(),
        "year_idx": med["year"].map(y_map).to_numpy(),
        "t_centered": (med["year"] - t_center).to_numpy(float),
        "zg_mask": np.array([
            1.0 if (c in INTERVENTION and y in INTERVENTION[c]) else 0.0
            for c, y in zip(med["canton"], med["year"])
        ]),
        "y_obs": np.log(med["median_premium"].to_numpy(float)),
    }
    return med, cantons, years, t_center, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    med, cantons, years, t_center, d = prepare()
    n_c, n_y = len(cantons), len(years)
    log(f"{len(med)} observations, {n_c} cantons, years {years[0]}-{years[-1]}")

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, progress_bar=False)
    log(f"running NUTS: {N_CHAINS} chains x ({N_WARMUP}+{N_SAMPLES})")
    mcmc.run(
        jrandom.PRNGKey(SEED),
        canton_idx=jnp.array(d["canton_idx"]),
        year_idx=jnp.array(d["year_idx"]),
        t_centered=jnp.array(d["t_centered"]),
        zg_mask=jnp.array(d["zg_mask"]),
        n_cantons=n_c, n_years=n_y,
        y=jnp.array(d["y_obs"]),
    )

    idata = az.from_numpyro(mcmc)
    # f[0] is a constant (zero variance) — summarize eps, not f,
    # to keep r_hat/ESS well defined for every row
    summ = az.summary(idata, var_names=["a0", "b0", "tau_a", "tau_b",
                                        "sigma_f", "delta_zg", "sigma", "nu",
                                        "a", "b", "eps"])
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT_SUM)
    bad_rhat = summ[summ["r_hat"] > 1.01]
    n_div = int(idata.sample_stats["diverging"].values.sum())
    log(f"diagnostics: max r_hat={summ['r_hat'].max():.3f}, "
        f"min ESS_bulk={summ['ess_bulk'].min():.0f}, divergences={n_div}")
    if len(bad_rhat) or n_div:
        log(f"WARNING: {len(bad_rhat)} params r_hat>1.01, {n_div} divergences "
            f"— inspect {OUT_SUM} before trusting the forecast")

    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    try:
        idata.to_netcdf(OUT_NC)
        log(f"posterior saved: {OUT_NC}")
    except (ValueError, ImportError) as e:
        log(f"WARNING: posterior not saved ({e}) — "
            f"install a netcdf backend: conda install -c conda-forge h5netcdf")

    # ---- forecast 2028 ------------------------------------------------
    post = mcmc.get_samples()
    n_draws = post["a"].shape[0]
    rng = np.random.default_rng(SEED)

    f_T = np.asarray(post["f"][:, -1])
    sigma_f = np.asarray(post["sigma_f"])
    f_next = f_T + rng.normal(0.0, 1.0, n_draws) * sigma_f
    t_next = FORECAST_YEAR - t_center

    nu = np.asarray(post["nu"])
    sigma = np.asarray(post["sigma"])
    rows = []
    last = med[med["year"] == years[-1]].set_index("canton")["median_premium"]
    for ci, canton in enumerate(cantons):
        a_c = np.asarray(post["a"][:, ci])
        b_c = np.asarray(post["b"][:, ci])
        mu = a_c + b_c * t_next + f_next          # intervention dummy = 0
        draw = mu + rng.standard_t(nu) * sigma     # predictive noise
        chf = np.exp(draw)
        q = np.percentile(chf, [5, 25, 50, 75, 95])
        growth = (q[2] / last[canton] - 1) * 100
        rows.append({
            "canton": canton, "forecast_year": FORECAST_YEAR,
            "mean_chf": float(np.mean(chf)),
            "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
            "last_actual_chf": float(last[canton]),
            "implied_growth_pct": float(growth),
        })
    fc = pd.DataFrame(rows).sort_values("p50", ascending=False)
    fc.to_csv(OUT_FC, index=False)
    log(f"forecast written: {OUT_FC}")

    zg = fc[fc.canton == "ZG"].iloc[0]
    log(f"headline: ZG 2028 median forecast {zg.p50:.0f} CHF "
        f"({zg.implied_growth_pct:+.1f}% vs 2027) — intervention snap-back")
    dz = np.asarray(post["delta_zg"])
    log(f"delta_zg (log scale): mean {dz.mean():+.3f}, "
        f"94% HDI [{np.percentile(dz, 3):+.3f}, {np.percentile(dz, 97):+.3f}] "
        f"= {100 * (np.exp(dz.mean()) - 1):+.1f}% premium effect")

    # ---- figure --------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fc_plot = fc.sort_values("p50")
    fig, ax = plt.subplots(figsize=(8, 9))
    ypos = np.arange(len(fc_plot))
    ax.hlines(ypos, fc_plot["p05"], fc_plot["p95"], color=GREY, lw=2)
    ax.hlines(ypos, fc_plot["p25"], fc_plot["p75"], color=BLUE, lw=5, alpha=0.7)
    ax.plot(fc_plot["p50"], ypos, "o", color=INK, ms=5, zorder=3,
            label="2028 forecast median")
    ax.plot(fc_plot["last_actual_chf"], ypos, "o", mfc="none", mec=RED,
            ms=6, mew=1.6, zorder=3, label=f"{years[-1]} actual")
    ax.set_yticks(ypos, fc_plot["canton"], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", color=GREY, alpha=0.35, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("standard premium, CHF/month", color=MUTED, fontsize=9)
    ax.set_title(
        f"Forecast {FORECAST_YEAR}: cantonal standard premium\n"
        "bands = 50% and 90% posterior predictive intervals",
        fontsize=11, color=INK, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f05_forecast_2028.png", dpi=150)
    plt.close(fig)
    log(f"figure written: {FIG_DIR / 'f05_forecast_2028.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
