# premium-ch — research findings (v1.0)

Scope: 13 premium years (2015-2027) of the BAG priminfo dataset, standard
premium definition (adult, CHF 300 franchise, base tariff, with accident),
cantonal medians across insurers and premium regions. Model and backtest as
described in the README. All numbers reproducible from the pipeline.

## F1. The Zug intervention is measurable, and encoding it is worth 13 points of accuracy

For 2026-2027 the canton of Zug covers 99% of inpatient hospital costs from
cantonal surpluses (~220M CHF), and insurers priced this into premiums.

- Posterior intervention effect on the standard premium: **-12.1%**
  (94% HDI [-13.6%, -10.5%]), vs the officially reported -14.7% on the
  all-models mean premium — consistent, with the gap explained by the
  intervention cutting deeper into managed-care models.
- Backtest, Zug 2026, model trained only on data through 2025 (no policy
  knowledge): forecast 490 CHF, actual 430 CHF — **13.9% error**, the worst
  of all 182 scored forecasts.
- Backtest, Zug 2027, intervention encoded as a dummy identified from 2026
  training data: forecast 445 CHF, actual 444 CHF — **0.2% error**, the best.

Conclusion: a pure data model cannot anticipate a parliamentary decision.
One published policy fact, encoded as a single parameter, moved the affected
forecast from worst to best. For premium forecasting, reading cantonal
politics is part of the method, not a nicety.

## F2. The falsifiable prediction: Zug snaps back in 2028

The intervention is legislated to expire after 2027. Removing the dummy from
the forecast year yields:

> **Zug 2028: 522 CHF/month (90% interval 503-545), +17.6% vs 2027.**

Zug's government additionally expects 2028 to carry cost growth accumulated
during the subsidized years, so this trend-level forecast is plausibly a
floor. To be scored publicly against the BAG publication in September 2027.

## F3. The 90% intervals are honest

Across 181 out-of-sample forecasts (dummy-informed Zug rows excluded), the
90% predictive interval covered the actual **89%** of the time. Median
absolute error of the point forecast: **2.11%**. Accuracy improves with the
training window, from 2.56% (origin 2020) to 0.60% (origin 2026) median APE.

## F4. The model lags regime shifts by one year — in both directions

- Targets 2021-2022: systematic over-prediction. Trained on the
  high-growth 2015-2019 period, the model extrapolated ~3.5-4%/yr into the
  2020-2022 plateau (2022: cantonal median growth -0.05%, the
  reserve-reduction year).
- Target 2024: systematic under-prediction. After the plateau, the model
  did not see the +6.9% median spike coming; all 26 actuals landed above
  the median forecast (50% coverage: 0%).

This is structural for any trend model: turning points are visible only
after one observation of the new regime. A cost-driver layer (health cost
growth leads premium growth) is the v1.2 route to earlier turning-point
detection.

## F5. Cantonal forecast errors are one national bet plus 26 small ones

The dominant forecast uncertainty is the shared national year effect f[t].
When it misses, all cantons miss the same side — which is why the 50%
interval's empirical coverage swings between 0% (2024) and 73% (2026-2027)
instead of hovering near 50%, while the wider 90% interval absorbs the
common shock. Users of the forecast should read the 26 cantonal intervals
as strongly correlated, not independent.

## F6. Ticino is the structural outlier

TI was the worst-forecast canton in 3 of 7 backtest years (2024: 6.4%,
2025: 5.3%, 2027: 4.8% error), always under-predicted. Its premium growth
(e.g. +9.7% in 2024, the national maximum) persistently outruns its fitted
slope. The constant-slope assumption is wrong for TI; a time-varying or
breakpoint slope is a v1.1 candidate.

## F7. Market consolidation, visible in one chart

Insurers offering the base tariff fell monotonically from **59 (2015) to
32 (2027)** — a halving in 12 years, with no single-year discontinuity.
Premium dispersion within cantons has not collapsed correspondingly
(see f01), suggesting consolidation absorbed small carriers rather than
removing price competition among the remaining ones. Not modeled in v1.0.

## F8. Dataset findings (for anyone reusing priminfo)

- **2027 code-system rename**: age classes (`AKL-ERW` → `AKA_03_ERW`),
  franchise levels (`FRAST1` → `FRASTU_01`), accident flags
  (`MIT-UNF` → `MIT_UNF`) and tariff types (`TAR-BASE` → `BASE`) all
  changed. Base-level mappings were proven by count identities
  (`FRASTU_01` count == `isBaseF=1` count; `BASE` count == `isBaseP=1`
  count). The non-base tariff taxonomy (`FLEX`/`PRAXIS`/`TEL_DIG` vs
  `TAR-DIV`/`TAR-HAM`/`TAR-HMO`) does not map 1:1 and was not forced.
- **2024 has two validity periods** (01.01-31.08 and 01.09-31.12) — a
  mid-year premium adjustment. Mixing them silently corrupts growth rates
  for both 2024 and 2025; this project uses the January 1 snapshot
  throughout.
- **Format drift**: the 2015 file is comma-separated, later files
  semicolon; encodings vary between cp1252 and UTF-8. Per-file sniffing is
  mandatory.
- **`C_ID` special codes**: besides the 26 cantons, `CH` appears in all
  years and `ZE`/`ZR` from 2015 (~120-170 rows/year). Their semantics are
  documented in BAG's "Erläuterungen zu den Prämiendaten"; they are
  excluded from cantonal modeling here via an explicit scope flag.
- The 2011-2014 archives use a fully coded schema (`G_ID`, `C_ID`, `M_ID`,
  `VAR_ID`, ...) and are downloaded but not yet harmonized.

## Limitations register

1. Unweighted median across insurers and regions (enrollment weights only
   published for the current year).
2. Standard premium != mittlere Prämie; levels differ by construction,
   growth rates track closely (2026: our +3.3% median vs BAG +4.4% mean;
   2024: +6.9% vs +8.7%).
3. No exogenous cost drivers; premiums are forecast from premiums.
4. One-year horizon only; multi-year forecasts would compound the
   random-walk variance and have not been validated.
