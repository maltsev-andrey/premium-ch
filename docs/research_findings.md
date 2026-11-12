# premium-ch — research findings (v1.1)

Scope: 17 premium years (2011-2027) of the BAG priminfo dataset, standard
premium definition (adult, CHF 300 franchise, base tariff, with accident),
cantonal medians across insurers and premium regions. Model and backtest as
described in the README. All numbers reproducible from the pipeline.

## F1. The Zug intervention is measurable, and encoding it is worth 14 points of accuracy

For 2026-2027 the canton of Zug covers 99% of inpatient hospital costs from
cantonal surpluses (~220M CHF), and insurers priced this into premiums.

- Posterior intervention effect on the standard premium: **-12.1%**
  (94% HDI [-13.7%, -10.4%]), vs the officially reported -14.7% on the
  all-models mean premium — consistent, with the gap explained by the
  intervention cutting deeper into managed-care models.
- Backtest, Zug 2026, model trained only on data through 2025 (no policy
  knowledge): forecast 491 CHF, actual 430 CHF — **14.0% error**, the worst
  of all 286 scored forecasts.
- Backtest, Zug 2027, intervention encoded as a dummy identified from 2026
  training data: forecast 445 CHF, actual 444 CHF — **0.3% error**, among
  the best.

Conclusion: a pure data model cannot anticipate a parliamentary decision.
One published policy fact, encoded as a single parameter, moved the affected
forecast from worst to best. For premium forecasting, reading cantonal
politics is part of the method, not a nicety.

## F2. The falsifiable prediction: Zug snaps back in 2028

The intervention is legislated to expire after 2027. Removing the dummy from
the forecast year yields:

> **Zug 2028: 522 CHF/month (90% interval 501-545), +17.7% vs 2027.**

Robustness: this forecast is invariant to the training window — the model
fitted on 2015-2027 (v1.0) and on 2011-2027 (v1.1) give the same median to
within 1 CHF and the same intervention effect to the decimal. Zug's
government additionally expects 2028 to carry cost growth accumulated during
the subsidized years, so this trend-level forecast is plausibly a floor.
To be scored publicly against the BAG publication in September 2027.

## F3. The 90% intervals are close to honest, and accuracy splits by regime

Across 285 out-of-sample forecasts (11 origins, targets 2017-2027,
dummy-informed Zug rows excluded): **median absolute error 1.75%, 90%
coverage 86%**.

The split matters more than the average. Calm years (targets 2017-2019,
2023, 2025-2027): median error 0.8-1.5%, coverage 88-100%. Regime-shift
years (2020, 2022, 2024): median error 3.2-3.5%, coverage 54-69%. The
model's intervals are honest when the world behaves and too narrow when it
turns — which is exactly what the correlated-error structure (F5) predicts.

## F4. The model lags regime shifts by one year — three documented episodes

- Target 2020: over-prediction everywhere (the pandemic-era cost plateau;
  the model extrapolated 2011-2019 growth).
- Targets 2021-2022: continued over-prediction through the plateau and the
  2022 reserve-reduction freeze (cantonal median growth 2022: -0.05%).
- Target 2024: under-prediction everywhere (the spike; 50% coverage 8%).

This is structural for any trend model: turning points are visible only
after one observation of the new regime. A cost-driver layer (health cost
growth leads premium growth) is the v1.2 route to earlier turning-point
detection.

## F5. Cantonal forecast errors are one national bet plus 26 small ones

The dominant forecast uncertainty is the shared national year effect f[t].
When it misses, all cantons miss the same side — which is why the 50%
interval's empirical coverage swings between 4% (2022) and 77% (2023)
instead of hovering near 50%, while the wider 90% interval absorbs most of
the common shock. Users of the forecast should read the 26 cantonal
intervals as strongly correlated, not independent.

## F6. Ticino is the structural outlier — and its 2028 forecast should not be trusted

TI was the worst-forecast canton in 3 of 11 backtest years (2024: 7.0%,
2025: 6.9%, 2027: 7.2% error), always under-predicted: its premium growth
(+9.7% in 2024, the national maximum) persistently outruns the fitted
constant slope. Consequence visible in the 2028 forecast: the model
predicts TI at -2.6% — a premium decline that is trend-reversion artifact,
not prediction, and is flagged as such. A time-varying or breakpoint slope
for TI is the first v1.2 item.

## F7. Market consolidation, visible in one chart

Insurers offering the base tariff fell monotonically from **59 (2015) to
32 (2027)**, and the extended 2011-2014 series shows the same decline from
a higher base — a halving within the observed period, with no single-year
discontinuity. Premium dispersion within cantons has not collapsed
correspondingly (see f01), suggesting consolidation absorbed small carriers
rather than removing price competition among the remaining ones. Not
modeled in v1.1.

## F8. Dataset findings (for anyone reusing priminfo)

- **2027 code-system rename**: age classes (`AKL-ERW` -> `AKA_03_ERW`),
  franchise levels (`FRAST1` -> `FRASTU_01`), accident flags
  (`MIT-UNF` -> `MIT_UNF`) and tariff types (`TAR-BASE` -> `BASE`) all
  changed. Base-level mappings were proven by count identities
  (`FRASTU_01` count == `isBaseF=1` count; `BASE` count == `isBaseP=1`
  count). The non-base tariff taxonomy (`FLEX`/`PRAXIS`/`TEL_DIG` vs
  `TAR-DIV`/`TAR-HAM`/`TAR-HMO`) does not map 1:1 and was not forced.
- **2024 has two validity periods** (01.01-31.08 and 01.09-31.12) — a
  mid-year premium adjustment. Mixing them silently corrupts growth rates
  for both 2024 and 2025; this project uses the January 1 snapshot
  throughout.
- **Format drift**: the 2015 file is comma-separated, later files
  semicolon, the coded era semicolon again; encodings vary between cp1252
  and UTF-8, and an ASCII header proves nothing — the 2011 file's first
  cp1252 umlaut appears 27 KB into the body. Sniff on a large sample and
  keep a fallthrough.
- **`C_ID`/`Kanton` special codes**: besides the 26 cantons, `CH` appears
  in all years and `ZE`/`ZR` from 2015 (~120-170 rows/year); the coded era
  also carries ~40-53 `C_ID=CH` service rows per year with no premium and
  no year. All excluded from cantonal modeling via an explicit scope flag.

## F9. The 2011-2014 coded era decodes cleanly — with one semantic trap

Each coded archive ships its own dictionary (`Doku_PraemienDaten.txt`),
which fully specifies the 22-column schema: `M_ID` {0,19,26} is the age
lower bound, `VAR_ID` {5,6} the accident flag, `V_TYP` the tariff type,
`F`/`F_STUFE` the franchise, `V2_ID` the age subgroup (same K1/J1/E1 codes
as the modern era), and insurer numbers are the modern BAG numbers,
zero-padded.

- **End-to-end proof**: with the decoding applied, the per-canton standard
  premium growth across the 2014/2015 schema break comes out at median
  **4.46%** (range 2.7-6.6%) — a plausible premium round, implemented as a
  permanent validation gate in s02. A deliberately broken decoding (mixing
  accident variants) produced an implausible 11.9% — the gate works.
- **The trap**: coded `isBASE_P` means "Grundprämie (OKP mit Unfall)" —
  base tariff AND with-accident — which is narrower than the modern
  `isBaseP` (base tariff only). The harmonizer derives `is_base_premium`
  by one rule for both eras and keeps the raw flag for cross-checks; 100%
  of derived standard-premium rows carry the raw flag.
- Payoff: the backtest grew from 181 to 285 scored forecasts, and the
  long-run CAGR check came out identical across eras (3.85%/yr over 12 and
  over 16 years) — CH median standard premium **323 CHF (2011) -> 592 CHF
  (2027), +83.1%**.

## F10. More history is not free

Adding 2011-2014 reweighted the long-run slopes toward that era's higher
growth: the 2024 spike forecast improved (median APE 3.63% -> 3.16%) while
the calm 2027 target worsened (0.60% -> 1.06%). Net effect positive
(overall median APE 2.11% -> 1.75% on the overlapping comparison basis),
but the trade-off is real and documented: a longer window buys robustness
at turning points and pays with a slightly stiffer trend in calm years.

## Limitations register

1. Unweighted median across insurers and regions (enrollment weights only
   published for the current year).
2. Standard premium != mittlere Prämie; levels differ by construction,
   growth rates track closely (2026: our +3.3% median vs BAG +4.4% mean;
   2024: +6.9% vs +8.7%).
3. No exogenous cost drivers; premiums are forecast from premiums.
4. One-year horizon only; multi-year forecasts would compound the
   random-walk variance and have not been validated.
5. Constant canton slopes; see F6 for the Ticino consequence.
