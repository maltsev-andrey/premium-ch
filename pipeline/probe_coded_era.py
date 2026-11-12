#!/usr/bin/env python3
"""
probe_coded_era.py — premium-ch v1.1, one-off evidence gathering for
decoding the 2011-2014 coded schema (G_ID/C_ID/M_ID/VAR_ID/...).

Prints, for pasting back:
  A. directory listings of archive/2011..2014 (is there a per-year
     Erläuterungen dictionary?)
  B. all sheets of every Erläuterungen workbook found in the coded-era
     dirs and in current/ (the authoritative code dictionary)
  C. value sets + premium statistics per coded column in the 2014 table
     (codes identify themselves: children's premiums are far lower,
     franchise amounts are literal, base flags are 0/1)
  D. the insurer-number bridge across the 2014/2015 schema break:
     G_ID (2014) vs Versicherer (2015) overlap and per-insurer premium
     continuity for the standard-premium-like slice

No files are written; output is the deliverable.

Usage (srv-anaconda-01, conda env premium-ch):
    python pipeline/probe_coded_era.py
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/nfs/shared/premium-ch")
RAW_DIR = BASE_DIR / "data" / "raw"
ARCHIVE_DIR = RAW_DIR / "archive"
CURRENT_DIR = RAW_DIR / "current"

CODED_YEARS = ["2011", "2012", "2013", "2014"]
CODED_TABLE = ARCHIVE_DIR / "2014" / "Praemien_CH_.csv"   # 22-col, cp1252, ';'
MODERN_TABLE = ARCHIVE_DIR / "2015" / "Pramien_CH.csv"    # 17-col, utf-8, ','

ERLAUT_RE = re.compile(r"erl(ae|a|ä)ut", re.IGNORECASE)
MAX_VALUES = 35          # cap per value-set listing
MAX_SHEET_ROWS = 80      # cap per dictionary sheet dump

CODED_COLUMNS = [
    "C_ID", "C_GRP", "R_ID", "M_ID", "VAR_ID", "V_ID", "V_TYP",
    "V2_ID", "V2_TYP", "F", "F_STUFE",
    "isBASE_P", "isBASE_F", "isBASE_V", "isBASE_V2", "IstTaetig",
]


def hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def read_coded(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="cp1252", sep=";", dtype=str,
                       low_memory=False)


def read_modern(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, dtype=str,
                                 low_memory=False)
                if df.shape[1] > 2:
                    return df
            except UnicodeDecodeError:
                break
    raise ValueError(f"cannot read {path}")


# ---------------------------------------------------------------------------
# A. directory listings
# ---------------------------------------------------------------------------
hr("A. Coded-era archive contents")
for year in CODED_YEARS:
    d = ARCHIVE_DIR / year
    print(f"\narchive/{year}/")
    for p in sorted(d.iterdir()):
        print(f"  {p.stat().st_size:>12,}  {p.name}")

# ---------------------------------------------------------------------------
# B. Erläuterungen dictionaries
# ---------------------------------------------------------------------------
hr("B. Erläuterungen workbooks (code dictionaries)")
dirs = [ARCHIVE_DIR / y for y in CODED_YEARS] + [ARCHIVE_DIR / "2015", CURRENT_DIR]
for d in dirs:
    if not d.exists():
        continue
    for p in sorted(d.iterdir()):
        if not ERLAUT_RE.search(p.name) or p.suffix.lower() not in (".xlsx", ".xls"):
            continue
        print(f"\n--- {p.relative_to(RAW_DIR)} ---")
        try:
            sheets = pd.read_excel(p, sheet_name=None, dtype=str, header=None)
        except Exception as e:
            print(f"  READ FAILED: {e}")
            continue
        for name, df in sheets.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            print(f"\n  [sheet: {name}] {df.shape[0]} rows x {df.shape[1]} cols")
            with pd.option_context("display.max_rows", MAX_SHEET_ROWS,
                                   "display.max_columns", 20,
                                   "display.width", 200):
                print(df.head(MAX_SHEET_ROWS).to_string(index=False, header=False))

# ---------------------------------------------------------------------------
# C. Coded-column value sets in 2014
# ---------------------------------------------------------------------------
hr(f"C. Value sets in {CODED_TABLE.name} (2014)")
coded = read_coded(CODED_TABLE)
coded["P_num"] = pd.to_numeric(coded["P"].str.replace(",", "."), errors="coerce")
print(f"rows: {len(coded)}, columns: {list(coded.columns)}")
for col in CODED_COLUMNS:
    if col not in coded.columns:
        print(f"\n{col}: MISSING")
        continue
    stats = (coded.groupby(col)["P_num"]
             .agg(n="size", p_min="min", p_median="median", p_max="max")
             .sort_values("n", ascending=False))
    print(f"\n{col}: {len(stats)} distinct values")
    print(stats.head(MAX_VALUES).to_string(float_format=lambda x: f"{x:8.1f}"))

# ---------------------------------------------------------------------------
# D. Insurer bridge 2014 (G_ID) vs 2015 (Versicherer)
# ---------------------------------------------------------------------------
hr("D. Insurer-number bridge across the 2014/2015 break")
modern = read_modern(MODERN_TABLE)
g2014 = set(coded["G_ID"].dropna().str.strip())
v2015 = set(modern["Versicherer"].dropna().str.strip())
print(f"distinct G_ID 2014: {len(g2014)}, distinct Versicherer 2015: {len(v2015)}")
print(f"intersection: {len(g2014 & v2015)}")
print(f"G_ID only (left market?): {sorted(g2014 - v2015)}")
print(f"Versicherer only (entered?): {sorted(v2015 - g2014)}")

# premium continuity for the shared insurers, base-like slice:
# 2015 standard premium vs 2014 rows with all isBASE flags = 1
m = modern[(modern["isBaseP"] == "1") & (modern["isBaseF"] == "1")
           & (modern["Altersklasse"] == "AKL-ERW")
           & (modern["Unfalleinschluss"] == "MIT-UNF")]
m_med = (pd.to_numeric(m["Prämie"].str.replace(",", "."), errors="coerce")
         .groupby([m["Versicherer"].str.strip(), m["Kanton"].str.strip()]).median())
base_mask = pd.Series(True, index=coded.index)
for flag in ("isBASE_P", "isBASE_F", "isBASE_V", "isBASE_V2"):
    if flag in coded.columns:
        base_mask &= coded[flag].str.strip() == "1"
c = coded[base_mask]
c_med = c.groupby([c["G_ID"].str.strip(), c["C_ID"].str.strip()])["P_num"].median()
joined = pd.concat([c_med.rename("p2014"), m_med.rename("p2015")], axis=1).dropna()
if len(joined):
    growth = (joined["p2015"] / joined["p2014"] - 1) * 100
    print(f"\nall-isBASE 2014 slice: {len(c)} rows; joined (insurer,canton) cells: {len(joined)}")
    print("implied 2014->2015 growth on joined cells, % "
          f"(plausible band is roughly 0..10):")
    print(growth.describe().to_string(float_format=lambda x: f"{x:8.2f}"))
    print("\nsample of 15 joined cells:")
    print(joined.join(growth.rename("growth_pct")).head(15)
          .to_string(float_format=lambda x: f"{x:8.2f}"))
else:
    print("\nno joined cells — the all-isBASE slice or the join key is wrong; "
          "see section C to pick better anchors")
