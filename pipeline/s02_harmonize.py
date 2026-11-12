#!/usr/bin/env python3
"""
s02_harmonize.py — premium-ch data layer, stage 2: harmonization (v4).

Builds one tidy parquet from BOTH priminfo schema eras:

  - modern era 2015+ (17 self-describing German columns), as in v3
  - coded era 2011-2014 (22-column G_ID/C_ID/... schema), decoded per the
    official per-year dictionary Doku_PraemienDaten.txt found in each
    coded archive, with every mapping listed in CODED_* below

Coded-era mapping (authoritative source: Doku_PraemienDaten.txt):
  G_ID   -> insurer (BAG number; zero-padded, '0008' == modern '8')
  C_ID   -> canton ('CH' pseudo-code exists here too)   C_GRP: const, drop
  JAHR/EJAHR -> year / survey_year
  R_ID 0..3  -> region 'PR-REG CH<r>'
  M_ID {0,19,26} -> age_class KIN/JUG/ERW (age lower bound)
  VAR_ID {5,6}   -> accident MIT-UNF / OHN-UNF
  V_ID   -> tariff        V_KBEZ -> tariff_name
  V_TYP {Base,HAM_RDS,HMO,DIV} -> tariff_type TAR-BASE/HAM/HMO/DIV
  V2_ID  -> age_subgroup (same K1..K3/J1/E1 codes as modern)
  F      -> franchise label + franchise_chf     F_STUFE -> FRAST<n>
  isBASE_F -> is_base_franchise ('Grundfranchise 300 bzw. 0 Franken')
  isBASE_P is 'Grundprämie (OKP mit Unfall)' — NARROWER than modern
    isBaseP (base tariff only). To keep one filter across eras,
    is_base_premium is DERIVED by the modern rule
    (tariff_type == TAR-BASE AND accident == MIT-UNF) for coded years,
    and the raw flag is kept as is_base_premium_raw for cross-checks.
  isBASE_V / isBASE_V2 / V_SORT_NR: presentation helpers, dropped
  IstTaetig {1.0,1.1}: active / restricted operating area -> active_restricted

Validation adds the insurer bridge across the 2014/2015 schema break:
per-canton standard-premium medians must imply a plausible 2014->2015
growth (median within BRIDGE_BAND), proving the decoding end to end.

Outputs:
    data/processed/premiums.parquet          tidy table, 2011-2027
    data/interim/harmonize/values_report.md  distinct values per categorical/year

Usage (srv-anaconda-01, conda env premium-ch):
    python pipeline/s02_harmonize.py
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/nfs/shared/premium-ch")
RAW_DIR = BASE_DIR / "data" / "raw"
ARCHIVE_DIR = RAW_DIR / "archive"
CURRENT_DIR = RAW_DIR / "current"
OUT_PARQUET = BASE_DIR / "data" / "processed" / "premiums.parquet"
REPORT_DIR = BASE_DIR / "data" / "interim" / "harmonize"

FIRST_CODED_YEAR = 2011
FIRST_MODERN_YEAR = 2015
BRIDGE_BAND = (0.0, 8.0)  # plausible median canton growth 2014->2015, %

PREMIUM_FILE_RE = re.compile(
    r"pr(?:ae|a|ä)m\w*_?CH(?![A-Za-z]).*\.csv$", re.IGNORECASE
)
MIDYEAR_RE = re.compile(r"gultig_von_(?!01\.01\.)", re.IGNORECASE)

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SEPARATORS = [";", ",", "\t"]

CANTONS_26 = {
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU",
    "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS",
    "ZG", "ZH",
}

# ---- modern era (2015+) ----------------------------------------------------
COLUMN_MAP = {
    "Geschäftsjahr": "year",
    "Erhebungsjahr": "survey_year",
    "Versicherer": "insurer",
    "Kanton": "canton",
    "Hoheitsgebiet": "territory",
    "Region": "region",
    "Altersklasse": "age_class",
    "Altersuntergruppe": "age_subgroup",
    "Franchise": "franchise",
    "Franchisestufe": "franchise_level",
    "Tarif": "tariff",
    "Tarifbezeichnung": "tariff_name",
    "Tariftyp": "tariff_type",
    "Unfalleinschluss": "accident",
    "isBaseF": "is_base_franchise",
    "isBaseP": "is_base_premium",
    "Prämie": "premium",
}
RENAMES_PRE = {"Tarif-Typ": "Tariftyp"}  # 2015/2016
DROP_RAW = {"Sort"}

# 2027 code-system rename, mappings proven by count identities
CODE_NORMALIZATION = {
    "age_class": {
        "AKA_01_KIN": "AKL-KIN",
        "AKA_02_JUG": "AKL-JUG",
        "AKA_03_ERW": "AKL-ERW",
    },
    "accident": {"MIT_UNF": "MIT-UNF", "OHN_UNF": "OHN-UNF"},
    "franchise_level": {f"FRASTU_0{i}": f"FRAST{i}" for i in range(1, 8)},
    "tariff_type": {"BASE": "TAR-BASE"},
}

# ---- coded era (2011-2014), per Doku_PraemienDaten.txt ---------------------
CODED_AGE_CLASS = {"0": "AKL-KIN", "19": "AKL-JUG", "26": "AKL-ERW"}
CODED_ACCIDENT = {"5": "MIT-UNF", "05": "MIT-UNF", "6": "OHN-UNF", "06": "OHN-UNF"}
CODED_TARIFF_TYPE = {"Base": "TAR-BASE", "HAM_RDS": "TAR-HAM",
                     "HMO": "TAR-HMO", "DIV": "TAR-DIV"}
CODED_DROP = {"C_GRP", "V_SORT_NR", "isBASE_V", "isBASE_V2", "V2_TYP"}

CATEGORICALS = [
    "age_class", "age_subgroup", "franchise", "franchise_chf",
    "franchise_level", "tariff_type", "accident",
    "is_base_franchise", "is_base_premium",
]

KEY_COLS = [
    "year", "insurer", "canton", "region", "age_class", "age_subgroup",
    "franchise_level", "tariff", "accident",
]

OUTPUT_COLS = [
    "year", "survey_year", "insurer", "canton", "territory", "region",
    "age_class", "age_subgroup", "franchise", "franchise_chf",
    "franchise_level", "tariff", "tariff_name", "tariff_type", "accident",
    "is_base_franchise", "is_base_premium", "is_base_premium_raw",
    "active_restricted", "premium", "scope", "era", "source_file",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------


def select_files() -> list[tuple[Path, str]]:
    """One year-start premium CSV per year dir, tagged 'coded'/'modern'."""
    picked: list[tuple[Path, str]] = []
    dirs = [d for d in sorted(ARCHIVE_DIR.glob("[0-9]" * 4))
            if int(d.name) >= FIRST_CODED_YEAR]
    if CURRENT_DIR.exists():
        dirs.append(CURRENT_DIR)

    for d in dirs:
        era = ("coded" if d.name.isdigit() and int(d.name) < FIRST_MODERN_YEAR
               else "modern")
        candidates = [
            p for p in d.iterdir()
            if PREMIUM_FILE_RE.search(p.name) and not MIDYEAR_RE.search(p.name)
        ]
        if not candidates:
            log(f"WARNING: no year-start premium CSV in {d}")
            continue
        candidates.sort(key=lambda p: len(p.name))
        picked.append((candidates[0], era))
        if len(candidates) > 1:
            skipped = ", ".join(p.name for p in candidates[1:])
            log(f"{d.name}: picked {candidates[0].name} (skipped: {skipped})")
    return picked


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Sniff encoding on a large sample (an ASCII header line proves
    nothing — 2011's first cp1252 umlaut sits 27 KB into the body),
    then fall through to the next encoding if the full read still fails."""
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc) as f:
                sample = f.read(1024 * 1024)
        except UnicodeDecodeError:
            continue
        header = sample.splitlines()[0] if sample else ""
        counts = {sep: header.count(sep) for sep in SEPARATORS}
        sep = max(counts, key=counts.get)
        if counts[sep] < 2:
            continue
        try:
            return pd.read_csv(path, encoding=enc, sep=sep, dtype=str,
                               low_memory=False)
        except UnicodeDecodeError:
            continue  # umlaut past the sample — try the next encoding
    raise ValueError(f"no working encoding/separator for {path}")


def finalize(df: pd.DataFrame, path: Path, era: str) -> pd.DataFrame:
    """Shared dtype work, scope tag, column order."""
    df["premium"] = pd.to_numeric(
        df["premium"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    for col in ("year", "survey_year"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["franchise_chf"] = pd.to_numeric(
        df["franchise"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).astype("Int64")
    for col in ("canton", "region", "age_class", "age_subgroup", "tariff",
                "tariff_type", "accident", "insurer", "franchise",
                "franchise_level"):
        df[col] = df[col].astype(str).str.strip()
    df["scope"] = df["canton"].map(
        lambda c: "canton" if c in CANTONS_26 else "special"
    )
    df["era"] = era
    df["source_file"] = str(path.relative_to(RAW_DIR))
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Modern era (2015+)
# ---------------------------------------------------------------------------


def harmonize_modern(path: Path) -> pd.DataFrame:
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENAMES_PRE)
    df = df.drop(columns=[c for c in DROP_RAW if c in df.columns])

    unknown = [c for c in df.columns if c not in COLUMN_MAP]
    if unknown:
        raise ValueError(f"{path.name}: unmapped columns {unknown} — "
                         f"extend COLUMN_MAP deliberately, do not guess")
    df = df.rename(columns=COLUMN_MAP)
    for col in COLUMN_MAP.values():
        if col not in df.columns:
            df[col] = pd.NA  # territory in 2015/2016
    for col, mapping in CODE_NORMALIZATION.items():
        df[col] = df[col].astype(str).str.strip().replace(mapping)
    return finalize(df, path, "modern")


# ---------------------------------------------------------------------------
# Coded era (2011-2014)
# ---------------------------------------------------------------------------


def harmonize_coded(path: Path) -> pd.DataFrame:
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]
    expected = {
        "G_ID", "C_ID", "C_GRP", "EJAHR", "JAHR", "R_ID", "M_ID", "VAR_ID",
        "V_ID", "V_TYP", "isBASE_V2", "V2_ID", "V2_TYP", "isBASE_P",
        "isBASE_F", "F_STUFE", "F", "P", "V_SORT_NR", "V_KBEZ", "isBASE_V",
        "IstTaetig",
    }
    unknown = set(df.columns) - expected
    missing = expected - set(df.columns)
    if unknown or missing:
        raise ValueError(f"{path.name}: coded schema drift — "
                         f"unknown {sorted(unknown)}, missing {sorted(missing)}")

    out = pd.DataFrame()
    out["year"] = df["JAHR"]
    out["survey_year"] = df["EJAHR"]
    out["insurer"] = df["G_ID"].astype(str).str.strip().str.lstrip("0")
    out["canton"] = df["C_ID"]
    out["region"] = "PR-REG CH" + df["R_ID"].astype(str).str.strip()
    out["age_class"] = df["M_ID"].astype(str).str.strip().map(CODED_AGE_CLASS)
    out["age_subgroup"] = df["V2_ID"]
    out["franchise"] = df["F"]
    out["franchise_level"] = "FRAST" + df["F_STUFE"].astype(str).str.strip()
    out["tariff"] = df["V_ID"]
    out["tariff_name"] = df["V_KBEZ"]
    out["tariff_type"] = df["V_TYP"].astype(str).str.strip().map(CODED_TARIFF_TYPE)
    out["accident"] = (df["VAR_ID"].astype(str).str.strip()
                       .map(CODED_ACCIDENT))
    out["is_base_franchise"] = df["isBASE_F"].astype(str).str.strip()
    # modern-rule derivation (see module docstring); raw flag kept
    out["is_base_premium"] = (
        (out["tariff_type"] == "TAR-BASE") & (out["accident"] == "MIT-UNF")
    ).astype(int).astype(str)
    out["is_base_premium_raw"] = df["isBASE_P"].astype(str).str.strip()
    out["active_restricted"] = (df["IstTaetig"].astype(str).str.strip()
                                == "1.1").astype(int).astype(str)
    out["premium"] = df["P"]

    for col, src in (("age_class", "M_ID"), ("accident", "VAR_ID"),
                     ("tariff_type", "V_TYP")):
        # fail on unmapped VALUES only; empty source cells pass through as
        # NA (the C_ID=CH service rows carry no M_ID/VAR_ID/premium)
        src_filled = df[src].notna() & (df[src].astype(str).str.strip() != "")
        bad_mask = out[col].isna() & src_filled
        if bad_mask.any():
            bad = df.loc[bad_mask, src].unique()[:10]
            raise ValueError(f"{path.name}: unmapped {src} values {bad}")
        n_empty = int((~src_filled).sum())
        if n_empty:
            log(f"  note: {path.parent.name}: {n_empty} rows with empty {src} "
                f"-> {col}=NA (service rows)")
    return finalize(out, path, "coded")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def standard_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["scope"] == "canton")
        & (df["is_base_premium"].astype(str) == "1")
        & (df["is_base_franchise"].astype(str) == "1")
        & (df["age_class"] == "AKL-ERW")
        & (df["accident"] == "MIT-UNF")
    ]


def validate(df: pd.DataFrame) -> list[str]:
    problems = []

    cantonal = df[df["scope"] == "canton"]
    bad_premium = cantonal["premium"].isna() | (cantonal["premium"] <= 0)
    if bad_premium.any():
        problems.append(f"{int(bad_premium.sum())} cantonal rows with "
                        f"missing/non-positive premium")
    n_special_bad = int(df[df["scope"] == "special"]["premium"].isna().sum())
    if n_special_bad:
        log(f"note: {n_special_bad} special-scope rows without premium "
            f"(e.g. coded-era C_ID=CH) — kept, not modeled")
    if cantonal["year"].isna().any():
        problems.append(f"{int(cantonal['year'].isna().sum())} cantonal rows "
                        f"with missing year")

    for year, g in df[df["scope"] == "canton"].groupby("year"):
        miss = CANTONS_26 - set(g["canton"].unique())
        if miss:
            problems.append(f"year {year}: missing cantons {sorted(miss)}")

    dupes = cantonal.duplicated(subset=KEY_COLS, keep=False)
    if dupes.any():
        problems.append(f"{int(dupes.sum())} cantonal rows share the "
                        f"logical key {KEY_COLS}")

    # insurer bridge across the 2014/2015 schema break
    std = standard_slice(df)
    med = std.groupby(["year", "canton"])["premium"].median().unstack(0)
    if 2014 in med.columns and 2015 in med.columns:
        growth = (med[2015] / med[2014] - 1) * 100
        g_med = float(growth.median())
        log(f"bridge check 2014->2015 (standard premium, canton medians): "
            f"median {g_med:.2f}%, min {growth.min():.2f}%, max {growth.max():.2f}%")
        if not (BRIDGE_BAND[0] <= g_med <= BRIDGE_BAND[1]):
            problems.append(
                f"bridge check FAILED: median canton growth 2014->2015 = "
                f"{g_med:.2f}%, outside {BRIDGE_BAND} — coded-era decoding suspect"
            )
    else:
        problems.append("bridge check skipped: 2014 or 2015 standard slice empty")

    # coded-era: derived is_base_premium vs raw Grundprämie flag
    coded_std = std[std["era"] == "coded"]
    if len(coded_std):
        raw_in_std = (coded_std["is_base_premium_raw"] == "1").mean()
        log(f"coded era: {raw_in_std:.0%} of derived standard-premium rows "
            f"also carry the raw Grundprämie flag (expect ~100%)")
        if raw_in_std < 0.95:
            problems.append(
                f"only {raw_in_std:.0%} of coded standard rows carry isBASE_P=1 "
                f"— derivation vs raw flag mismatch, inspect before trusting"
            )
    return problems


# ---------------------------------------------------------------------------
# Values report
# ---------------------------------------------------------------------------


def write_values_report(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# premium-ch — harmonized value sets (s02 v4, both eras)",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Rows: {len(df)}, years: {sorted(df['year'].dropna().unique().tolist())}",
        "",
    ]
    for col in CATEGORICALS + ["scope", "era"]:
        lines.append(f"## {col}")
        lines.append("")
        pivot = df.groupby([col, "year"], dropna=False).size().unstack(fill_value=0)
        lines.append(pivot.to_markdown())
        lines.append("")
    lines.append("## rows per year / scope")
    lines.append("")
    lines.append(df.groupby(["year", "scope"]).size().unstack(fill_value=0).to_markdown())
    lines.append("")
    (REPORT_DIR / "values_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"values report: {REPORT_DIR / 'values_report.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    files = select_files()
    if not files:
        log("no input files — run s00/s01 first")
        return 1
    log(f"harmonizing {len(files)} files")

    frames = []
    for path, era in files:
        df = harmonize_coded(path) if era == "coded" else harmonize_modern(path)
        years = sorted(df["year"].dropna().unique().tolist())
        log(f"  [{era}] {path.parent.name}/{path.name}: {len(df)} rows, year={years}")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    problems = validate(full)
    for p in problems:
        log(f"VALIDATION: {p}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PARQUET, index=False)
    log(
        f"written {OUT_PARQUET} — {len(full)} rows, "
        f"{full['year'].nunique()} years "
        f"({int(full['year'].min())}-{int(full['year'].max())})"
    )

    write_values_report(full)
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
