#!/usr/bin/env python3
"""
s02_harmonize.py — premium-ch data layer, stage 2: harmonization.

Builds one tidy parquet from the 2015+ priminfo premium tables
(the self-describing German-column era, 17 columns). Facts from s01:

  - 2015/2016 name the tariff-type column 'Tarif-Typ' (later 'Tariftyp')
    and lack 'Hoheitsgebiet' -> renamed / filled with NA
  - 2024 ships two validity periods; we take the valid-from-01.01
    snapshot (consistent "premiums as set at year start" definition
    for every year) and ignore the 01.09 mid-year adjustment file
  - Kanton contains the 26 cantons plus special codes (CH, ZE, ZR);
    special rows are kept but tagged scope='special' so the cantonal
    model can filter on scope='canton'
  - the authoritative code semantics live in
    'Erlauterungen zu den Pramiendaten.xlsx' (in every archive)

The 2011-2014 coded era (G_ID/C_ID/M_ID/VAR_ID/...) is NOT harmonized
here — deferred until the codes are mapped against the data dictionary.

Outputs:
    data/processed/premiums.parquet          tidy table, all years
    data/interim/harmonize/values_report.md  distinct values per categorical
                                             per year (evidence for choosing
                                             the v1.0 'standard premium' filter)

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

FIRST_MODERN_YEAR = 2015  # coded era (2011-2014) deferred

PREMIUM_FILE_RE = re.compile(
    r"pr(?:ae|a|ä)m\w*_?CH(?![A-Za-z]).*\.csv$", re.IGNORECASE
)
# validity-suffixed files not starting on Jan 1 are mid-year adjustments
MIDYEAR_RE = re.compile(r"gultig_von_(?!01\.01\.)", re.IGNORECASE)

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

CANTONS_26 = {
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU",
    "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS",
    "ZG", "ZH",
}

# raw German column -> harmonized name (after the Tarif-Typ rename)
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
DROP_RAW = {"Sort"}  # 2015/2016 ordering helper, no information

# 2027 code-system rename, mappings PROVEN by count identities in the
# s02 values report (FRASTU_01 == is_base_franchise=1, BASE == is_base_premium=1):
CODE_NORMALIZATION = {
    "age_class": {
        "AKA_01_KIN": "AKL-KIN",
        "AKA_02_JUG": "AKL-JUG",
        "AKA_03_ERW": "AKL-ERW",
    },
    "accident": {"MIT_UNF": "MIT-UNF", "OHN_UNF": "OHN-UNF"},
    "franchise_level": {f"FRASTU_0{i}": f"FRAST{i}" for i in range(1, 8)},
    # only the base tariff maps cleanly across the 2027 boundary;
    # FLEX/PRAXIS/TEL_DIG keep their new labels (counts do not reconcile
    # with TAR-DIV/TAR-HAM/TAR-HMO)
    "tariff_type": {"BASE": "TAR-BASE"},
}

CATEGORICALS = [
    "age_class", "age_subgroup", "franchise", "franchise_chf",
    "franchise_level", "tariff_type", "accident",
    "is_base_franchise", "is_base_premium",
]

KEY_COLS = [
    "year", "insurer", "canton", "region", "age_class", "age_subgroup",
    "franchise_level", "tariff", "accident",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------


def select_files() -> list[Path]:
    """One year-start premium CSV per year dir (2015+), plus current/."""
    dirs = [d for d in sorted(ARCHIVE_DIR.glob("[0-9]" * 4))
            if int(d.name) >= FIRST_MODERN_YEAR]
    if CURRENT_DIR.exists():
        dirs.append(CURRENT_DIR)

    picked = []
    for d in dirs:
        candidates = [
            p for p in d.iterdir()
            if PREMIUM_FILE_RE.search(p.name) and not MIDYEAR_RE.search(p.name)
        ]
        if not candidates:
            log(f"WARNING: no year-start premium CSV in {d}")
            continue
        candidates.sort(key=lambda p: len(p.name))  # unsuffixed name first
        picked.append(candidates[0])
        if len(candidates) > 1:
            skipped = ", ".join(p.name for p in candidates[1:])
            log(f"{d.name}: picked {candidates[0].name} (skipped: {skipped})")
    return picked


# ---------------------------------------------------------------------------
# Reading and harmonizing one file
# ---------------------------------------------------------------------------


SEPARATORS = [";", ",", "\t"]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Sniff encoding and separator (same approach as s01), then read."""
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc) as f:
                header = f.readline()
        except UnicodeDecodeError:
            continue
        counts = {sep: header.count(sep) for sep in SEPARATORS}
        sep = max(counts, key=counts.get)
        if counts[sep] < 2:
            continue
        return pd.read_csv(path, encoding=enc, sep=sep, dtype=str,
                           low_memory=False)
    raise ValueError(f"no working encoding/separator for {path}")


def harmonize_file(path: Path) -> pd.DataFrame:
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
            df[col] = pd.NA  # e.g. territory in 2015/2016

    # dtypes
    df["premium"] = pd.to_numeric(
        df["premium"].str.replace(",", ".", regex=False), errors="coerce"
    )
    for col in ("year", "survey_year"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ("canton", "region", "age_class", "age_subgroup", "tariff",
                "tariff_type", "accident", "insurer", "franchise",
                "franchise_level"):
        df[col] = df[col].astype(str).str.strip()

    # franchise is a label (not numeric) — keep it raw, extract CHF amount
    df["franchise_chf"] = pd.to_numeric(
        df["franchise"].str.extract(r"(\d+)", expand=False), errors="coerce"
    ).astype("Int64")

    # normalize the 2027 code-system rename onto the 2015-2026 codes
    for col, mapping in CODE_NORMALIZATION.items():
        df[col] = df[col].replace(mapping)

    df["scope"] = df["canton"].map(
        lambda c: "canton" if c in CANTONS_26 else "special"
    )
    df["source_file"] = str(path.relative_to(RAW_DIR))
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(df: pd.DataFrame) -> list[str]:
    problems = []

    bad_premium = df["premium"].isna() | (df["premium"] <= 0)
    if bad_premium.any():
        problems.append(f"{int(bad_premium.sum())} rows with missing/non-positive premium")

    bad_year = df["year"].isna()
    if bad_year.any():
        problems.append(f"{int(bad_year.sum())} rows with missing Geschäftsjahr")

    for year, g in df[df["scope"] == "canton"].groupby("year"):
        missing = CANTONS_26 - set(g["canton"].unique())
        if missing:
            problems.append(f"year {year}: missing cantons {sorted(missing)}")

    dupes = df.duplicated(subset=KEY_COLS, keep=False)
    if dupes.any():
        problems.append(
            f"{int(dupes.sum())} rows share a logical key {KEY_COLS} — "
            f"key is incomplete (check franchise_level/is_base_* role)"
        )
    return problems


# ---------------------------------------------------------------------------
# Values report
# ---------------------------------------------------------------------------


def write_values_report(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# premium-ch — harmonized value sets (s02)",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Rows: {len(df)}, years: {sorted(df['year'].dropna().unique().tolist())}",
        "",
        "Purpose: evidence for choosing the v1.0 'standard premium' filter",
        "(adult, CHF 300, base tariff, accident incl/excl) in s03. Verify",
        "meanings against 'Erlauterungen zu den Pramiendaten.xlsx'.",
        "",
    ]
    for col in CATEGORICALS + ["scope"]:
        lines.append(f"## {col}")
        lines.append("")
        pivot = (
            df.groupby([col, "year"], dropna=False).size().unstack(fill_value=0)
        )
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
    for path in files:
        df = harmonize_file(path)
        years = sorted(df["year"].dropna().unique().tolist())
        log(f"  {path.parent.name}/{path.name}: {len(df)} rows, Geschäftsjahr={years}")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    problems = validate(full)
    for p in problems:
        log(f"VALIDATION: {p}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PARQUET, index=False)
    log(
        f"written {OUT_PARQUET} — {len(full)} rows, "
        f"{full['year'].nunique()} years, "
        f"{full[full.scope == 'canton']['canton'].nunique()} cantons"
    )

    write_values_report(full)
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
