#!/usr/bin/env python3
"""
s01_audit.py — premium-ch data layer, stage 1: coverage and schema audit (v2).

v2 changes after the first run against real data:
  - filename regex now also matches the ASCII-folded 'Pramien_CH.*' names
    that s00's sanitizer produces (this hid 2015-2026 and current/ in v1)
  - Excel workbooks are audited sheet-by-sheet (2014/2015 are one-sheet-
    per-canton workbooks, not flat tables); the largest sheet is profiled
    and all sheet names are reported
  - coded-schema candidates added: M_ID (age class), VAR_ID/V_TYP (model),
    F/F_STUFE (franchise) — semantics to be confirmed against
    'Erlauterungen zu den Pramiendaten.xlsx' in s02
  - canton check expects the 26 cantons plus the 'CH' pseudo-code

Outputs:
    data/interim/audit/schema_by_year.csv    one row per (year, column)
    data/interim/audit/coverage_by_year.csv  one row per year
    data/interim/audit/audit_report.md       human-readable report

Usage (srv-anaconda-01, conda env premium-ch):
    python pipeline/s01_audit.py
"""

import csv
import re
import sys
import warnings
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
AUDIT_DIR = BASE_DIR / "data" / "interim" / "audit"

# Matches Praemien_CH / Prämien_CH / Pramien_CH (s00 sanitizer folds umlauts).
PREMIUM_FILE_RE = re.compile(
    r"pr(?:ae|a|ä)m\w*_?CH(?![A-Za-z]).*\.(csv|xlsx|xls)$", re.IGNORECASE
)

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SEPARATORS = [";", ",", "\t"]

CANTONS_26 = {
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU",
    "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS",
    "ZG", "ZH",
}

# Candidate raw-column names per canonical field. Audit-only; the real
# mapping (and code semantics) is fixed in s02 against the BAG data
# dictionary 'Erlauterungen zu den Pramiendaten.xlsx'.
COLUMN_CANDIDATES = {
    "insurer": ["Versicherer", "Insurer", "G_ID", "V_NR", "Versicherer-Nr"],
    "canton": ["Kanton", "Canton", "C_ID", "K_ID"],
    "region": ["Region", "R_ID", "Prämienregion", "Praemienregion"],
    "age_group": ["Altersklasse", "AK", "Alterklasse", "A_ID", "Altersgruppe", "M_ID"],
    "franchise": ["Franchise", "F_ID", "Franchisestufe", "F", "F_STUFE"],
    "model": ["Tarif", "Tariftyp", "Modell", "T_ID", "Tarifbezeichnung",
              "VM_ID", "VAR_ID", "V_TYP"],
    "premium": ["Prämie", "Praemie", "P", "Monatsprämie", "Praemie_CHF", "PRAEMIE"],
    "year": ["Jahr", "Year", "J", "Geschäftsjahr", "Geschaeftsjahr", "JAHR"],
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def excel_engine() -> str | None:
    """Prefer calamine (pandas 3.x recommendation) when installed."""
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return None


EXCEL_ENGINE = excel_engine()

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_premium_files() -> dict[str, Path]:
    """Map premium year -> premium table file. CSV wins over Excel."""
    found: dict[str, Path] = {}

    def pick(year: str, candidates: list[Path]) -> None:
        if not candidates:
            return
        candidates.sort(key=lambda p: (p.suffix.lower() != ".csv", p.name))
        found[year] = candidates[0]

    for year_dir in sorted(ARCHIVE_DIR.glob("[0-9]" * 4)):
        matches = [p for p in year_dir.iterdir() if PREMIUM_FILE_RE.search(p.name)]
        pick(year_dir.name, matches)

    if CURRENT_DIR.exists():
        matches = [p for p in CURRENT_DIR.iterdir() if PREMIUM_FILE_RE.search(p.name)]
        pick("current", matches)

    return found


# ---------------------------------------------------------------------------
# Robust reading
# ---------------------------------------------------------------------------


def sniff_csv(path: Path) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc) as f:
                sample = f.read(64 * 1024)
        except UnicodeDecodeError:
            continue
        header = sample.splitlines()[0] if sample else ""
        counts = {sep: header.count(sep) for sep in SEPARATORS}
        sep = max(counts, key=counts.get)
        if counts[sep] >= 2:
            return enc, sep
    raise ValueError(f"could not sniff encoding/separator for {path}")


def read_table(path: Path) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Return (df, encoding, separator, sheet_info). For Excel workbooks,
    df is the largest sheet; sheet_info lists 'name:rows' for all sheets."""
    if path.suffix.lower() == ".csv":
        enc, sep = sniff_csv(path)
        df = pd.read_csv(path, encoding=enc, sep=sep, dtype=str, low_memory=False)
        return df, enc, sep, []

    kwargs = {"dtype": str, "sheet_name": None}
    if EXCEL_ENGINE:
        kwargs["engine"] = EXCEL_ENGINE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sheets: dict[str, pd.DataFrame] = pd.read_excel(path, **kwargs)
    sheet_info = [f"{name}:{len(df)}" for name, df in sheets.items()]
    largest = max(sheets, key=lambda k: len(sheets[k]))
    return sheets[largest], "excel", f"sheet={largest}", sheet_info


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        hit = cols.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None


# ---------------------------------------------------------------------------
# Per-year audit
# ---------------------------------------------------------------------------


def audit_year(year: str, path: Path) -> tuple[dict, list[str]]:
    df, enc, sep, sheet_info = read_table(path)
    columns = [str(c).strip() for c in df.columns]

    row: dict = {
        "year": year,
        "file": str(path.relative_to(RAW_DIR)),
        "encoding": enc,
        "separator": sep,
        "n_rows": len(df),
        "n_columns": len(columns),
        "n_sheets": len(sheet_info) or "",
        "sheets": " | ".join(sheet_info[:40]),
    }

    for canon, candidates in COLUMN_CANDIDATES.items():
        col = first_present(df, candidates)
        row[f"col_{canon}"] = col or ""
        if col is None:
            continue
        series = df[col].dropna().astype(str).str.strip()
        if canon in ("insurer", "canton", "region", "age_group", "franchise", "model"):
            row[f"n_{canon}"] = series.nunique()
        if canon == "canton":
            row["cantons"] = " ".join(sorted(series.unique())[:30])
        if canon == "age_group":
            row["age_groups"] = " ".join(sorted(series.unique())[:15])
        if canon == "franchise":
            row["franchises"] = " ".join(sorted(series.unique())[:20])
        if canon == "year":
            row["data_years"] = " ".join(sorted(series.unique())[:10])
        if canon == "premium":
            numeric = pd.to_numeric(series.str.replace(",", "."), errors="coerce")
            row["premium_min"] = round(float(numeric.min()), 2)
            row["premium_max"] = round(float(numeric.max()), 2)
            row["premium_n_nan"] = int(numeric.isna().sum())

    return row, columns


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(coverage: list[dict], schemas: dict[str, list[str]]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    cov_df = pd.DataFrame(coverage)
    cov_df.to_csv(AUDIT_DIR / "coverage_by_year.csv", index=False)

    with open(AUDIT_DIR / "schema_by_year.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "position", "column"])
        for year, cols in sorted(schemas.items()):
            for i, c in enumerate(cols):
                w.writerow([year, i, c])

    newest = max(schemas)
    ref = set(schemas[newest])
    lines = [
        "# premium-ch — raw data audit (s01 v2)",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Reference schema year: **{newest}** ({len(ref)} columns)",
        f"Excel engine: {EXCEL_ENGINE or 'pandas default (xlrd/openpyxl)'}",
        "",
        "## Coverage by year",
        "",
        cov_df.to_markdown(index=False),
        "",
        "## Schema drift vs reference year",
        "",
    ]
    for year in sorted(schemas):
        cols = set(schemas[year])
        missing = sorted(ref - cols)
        extra = sorted(cols - ref)
        if not missing and not extra:
            lines.append(f"- **{year}**: identical to reference")
        else:
            lines.append(f"- **{year}**:")
            if missing:
                lines.append(f"  - missing vs ref: {', '.join(missing)}")
            if extra:
                lines.append(f"  - extra vs ref: {', '.join(extra)}")
    lines.append("")

    lines.append("## Red flags")
    lines.append("")
    flags = []
    for row in coverage:
        y = row["year"]
        cantons = set((row.get("cantons") or "").split())
        if cantons:
            unexpected = cantons - CANTONS_26 - {"CH"}
            missing_c = CANTONS_26 - cantons
            if unexpected:
                flags.append(f"- {y}: unexpected canton codes: {', '.join(sorted(unexpected))}")
            if missing_c:
                flags.append(f"- {y}: missing cantons: {', '.join(sorted(missing_c))}")
        if not row.get("col_premium"):
            flags.append(f"- {y}: no premium column recognized — extend COLUMN_CANDIDATES")
        if row.get("premium_n_nan", 0):
            flags.append(f"- {y}: {row['premium_n_nan']} non-numeric premium values")
        if row.get("n_sheets") and row.get("n_sheets") != "" and int(row["n_sheets"]) > 1:
            flags.append(
                f"- {y}: Excel workbook with {row['n_sheets']} sheets "
                f"(likely one per canton) — s02 must read ALL sheets, "
                f"audit profiled only the largest"
            )
    lines.extend(flags if flags else ["- none"])
    lines.append("")

    (AUDIT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"report written: {AUDIT_DIR / 'audit_report.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    files = find_premium_files()
    if not files:
        log(f"no premium files found under {RAW_DIR} — run s00_download.py first")
        return 1

    log(f"found premium tables for {len(files)} years: {', '.join(sorted(files))}")

    coverage: list[dict] = []
    schemas: dict[str, list[str]] = {}
    for year in sorted(files):
        path = files[year]
        log(f"auditing {year}: {path.name}")
        try:
            row, columns = audit_year(year, path)
        except Exception as e:
            log(f"  FAILED: {e}")
            coverage.append({"year": year, "file": str(path), "error": str(e)})
            continue
        coverage.append(row)
        schemas[year] = columns
        log(
            f"  {row['n_rows']} rows, {row['n_columns']} cols, "
            f"insurers={row.get('n_insurer', '?')}, cantons={row.get('n_canton', '?')}"
            + (f", sheets={row['n_sheets']}" if row.get("n_sheets") else "")
        )

    if schemas:
        write_report(coverage, schemas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
