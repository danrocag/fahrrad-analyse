"""Download the hourly bike counter data and prepare it for analysis.

Combines the non-exploratory steps of fetch_data.ipynb and prepare.ipynb:

1. Download the monthly eco-counter files from mobidata-bw.de (cached in data/raw).
2. Keep only the counter sites that cover the whole period.
3. Drop the columns that carry no information.
4. Split the site columns into their own table, one row per counter_site_id.

A site keeps its counter_site_id when its counter is replaced (new counter_serial),
renamed (König-Karls-Brücke Barometer -> König-Karls-Brücke) or re-surveyed (Böblinger
Straße 9 m, Samaraweg 93 m). Those changes are not new stations, so the site table
holds the latest name and coordinates, and counter_serial stays in the hourly counts
so device changes remain visible.

Writes data/location_data.csv and data/counters.csv.

Run with:  uv run python -m fahrrad_analyse.build_dataset
Set FAHRRAD_CONTACT to put a contact address in the User-Agent of downloads.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://mobidata-bw.de/daten/eco-counter/v2/fahrradzaehler_stundenwerten_{}.csv.gz"
DATA = Path(__file__).resolve().parents[2] / "data"
CACHE = DATA / "raw"

FIRST_MONTH = "2021-01"
LAST_MONTH = "2026-10"
# A counter site is kept only if its data spans this whole window.
COVERAGE_START = "2021-01-01"
COVERAGE_END = "2026-10-30"

# These columns hold no information (a single value throughout).
EMPTY_COLS = ["timezone", "operator_name", "site_snow_accumulation"]
# Attributes of a counter site, moved from the hourly counts to the site table.
SITE_COLS = ["domain_id", "domain_name", "counter_site", "latitude", "longitude"]

session = requests.Session()
_contact = os.environ.get("FAHRRAD_CONTACT")
session.headers["User-Agent"] = "fahrrad-analyse (private research project" + (
    f"; {_contact})" if _contact else ")")


def fetch_month(ym: str) -> Path | None:
    """Download one monthly file unless it is already cached. Returns the path or None."""
    f = CACHE / f"{ym}.csv.gz"
    if f.exists():
        return f                      # cached: no request at all
    for attempt in range(3):
        try:
            r = session.get(BASE.format(ym), timeout=(10, 60))  # (connect, read) seconds
        except requests.RequestException as e:
            print(f"{ym}: {e}")
        else:
            if r.status_code == 404:
                print(f"{ym}: not available")
                return None
            if r.ok:
                f.write_bytes(r.content)
                return f
            print(f"{ym}: HTTP {r.status_code}")
        time.sleep(10 * 2**attempt)   # back off 10 s, 20 s, 40 s on errors
    return None


def fetch_all() -> pd.DataFrame:
    """Return the hourly data of all months, downloading what is not cached yet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M"):
        ym = m.strftime("%Y%m")
        cached = (CACHE / f"{ym}.csv.gz").exists()
        f = fetch_month(ym)
        if f is not None:
            frames.append(pd.read_csv(f, low_memory=False, na_values=["na"]))
        if not cached:
            time.sleep(2)             # pause between real downloads
    return pd.concat(frames, ignore_index=True)


def filter_full_coverage(hourly: pd.DataFrame) -> pd.DataFrame:
    """Keep only the counter sites whose data spans the whole coverage window."""
    ts = pd.to_datetime(hourly["iso_timestamp"], utc=True)
    coverage = hourly.assign(ts=ts).groupby("counter_site_id")["ts"].agg({
        "ts": ["min", "max"],
    })
    zero_days = hourly.groupby()
    full = coverage[(coverage["min"] < COVERAGE_START) & (coverage["max"] > COVERAGE_END)]
    return hourly[hourly["counter_site_id"].isin(full.index)]


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the counts and split off the sites. Returns (counts, sites)."""
    df = df.assign(datetime=pd.to_datetime(df["iso_timestamp"], utc=True))
    df = df.drop(columns=["iso_timestamp", *EMPTY_COLS])
    df = df.sort_values(["counter_site_id", "datetime"], ignore_index=True)

    if (df.groupby("counter_site_id")["domain_id"].nunique() > 1).any():
        raise ValueError("a counter_site_id appears in more than one domain")

    # Latest name and coordinates of each site
    sites = df.groupby("counter_site_id")[SITE_COLS].last()
    counts = df.drop(columns=SITE_COLS)
    return counts, sites


def main() -> None:
    hourly = fetch_all()
    counts, sites = prepare(filter_full_coverage(hourly))
    sites.to_csv(DATA / "location_data.csv")
    counts.to_csv(DATA / "counters.csv")
    print(f"{len(sites)} sites, {len(counts)} hourly rows written to {DATA}")


if __name__ == "__main__":
    main()
