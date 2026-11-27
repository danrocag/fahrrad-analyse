"""Download the hourly bike counter data and prepare it for analysis.

Combines the non-exploratory steps of fetch_data.ipynb and prepare.ipynb:

1. Download the monthly eco-counter files from mobidata-bw.de (cached in data/raw).
2. Keep only the counter sites with data on at least 98% of the days in the period
   and no extreme outliers among their daily peaks.
3. Drop the columns that carry no information.
4. Split the site columns into their own table, one row per counter_site_id.
5. Aggregate the hourly counts to daily minima, maxima and means per site.

A site keeps its counter_site_id when its counter is replaced (new counter_serial),
renamed (König-Karls-Brücke Barometer -> König-Karls-Brücke) or re-surveyed (Böblinger
Straße 9 m, Samaraweg 93 m). Those changes are not new stations, so the site table
holds the latest name and coordinates, and counter_serial stays in the hourly counts
so device changes remain visible.

Writes data/location_data.csv, data/counters_hourly.csv and data/counters_daily.csv.

Run with:  uv run python -m fahrrad_analyse.build_dataset
Set FAHRRAD_CONTACT to put a contact address in the User-Agent of downloads.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.covariance import EllipticEnvelope

BASE = "https://mobidata-bw.de/daten/eco-counter/v2/fahrradzaehler_stundenwerten_{}.csv.gz"
DATA = Path(__file__).resolve().parents[2] / "data"
CACHE = DATA / "raw"

FIRST_MONTH = "2021-01"
LAST_MONTH = "2026-10"
# Days are counted in local time over this window (both ends included).
COVERAGE_START = "2021-01-01"
COVERAGE_END = "2026-10-30"
TIMEZONE = "Europe/Berlin"
# A counter site is kept only if it has data on this share of the window's days ...
MIN_DAY_SHARE = 0.95
# ... and no daily peak beyond this shifted Mahalanobis distance (see outlier_scores).
MAX_MAHALANOBIS = 1000

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


def daily_peaks(hourly: pd.DataFrame) -> pd.DataFrame:
    """Highest hourly channels_all of each site and local day inside the coverage window.

    Returns one row per (counter_site_id, date); days without any rows are absent.
    """
    local = pd.to_datetime(hourly["iso_timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    date = local.dt.tz_localize(None).dt.normalize()
    peaks = (
        hourly.assign(date=date)
        .groupby(["counter_site_id", "date"], as_index=False)["channels_all"]
        .max()
    )
    in_window = peaks["date"].between(COVERAGE_START, COVERAGE_END)
    return peaks[in_window]


def outlier_scores(peaks: pd.Series) -> pd.Series:
    """Shifted Mahalanobis distance of each of a site's daily peaks under a robust Gaussian fit.

    EllipticEnvelope.decision_function is the negated squared Mahalanobis distance
    shifted by offset_, so that points beyond the envelope are negative. Its negation
    is what MAX_MAHALANOBIS is compared against.
    """
    X = peaks.to_numpy(dtype=float).reshape(-1, 1)
    envelope = EllipticEnvelope(random_state=0).fit(X)
    return pd.Series(-envelope.decision_function(X), index=peaks.index)


def select_sites(hourly: pd.DataFrame) -> pd.DataFrame:
    """Keep only the counter sites with near-complete, outlier-free data.

    A site is kept if
    - at least MIN_DAY_SHARE of the days in the coverage window are present, where a
      day is missing if it has no rows or channels_all is zero all day, and
    - none of its daily peaks lies further than MAX_MAHALANOBIS from the rest,
      according to sklearn's EllipticEnvelope.

    Prints every dropped site with the reason.
    """
    names = hourly.groupby("counter_site_id")["counter_site"].last()
    n_days = len(pd.date_range(COVERAGE_START, COVERAGE_END, freq="D"))
    peaks = daily_peaks(hourly)
    present = peaks[peaks["channels_all"] > 0]

    # Sites without any day in the window are missing from the groupby, hence the reindex.
    day_share = (present.groupby("counter_site_id").size() / n_days).reindex(names.index, fill_value=0)
    complete = day_share.index[day_share >= MIN_DAY_SHARE]

    checked = present[present["counter_site_id"].isin(complete)]
    checked = checked.assign(
        score=checked.groupby("counter_site_id")["channels_all"].transform(outlier_scores))
    # The most extreme day of each site
    worst = checked.loc[checked.groupby("counter_site_id")["score"].idxmax()].set_index("counter_site_id")
    median = checked.groupby("counter_site_id")["channels_all"].median()
    keep = worst.index[worst["score"] <= MAX_MAHALANOBIS]

    incomplete = day_share[day_share < MIN_DAY_SHARE].sort_values()
    outliers = worst[worst["score"] > MAX_MAHALANOBIS].sort_values("score", ascending=False)
    print(f"{len(names)} sites, {len(keep)} kept, {len(incomplete) + len(outliers)} dropped.")
    print(f"\nDropped: data on fewer than {MIN_DAY_SHARE:.0%} of the {n_days} days "
          f"from {COVERAGE_START} to {COVERAGE_END} ({len(incomplete)} sites)")
    for site, share in incomplete.items():
        missing = n_days - round(share * n_days)
        print(f"  {site}  {share:6.1%} of days ({missing:4} missing)  {names[site]}")
    print(f"\nDropped: a daily peak with shifted Mahalanobis distance above {MAX_MAHALANOBIS} "
          f"({len(outliers)} sites)")
    for site, row in outliers.iterrows():
        print(f"  {site}  distance {row['score']:7.0f} on {row['date']:%Y-%m-%d}: peak "
              f"{row['channels_all']} per hour, median {median[site]:g}  {names[site]}")

    return hourly[hourly["counter_site_id"].isin(keep)]


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


def daily_stats(counts: pd.DataFrame) -> pd.DataFrame:
    """Daily min, max and mean of every numeric feature per site and local day.

    Days are local days (TIMEZONE) inside the coverage window, as in daily_peaks.
    Columns are named <feature>_min, <feature>_max and <feature>_mean; a day with
    only missing values for a feature gets NaN there.
    """
    features = counts.drop(columns=["counter_site_id"]).select_dtypes("number").columns
    date = counts["datetime"].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    daily = (
        counts.assign(date=date)
        .groupby(["counter_site_id", "date"])[list(features)]
        .agg(["min", "max", "mean"])
    )
    daily.columns = [f"{feature}_{stat}" for feature, stat in daily.columns]
    daily = daily.reset_index()
    return daily[daily["date"].between(COVERAGE_START, COVERAGE_END)].reset_index(drop=True)


def main() -> None:
    hourly = fetch_all()
    hourly.to_csv(DATA / "counters_raw.csv")
    counts, sites = prepare(select_sites(hourly))
    sites.to_csv(DATA / "location_data.csv")
    counts.to_csv(DATA / "counters_hourly.csv")
    daily = daily_stats(counts)
    daily.to_csv(DATA / "counters_daily.csv", index=False)
    print(f"{len(sites)} sites, {len(counts)} hourly and {len(daily)} daily rows written to {DATA}")


if __name__ == "__main__":
    main()
