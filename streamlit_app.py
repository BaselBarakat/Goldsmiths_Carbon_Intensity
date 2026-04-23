"""
Carbon Intensity Dashboard
==========================

A Streamlit dashboard for monitoring UK regional carbon intensity, backed by
the National Grid ESO Carbon Intensity API (https://carbonintensity.org.uk/).

Features
--------
* Live current intensity with visual status + user‑settable threshold alerts.
* 48‑hour forward forecast and automatic "greenest upcoming slots" picker.
* Historical trends: interactive line chart, hour×day heatmap, daily box‑plot.
* Generation‑mix breakdown (renewables vs fossil) as a stacked area chart.
* Configurable region (any UK postcode area) and history window.
* Incremental local caching to a CSV so the API is only hit for missing slots.

Run with:
    streamlit run carbon_dashboard.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UK_TZ = pytz.timezone("Europe/London")
DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "carbon.csv"
API_BASE = "https://api.carbonintensity.org.uk"
DEFAULT_POSTCODE = "ME4"
DEFAULT_HISTORY_DAYS = 14
# Earliest date to back-fill from if the CSV is empty. Kept recent to avoid
# hammering the API on first run; adjust if you want a longer back-catalogue.
DEFAULT_FETCH_START = datetime.now(pytz.UTC) - timedelta(days=30)

# Canonical intensity bands -- the single source of truth for both the
# categorical label and the colour used across all charts/markdown.
# (name, lower_bound, upper_bound, colour)
INTENSITY_BANDS: list[tuple[str, float, float, str]] = [
    ("Very Low", 0, 50, "#2ecc71"),
    ("Low", 50, 100, "#27ae60"),
    ("Moderate", 100, 200, "#f39c12"),
    ("High", 200, 300, "#e67e22"),
    ("Very High", 300, 10_000, "#c0392b"),
]
COLOR_MAP: dict[str, str] = {name.lower(): color for name, _, _, color in INTENSITY_BANDS}

# Fuel-type colour palette, roughly grouped by renewable/fossil character.
FUEL_COLOURS: dict[str, str] = {
    "gas": "#d35400",
    "coal": "#2c3e50",
    "biomass": "#8e44ad",
    "nuclear": "#f1c40f",
    "hydro": "#2980b9",
    "imports": "#7f8c8d",
    "solar": "#f39c12",
    "wind": "#27ae60",
    "other": "#95a5a6",
}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UK Carbon Intensity Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Data layer
# ===========================================================================

def ensure_utc(df: pd.DataFrame, cols: tuple[str, ...] = ("from", "to")) -> pd.DataFrame:
    """Force the given columns to tz-aware UTC datetime dtype.

    Idempotent and tolerant of mixed/stringified timestamps left behind by
    older CSV writes. Rows whose timestamps cannot be parsed are dropped
    rather than poisoning the whole column's dtype.
    """
    if df.empty:
        # Even an empty frame must carry the right dtypes for downstream
        # .dt accessors to be valid.
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # Drop rows where the primary timestamp failed to parse.
    if "from" in df.columns:
        df = df.dropna(subset=["from"]).copy()
    return df


@st.cache_data(show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    """Load the on-disk history, returning an empty frame with the expected
    schema if nothing is cached yet."""
    columns = ["from", "to", "forecast", "index"]
    if not path.exists():
        return ensure_utc(pd.DataFrame(columns=columns))

    # `parse_dates` is a best-effort hint; `ensure_utc` is the source of truth
    # and will recover from mixed or malformed timestamp strings.
    df = pd.read_csv(path, parse_dates=["from", "to"])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return ensure_utc(df)


def _flatten_records(raw: list[dict]) -> pd.DataFrame:
    """Convert the nested API response into a flat half-hour DataFrame."""
    rows: list[dict] = []
    for entry in raw:
        row = {
            "from": entry["from"],
            "to": entry["to"],
            "forecast": entry["intensity"]["forecast"],
            "index": entry["intensity"]["index"],
        }
        for mix in entry.get("generationmix", []) or []:
            row[mix["fuel"]] = mix["perc"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["from"] = pd.to_datetime(df["from"], utc=True)
    df["to"] = pd.to_datetime(df["to"], utc=True)
    return df


def _get_json(url: str) -> list[dict]:
    """GET the API, returning the inner `data.data` list or [] on any error."""
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data", {}).get("data", []) or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("API call failed (%s): %s", url, exc)
        return []


def fetch_range(start: datetime, end: datetime, postcode: str) -> list[dict]:
    """Historical intensity between two UTC timestamps for a given postcode."""
    url = (
        f"{API_BASE}/regional/intensity/"
        f"{start.strftime('%Y-%m-%dT%H:%MZ')}/"
        f"{end.strftime('%Y-%m-%dT%H:%MZ')}/postcode/{postcode}"
    )
    return _get_json(url)


@st.cache_data(ttl=1800, show_spinner="Fetching 48-hour forecast…")
def get_forward_forecast(postcode: str) -> pd.DataFrame:
    """Return the next ~48 hours of half-hourly forecast for a postcode."""
    now = datetime.now(pytz.UTC)
    url = (
        f"{API_BASE}/regional/intensity/"
        f"{now.strftime('%Y-%m-%dT%H:%MZ')}/fw48h/postcode/{postcode}"
    )
    return ensure_utc(_flatten_records(_get_json(url)))


def refresh_local_data(postcode: str) -> pd.DataFrame:
    """Ensure the local CSV is caught up to the current half-hour slot and
    return the combined dataset. Fetches incrementally from the last saved
    timestamp to avoid re-downloading existing rows."""
    df = load_local_csv(CSV_PATH).copy()

    # Work out what range is missing.
    last_ts = df["to"].max() if not df.empty else pd.NaT
    start = (last_ts + timedelta(minutes=30)) if pd.notna(last_ts) else DEFAULT_FETCH_START
    end = datetime.now(pytz.UTC).replace(second=0, microsecond=0)

    if start >= end:
        return ensure_utc(df).sort_values("from").reset_index(drop=True)

    # Fetch day-by-day so a single failure doesn't discard everything.
    all_new: list[dict] = []
    cursor = start
    status = st.sidebar.empty()
    while cursor < end:
        window_end = min(cursor + timedelta(days=1), end)
        status.caption(f"⬇️ Fetching {cursor:%Y-%m-%d}…")
        all_new.extend(fetch_range(cursor, window_end, postcode))
        cursor = window_end
    status.empty()

    new_df = _flatten_records(all_new)
    if new_df.empty:
        return ensure_utc(df).sort_values("from").reset_index(drop=True)

    combined = (
        pd.concat([df, new_df], ignore_index=True)
        .drop_duplicates(subset=["from"], keep="last")
        .sort_values("from")
        .reset_index(drop=True)
    )
    combined = ensure_utc(combined)

    # Persist for future sessions and invalidate the loader cache.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CSV_PATH, index=False)
    load_local_csv.clear()
    return combined


# ===========================================================================
# Analysis helpers
# ===========================================================================

def categorize_intensity(value: float) -> str:
    """Map a numeric forecast to its intensity band name."""
    for name, lo, hi, _ in INTENSITY_BANDS:
        if lo <= value < hi:
            return name
    return "Unknown"


def best_slots(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Return the N lowest-intensity rows, sorted chronologically."""
    if df.empty:
        return df
    out = ensure_utc(df.copy())
    out["local_time"] = out["from"].dt.tz_convert(UK_TZ)
    return out.nsmallest(n, "forecast").sort_values("from")


def hour_day_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format average intensity per (date, hour-of-day) in local time."""
    df = ensure_utc(df)
    local = df["from"].dt.tz_convert(UK_TZ)
    out = pd.DataFrame({
        "date": local.dt.date,
        "hour": local.dt.hour,
        "forecast": df["forecast"].astype(float),
    })
    return out.groupby(["date", "hour"], as_index=False)["forecast"].mean()


def fuel_columns(df: pd.DataFrame) -> list[str]:
    """Columns representing generation-mix fuel shares."""
    known = {"from", "to", "forecast", "index", "day", "hour", "local_time"}
    return [c for c in df.columns if c not in known and df[c].notna().any()]


# ===========================================================================
# Chart builders (Plotly)
# ===========================================================================

def intensity_line_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Interactive intensity time-series, coloured by band."""
    d = ensure_utc(df.copy())
    d["local_time"] = d["from"].dt.tz_convert(UK_TZ)
    d["category"] = d["forecast"].apply(categorize_intensity)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["local_time"],
        y=d["forecast"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(
            size=6,
            color=[COLOR_MAP[c.lower()] for c in d["category"]],
            line=dict(width=1, color="white"),
        ),
        customdata=d["category"],
        hovertemplate=(
            "<b>%{x|%a %d %b %H:%M}</b><br>"
            "Intensity: %{y:.0f} gCO₂/kWh<br>"
            "Band: %{customdata}<extra></extra>"
        ),
        name="Forecast",
    ))

    # Overlay dotted reference lines at each band boundary.
    for name, lo, _hi, color in INTENSITY_BANDS[1:]:
        fig.add_hline(
            y=lo,
            line=dict(color=color, width=1, dash="dot"),
            annotation_text=name,
            annotation_position="right",
            annotation_font_color=color,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (UK)",
        yaxis_title="Carbon intensity (gCO₂/kWh)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
    )
    return fig


def heatmap_chart(df: pd.DataFrame) -> go.Figure:
    """Hour-of-day × date heatmap of mean intensity."""
    data = hour_day_heatmap(df)
    pivot = data.pivot(index="hour", columns="date", values="forecast")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[
            (0.00, "#2ecc71"),
            (0.25, "#27ae60"),
            (0.50, "#f39c12"),
            (0.75, "#e67e22"),
            (1.00, "#c0392b"),
        ],
        labels=dict(x="Date", y="Hour of day", color="gCO₂/kWh"),
    )
    fig.update_layout(
        title="Mean intensity by hour of day",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def daily_boxplot(df: pd.DataFrame) -> go.Figure:
    """Day-level distribution of intensity across the history window."""
    d = ensure_utc(df.copy())
    d["day"] = d["from"].dt.tz_convert(UK_TZ).dt.date.astype(str)
    fig = px.box(
        d, x="day", y="forecast",
        labels={"day": "Day", "forecast": "gCO₂/kWh"},
        points=False,
        color_discrete_sequence=["#3498db"],
    )
    fig.update_layout(
        title="Daily distribution of carbon intensity",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def generation_mix_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Stacked-area chart of the generation-mix share over time."""
    fuels = fuel_columns(df)
    if not fuels:
        return None
    d = ensure_utc(df.copy())
    d["local_time"] = d["from"].dt.tz_convert(UK_TZ)
    long = d.melt(
        id_vars=["local_time"], value_vars=fuels,
        var_name="fuel", value_name="percent",
    )
    fig = px.area(
        long, x="local_time", y="percent", color="fuel",
        color_discrete_map=FUEL_COLOURS,
        labels={"local_time": "Time (UK)", "percent": "Share (%)", "fuel": "Fuel"},
    )
    fig.update_layout(
        title="Generation mix over time",
        height=420,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ===========================================================================
# UI sections
# ===========================================================================

def sidebar_controls() -> dict:
    """Render the sidebar and return the current user settings."""
    st.sidebar.title("⚙️ Controls")
    postcode = st.sidebar.text_input(
        "Postcode area",
        value=DEFAULT_POSTCODE,
        help="Outward portion of a UK postcode, e.g. ME4, SR1, E1W.",
    ).strip().upper()

    history_days = st.sidebar.slider(
        "History window (days)",
        min_value=1, max_value=30,
        value=DEFAULT_HISTORY_DAYS,
        help="Range used for the heatmap, box plot and statistics.",
    )

    alert_threshold = st.sidebar.slider(
        "Alert threshold (gCO₂/kWh)",
        min_value=50, max_value=500,
        value=250, step=10,
        help="Warn when live intensity is above this value.",
    )

    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh every 5 min", value=False,
        help="Streamlit re-runs periodically to pull the latest slot.",
    )

    if st.sidebar.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: [Carbon Intensity API](https://carbonintensity.org.uk/) · "
        "National Grid ESO."
    )

    return dict(
        postcode=postcode,
        history_days=history_days,
        alert_threshold=alert_threshold,
        auto_refresh=auto_refresh,
    )


def render_header_metrics(latest: pd.Series, threshold: float) -> None:
    """Top-of-page three-column metric strip + alert banner."""
    forecast = float(latest["forecast"])
    index = str(latest["index"]).title()
    ts = latest["from"].tz_convert(UK_TZ)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current intensity", f"{forecast:.0f} gCO₂/kWh", index)
    col2.metric("Slot (UK)", ts.strftime("%a %d %b, %H:%M"))
    col3.metric("Category", categorize_intensity(forecast))

    color = COLOR_MAP.get(index.lower(), "#7f8c8d")
    st.markdown(
        f"""
        <div style='padding: 12px 16px; border-left: 6px solid {color};
                    background: rgba(127,140,141,0.08); border-radius: 4px;
                    margin: 0.5rem 0 1rem 0;'>
            <b>Status:</b>
            <span style='color:{color}; font-weight:600;'>{index}</span>
            — approximately <b>{forecast:.0f} gCO₂/kWh</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if forecast >= threshold:
        st.warning(
            f"⚠️ Live intensity **{forecast:.0f} gCO₂/kWh** is above your "
            f"alert threshold of {threshold:.0f}. Consider deferring heavy usage."
        )
    else:
        st.success(
            f"✅ Live intensity **{forecast:.0f} gCO₂/kWh** is below your "
            f"alert threshold of {threshold:.0f} — a good time to run appliances."
        )


def render_best_times(forecast_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    """Show greenest upcoming slots, falling back to yesterday if needed."""
    st.subheader("⚡ Greenest upcoming windows")

    if not forecast_df.empty:
        best = best_slots(forecast_df, n=6)
        subtitle = "Based on the live 48‑hour forecast."
    else:
        # Fallback: yesterday's data from the local cache.
        df = history_df.copy()
        df["local_time"] = df["from"].dt.tz_convert(UK_TZ)
        yesterday = (datetime.now(UK_TZ) - timedelta(days=1)).date()
        df = df[df["local_time"].dt.date == yesterday]
        best = best_slots(df, n=6)
        subtitle = "Forecast unavailable — showing yesterday's greenest slots as a proxy."

    if best.empty:
        st.info("No forecast or recent data available to recommend windows.")
        return

    st.caption(subtitle)
    display = best.assign(
        When=best["local_time"].dt.strftime("%a %d %b · %H:%M"),
        End=(best["local_time"] + pd.Timedelta(minutes=30)).dt.strftime("%H:%M"),
        Intensity=best["forecast"].round(0).astype(int).astype(str) + " gCO₂/kWh",
        Category=best["forecast"].apply(categorize_intensity),
    )[["When", "End", "Intensity", "Category"]]

    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption(
        "💡 Shift dishwashers, EV charging or laundry into these windows "
        "to cut your electricity emissions."
    )


def render_live_forecast_tab(
    latest: pd.Series,
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame,
    threshold: float,
) -> None:
    render_header_metrics(latest, threshold)

    if not forecast_df.empty:
        st.plotly_chart(
            intensity_line_chart(forecast_df, "48‑hour forward forecast"),
            use_container_width=True,
        )
    else:
        st.info("Forward forecast is currently unavailable from the API.")

    render_best_times(forecast_df, history_df)


def render_historical_tab(history_df: pd.DataFrame, days: int) -> None:
    cutoff = datetime.now(pytz.UTC) - timedelta(days=days)
    recent = history_df[history_df["from"] >= cutoff]

    if recent.empty:
        st.info("No historical data in the selected window.")
        return

    st.plotly_chart(
        intensity_line_chart(recent, f"Last {days} days · half‑hourly"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(heatmap_chart(recent), use_container_width=True)
    with col2:
        st.plotly_chart(daily_boxplot(recent), use_container_width=True)


def render_generation_tab(history_df: pd.DataFrame, days: int) -> None:
    cutoff = datetime.now(pytz.UTC) - timedelta(days=days)
    recent = history_df[history_df["from"] >= cutoff]
    fig = generation_mix_chart(recent)

    if fig is None:
        st.info("Generation-mix data is not present in the cached dataset.")
        return

    st.plotly_chart(fig, use_container_width=True)

    fuels = fuel_columns(recent)
    if fuels:
        # Summary table: mean share per fuel across the window.
        summary = (
            recent[fuels].mean().round(1).sort_values(ascending=False)
            .rename_axis("Fuel").reset_index(name="Mean share (%)")
        )
        st.subheader("Mean share by fuel")
        st.dataframe(summary, hide_index=True, use_container_width=True)


def render_data_explorer_tab(history_df: pd.DataFrame) -> None:
    st.subheader("📅 Filter history")

    min_ts = history_df["from"].min().to_pydatetime()
    max_ts = history_df["from"].max().to_pydatetime()

    if min_ts == max_ts:
        st.info("Not enough data yet to show a range slider.")
        return

    start, end = st.slider(
        "Select a date range",
        min_value=min_ts, max_value=max_ts,
        value=(max_ts - timedelta(days=7), max_ts),
        format="YYYY-MM-DD HH:mm",
    )

    mask = (history_df["from"] >= pd.Timestamp(start, tz="UTC")) & (
        history_df["from"] <= pd.Timestamp(end, tz="UTC")
    )
    subset = history_df.loc[mask].sort_values("from", ascending=False)

    st.caption(f"{len(subset):,} half‑hour slots in the selection.")
    st.dataframe(subset, use_container_width=True, height=320)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Summary statistics")
        numeric = subset.select_dtypes("number")
        st.dataframe(numeric.describe().round(2), use_container_width=True)
    with c2:
        st.subheader("⬇️ Export")
        st.download_button(
            "Download CSV",
            subset.to_csv(index=False).encode("utf-8"),
            file_name=f"carbon_intensity_{start:%Y%m%d}_{end:%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ===========================================================================
# Main app
# ===========================================================================

def main() -> None:
    settings = sidebar_controls()

    # Auto-refresh via a lightweight meta tag. Simpler and dep-free compared
    # to a background thread or streamlit-autorefresh.
    if settings["auto_refresh"]:
        st.markdown(
            '<meta http-equiv="refresh" content="300">', unsafe_allow_html=True
        )

    st.title("🌍 UK Carbon Intensity Dashboard")
    st.caption(
        f"Region: **{settings['postcode']}**  ·  "
        f"History window: **{settings['history_days']} days**  ·  "
        f"Alert threshold: **{settings['alert_threshold']} gCO₂/kWh**"
    )

    # Pull data (incrementally) and the live forward forecast.
    with st.spinner("Updating local dataset…"):
        history_df = refresh_local_data(settings["postcode"])
    history_df = ensure_utc(history_df)

    if history_df.empty:
        st.error(
            "No carbon-intensity data is available yet. Check your network "
            "connection and the postcode, then press 🔄 Refresh now."
        )
        return

    forecast_df = get_forward_forecast(settings["postcode"])

    latest = (
        forecast_df.iloc[0] if not forecast_df.empty
        else history_df.sort_values("from").iloc[-1]
    )

    tab_live, tab_hist, tab_gen, tab_data = st.tabs(
        ["Live & Forecast", "Historical Trends", "Generation Mix", "Data Explorer"]
    )

    with tab_live:
        render_live_forecast_tab(
            latest, forecast_df, history_df, settings["alert_threshold"]
        )
    with tab_hist:
        render_historical_tab(history_df, settings["history_days"])
    with tab_gen:
        render_generation_tab(history_df, settings["history_days"])
    with tab_data:
        render_data_explorer_tab(history_df)


if __name__ == "__main__":
    main()
