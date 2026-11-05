"""
Animal Welfare Monitoring POC — Executive Dashboard

Run locally:
    streamlit run app.py

Expected CSV files:
    data/curated/video_flags.csv
    data/curated/acoustic_records.csv
    data/curated/process_records.csv
    data/curated/welfare_events.csv
    data/curated/ear_tag_records.csv
    data/curated/data_quality_report.csv
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_DIR / "data" / "curated"

NAVY = "#0B1F3A"
TEAL = "#1F9E9E"
AMBER = "#E0A100"
RED = "#C8443B"
STEEL = "#5B7083"
LIGHT = "#F4F6F9"
PALETTE = [NAVY, TEAL, AMBER, RED, STEEL, "#8FB8DE", "#9AA5B1", "#2E5E8C"]

FILES = {
    "video": "video_flags.csv",
    "acoustic": "acoustic_records.csv",
    "process": "process_records.csv",
    "events": "welfare_events.csv",
    "tags": "ear_tag_records.csv",
    "quality": "data_quality_report.csv",
}

REQUIRED_COLUMNS = {
    "video": ["ts_start", "ts_end", "zone", "event_type", "confidence", "reviewed", "review_outcome"],
    "acoustic": ["ts", "zone", "noise_db", "vocalisation_rate"],
    "process": ["ts", "zone", "wait_minutes", "queue_length", "line_speed"],
    "events": ["ts_logged", "zone", "category", "stated_cause"],
    "tags": ["arrival_ts", "animal_id", "batch_id", "origin_farm", "transport_hours"],
    "quality": ["dataset", "valid_records", "total_records", "final_records",
                "rejected_records", "duplicate_records"],
}

DATE_COLUMNS = {
    "video": ["ts_start", "ts_end"],
    "acoustic": ["ts"],
    "process": ["ts"],
    "events": ["ts_logged"],
    "tags": ["arrival_ts"],
}

st.set_page_config(
    page_title="Animal Welfare Monitoring",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: {LIGHT}; }}
    .block-container {{ padding-top: .8rem; max-width: 1800px; }}
    .hdr {{
        background: {NAVY}; color: white; padding: 20px 26px;
        border-radius: 10px; margin-bottom: 16px;
    }}
    .hdr h1 {{ margin: 0; font-size: 27px; color: white; }}
    .hdr p {{ margin: 5px 0 0 0; color: #B8C4D6; font-size: 13px; }}
    .kpi {{
        background: white; border-left: 5px solid {NAVY}; border-radius: 8px;
        padding: 11px 13px; box-shadow: 0 1px 3px rgba(0,0,0,.07);
        min-height: 91px;
    }}
    .kpi .l {{ font-size: 10px; text-transform: uppercase; letter-spacing: .7px; color: {STEEL}; }}
    .kpi .v {{ font-size: 25px; font-weight: 700; color: {NAVY}; line-height: 1.25; }}
    .kpi .s {{ font-size: 10px; color: #8793A3; }}
    .sec {{
        color: {NAVY}; font-size: 17px; font-weight: 700;
        border-bottom: 2px solid {TEAL}; padding-bottom: 4px; margin: 22px 0 8px 0;
    }}
    .note {{
        background: #FFF8E5; border-left: 4px solid {AMBER}; padding: 9px 12px;
        font-size: 12px; color: #5A4A12; border-radius: 4px; margin: 8px 0 12px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", color=NAVY):
    col.markdown(
        f'<div class="kpi" style="border-left-color:{color}">'
        f'<div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial", size=12, color=NAVY),
        title_font=dict(size=14, color=NAVY),
        legend=dict(orientation="h", y=-0.22),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor="#E8ECF1")
    fig.update_yaxes(gridcolor="#E8ECF1")
    return fig


def show(fig, height=330):
    st.plotly_chart(style_fig(fig, height), use_container_width=True)


@st.cache_data(show_spinner="Loading curated datasets...")
def load_data(folder_string):
    folder = Path(folder_string).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {folder}")

    frames = {}
    missing = []
    for key, filename in FILES.items():
        file_path = folder / filename
        if not file_path.is_file():
            missing.append(str(file_path))
            continue
        # utf-8-sig handles standard UTF-8 CSVs with or without a BOM.
        # cp1252 is a fallback for CSVs exported from some Windows tools.
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="cp1252")

        missing_columns = [c for c in REQUIRED_COLUMNS[key] if c not in df.columns]
        if missing_columns:
            raise ValueError(
                f"{filename} is missing required column(s): {', '.join(missing_columns)}"
            )
        for col in DATE_COLUMNS.get(key, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        frames[key] = df

    if missing:
        raise FileNotFoundError(
            "The following required CSV file(s) are missing:\n\n"
            + "\n".join(f"• {item}" for item in missing)
        )

    frames["video"]["reviewed"] = (
        frames["video"]["reviewed"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "y"])
    )
    frames["video"]["review_outcome"] = (
        frames["video"]["review_outcome"].fillna("").astype(str).str.strip()
    )
    return frames


def filter_window(df, timestamp_column, start_date, end_date, selected_zones=None):
    if df.empty:
        return df.copy()
    valid = df[timestamp_column].notna()
    dates = df[timestamp_column].dt.date
    mask = valid & (dates >= start_date) & (dates <= end_date)
    if selected_zones is not None and "zone" in df.columns:
        mask &= df["zone"].isin(selected_zones)
    return df.loc[mask].copy()


def safe_mean(series):
    return float(series.mean()) if len(series.dropna()) else np.nan


# ---------------------------------------------------------------------
# Sidebar and data loading
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Dashboard filters")
    data_folder = st.text_input("Curated CSV folder", value=str(DEFAULT_DATA_DIR))
    st.caption("The default is the data/curated folder beside app.py.")

try:
    data = load_data(data_folder)
except Exception as exc:
    st.error("The dashboard could not load its datasets.")
    st.code(str(exc))
    st.info(
        "Check that all six curated CSV files are inside the selected folder "
        "and that their column names match the expected schema."
    )
    st.stop()

video_all = data["video"]
acoustic_all = data["acoustic"]
process_all = data["process"]
events_all = data["events"]
tags_all = data["tags"]
quality = data["quality"]

date_series = [
    video_all["ts_start"],
    acoustic_all["ts"],
    process_all["ts"],
    events_all["ts_logged"],
]
all_timestamps = pd.concat(date_series, ignore_index=True).dropna()

if all_timestamps.empty:
    st.error("No valid timestamps were found in the video, acoustic, process or welfare-event datasets.")
    st.stop()

min_date = all_timestamps.min().date()
max_date = all_timestamps.max().date()

zone_values = []
for frame in [video_all, acoustic_all, process_all, events_all]:
    if "zone" in frame.columns:
        zone_values.extend(frame["zone"].dropna().astype(str).unique().tolist())
all_zones = sorted(set(zone_values))

with st.sidebar:
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        elif len(selected_dates) == 1:
            start_date = end_date = selected_dates[0]
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = selected_dates

    selected_zones = st.multiselect("Zones", all_zones, default=all_zones)
    outcomes = sorted(x for x in video_all["review_outcome"].unique() if x)
    likely_confirmed = [
        x for x in outcomes
        if re.search(r"confirm|true|valid|real|yes", x, re.I)
        and not re.search(r"not|false|un|no\\b", x, re.I)
    ]
    confirmed_labels = st.multiselect(
        "Review outcomes counted as confirmed",
        outcomes,
        default=likely_confirmed,
        help="Select the review_outcome values that your review process defines as confirmed.",
    )
    timeline_grain = st.radio("Timeline grain", ["Hour", "Day"], horizontal=True)
    event_types = sorted(video_all["event_type"].dropna().astype(str).unique())
    selected_types = st.multiselect("Video event types", event_types, default=event_types)
    event_categories = sorted(events_all["category"].dropna().astype(str).unique())
    selected_categories = st.multiselect(
        "Welfare event categories", event_categories, default=event_categories
    )

# Apply filters. The date range is based on the selected monitoring streams.
video = filter_window(video_all, "ts_start", start_date, end_date, selected_zones)
acoustic = filter_window(acoustic_all, "ts", start_date, end_date, selected_zones)
process = filter_window(process_all, "ts", start_date, end_date, selected_zones)
events = filter_window(events_all, "ts_logged", start_date, end_date, selected_zones)
tags = filter_window(tags_all, "arrival_ts", start_date, end_date, selected_zones=None)

video = video[video["event_type"].astype(str).isin(selected_types)]
events = events[events["category"].astype(str).isin(selected_categories)]
video["confirmed"] = video["reviewed"] & video["review_outcome"].isin(confirmed_labels)

freq = "60min" if timeline_grain == "Hour" else "1D"

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hdr">
      <h1>Animal Welfare Monitoring — Executive Overview</h1>
      <p>Video · Acoustic · Process · Welfare events · Ear-tag traceability
      &nbsp;|&nbsp; {start_date:%d %b %Y} – {end_date:%d %b %Y}
      &nbsp;|&nbsp; Proof of concept using simulated data</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not selected_zones:
    st.warning("No zones are selected. Zone-based monitoring streams will show no records.")

# ---------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------
reviewed_count = int(video["reviewed"].sum())
confirmed_count = int(video["confirmed"].sum())
validation_rate = (
    quality["valid_records"].sum() / quality["total_records"].sum() * 100
    if len(quality) and quality["total_records"].sum() else np.nan
)
review_rate = reviewed_count / len(video) * 100 if len(video) else np.nan
confirmation_rate = confirmed_count / reviewed_count * 100 if reviewed_count else np.nan

kpi_rows = [st.columns(5), st.columns(5)]
kpi(kpi_rows[0][0], "Animals", f"{tags['animal_id'].nunique():,}", "Unique ear tags")
kpi(kpi_rows[0][1], "Batches", f"{tags['batch_id'].nunique():,}", f"{tags['origin_farm'].nunique()} origin farms")
kpi(kpi_rows[0][2], "Video flags", f"{len(video):,}", "AI detections, not all confirmed", AMBER)
kpi(kpi_rows[0][3], "Reviewed", f"{review_rate:.0f}%" if pd.notna(review_rate) else "—", f"{reviewed_count:,} of {len(video):,}", AMBER)
kpi(kpi_rows[0][4], "Confirmation rate", f"{confirmation_rate:.0f}%" if pd.notna(confirmation_rate) else "—", f"{confirmed_count:,} confirmed of reviewed", AMBER)
kpi(kpi_rows[1][0], "Welfare events", f"{len(events):,}", "Client-recorded events", RED)
kpi(kpi_rows[1][1], "Avg wait (min)", f"{safe_mean(process['wait_minutes']):.1f}" if process["wait_minutes"].notna().any() else "—", f"Avg queue: {safe_mean(process['queue_length']):.1f}" if process["queue_length"].notna().any() else "", TEAL)
kpi(kpi_rows[1][2], "Avg line speed", f"{safe_mean(process['line_speed']):.1f}" if process["line_speed"].notna().any() else "—", "Process stream", TEAL)
kpi(kpi_rows[1][3], "Avg noise (dB)", f"{safe_mean(acoustic['noise_db']):.1f}" if acoustic["noise_db"].notna().any() else "—", f"Vocalisation: {safe_mean(acoustic['vocalisation_rate']):.2f}" if acoustic["vocalisation_rate"].notna().any() else "", TEAL)
kpi(kpi_rows[1][4], "Data validation", f"{validation_rate:.0f}%" if pd.notna(validation_rate) else "—", "Valid / total records", STEEL)

# ---------------------------------------------------------------------
# 1. Unified timeline
# ---------------------------------------------------------------------
section("1 | Unified monitoring timeline")

video_flags = video.groupby(video["ts_start"].dt.floor(freq)).size().rename("Video flags")
confirmed_flags = (
    video.loc[video["confirmed"]]
    .groupby(video.loc[video["confirmed"], "ts_start"].dt.floor(freq))
    .size().rename("Confirmed flags")
)
welfare_counts = events.groupby(events["ts_logged"].dt.floor(freq)).size().rename("Welfare events")
process_trend = process.groupby(process["ts"].dt.floor(freq)).agg(
    wait=("wait_minutes", "mean"), queue=("queue_length", "mean")
)
acoustic_trend = acoustic.groupby(acoustic["ts"].dt.floor(freq)).agg(
    noise=("noise_db", "mean"), vocalisation=("vocalisation_rate", "mean")
)
timeline = pd.concat(
    [video_flags, confirmed_flags, welfare_counts, process_trend, acoustic_trend], axis=1
).sort_index()

if timeline.empty:
    st.info("No records match the selected date and zone filters.")
else:
    for column in ["Video flags", "Confirmed flags", "Welfare events"]:
        if column not in timeline:
            timeline[column] = 0
        timeline[column] = timeline[column].fillna(0)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Video flags, confirmed flags and recorded welfare events",
            "Operational pressure: waiting time and queue length",
            "Acoustic conditions: noise and vocalisation rate",
        ),
    )
    fig.add_bar(x=timeline.index, y=timeline["Video flags"], name="Video flags",
                marker_color="#8FB8DE", row=1, col=1)
    fig.add_bar(x=timeline.index, y=timeline["Confirmed flags"], name="Confirmed flags",
                marker_color=AMBER, row=1, col=1)
    fig.add_scatter(x=timeline.index, y=timeline["Welfare events"], name="Welfare events",
                    mode="lines+markers", line=dict(color=RED, width=2), row=1, col=1)
    if "wait" in timeline:
        fig.add_scatter(x=timeline.index, y=timeline["wait"], name="Avg wait (min)",
                        line=dict(color=NAVY, width=2), row=2, col=1)
    if "queue" in timeline:
        fig.add_scatter(x=timeline.index, y=timeline["queue"], name="Avg queue",
                        line=dict(color=TEAL, width=2, dash="dot"),
                        row=2, col=1, secondary_y=True)
    if "noise" in timeline:
        fig.add_scatter(x=timeline.index, y=timeline["noise"], name="Avg noise (dB)",
                        line=dict(color=STEEL, width=2), row=3, col=1)
    if "vocalisation" in timeline:
        fig.add_scatter(x=timeline.index, y=timeline["vocalisation"], name="Vocalisation rate",
                        line=dict(color=RED, width=1.5, dash="dot"),
                        row=3, col=1, secondary_y=True)
    fig.update_layout(barmode="group", hovermode="x unified")
    show(fig, 650)

# ---------------------------------------------------------------------
# 2. Zone-level comparison
# ---------------------------------------------------------------------
section("2 | Operational pressure by zone")

zone_index = pd.Index(selected_zones, name="zone")
zone_summary = pd.DataFrame(index=zone_index)
zone_summary["Avg wait"] = process.groupby("zone")["wait_minutes"].mean()
zone_summary["Avg queue"] = process.groupby("zone")["queue_length"].mean()
zone_summary["Line speed"] = process.groupby("zone")["line_speed"].mean()
zone_summary["Noise dB"] = acoustic.groupby("zone")["noise_db"].mean()
zone_summary["Vocalisation rate"] = acoustic.groupby("zone")["vocalisation_rate"].mean()
zone_summary["Video flags"] = video.groupby("zone").size()
zone_summary["Confirmed flags"] = video.loc[video["confirmed"]].groupby("zone").size()
zone_summary["Welfare events"] = events.groupby("zone").size()
zone_summary = zone_summary.fillna(0)

left, right = st.columns([1.15, 1])
with left:
    if len(zone_summary) and zone_summary.to_numpy().size:
        normalized = (zone_summary - zone_summary.min()) / (
            zone_summary.max() - zone_summary.min()
        ).replace(0, np.nan)
        heatmap = go.Figure(go.Heatmap(
            z=normalized.fillna(0.5).values,
            x=zone_summary.columns,
            y=zone_summary.index.astype(str),
            colorscale=[[0, "#EAF3F3"], [1, NAVY]],
            text=zone_summary.round(2).values,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="%{y} | %{x}: %{text}<extra></extra>",
        ))
        heatmap.update_layout(
            title="Zone profile (colour = relative level within each metric; labels = actual values)"
        )
        show(heatmap, 340)
    else:
        st.info("No zone-level records match the current filters.")

with right:
    # Aggregate each stream by zone-hour, then join explicitly on the shared keys.
    parts = []
    if not process.empty:
        parts.append(process.assign(hour=process["ts"].dt.floor("60min"))
                     .groupby(["zone", "hour"]).agg(
                         wait=("wait_minutes", "mean"), queue=("queue_length", "mean")).reset_index())
    if not acoustic.empty:
        parts.append(acoustic.assign(hour=acoustic["ts"].dt.floor("60min"))
                     .groupby(["zone", "hour"]).agg(
                         noise=("noise_db", "mean"), vocalisation=("vocalisation_rate", "mean")).reset_index())
    if not video.empty:
        parts.append(video.assign(hour=video["ts_start"].dt.floor("60min"))
                     .groupby(["zone", "hour"]).size().rename("flags").reset_index())
        parts[-1] = parts[-1].rename(columns={"ts_start": "hour"})
    if not events.empty:
        parts.append(events.assign(hour=events["ts_logged"].dt.floor("60min"))
                     .groupby(["zone", "hour"]).size().rename("events").reset_index())

    if parts:
        hourly = parts[0]
        for part in parts[1:]:
            hourly = hourly.merge(part, on=["zone", "hour"], how="outer")
        correlation_pairs = [
            ("wait", "flags", "Wait vs video flags"),
            ("queue", "flags", "Queue vs video flags"),
            ("noise", "flags", "Noise vs video flags"),
            ("vocalisation", "flags", "Vocalisation vs video flags"),
            ("wait", "events", "Wait vs welfare events"),
            ("noise", "events", "Noise vs welfare events"),
        ]
        correlations = []
        for x_col, y_col, label in correlation_pairs:
            if x_col in hourly and y_col in hourly:
                pair = hourly[[x_col, y_col]].dropna()
                if len(pair) > 3 and pair[x_col].nunique() > 1 and pair[y_col].nunique() > 1:
                    correlations.append((label, pair[x_col].rank().corr(pair[y_col].rank())))
        corr_df = pd.DataFrame(correlations, columns=["Comparison", "Spearman rho"])
        if not corr_df.empty:
            corr_fig = px.bar(
                corr_df, x="Spearman rho", y="Comparison", orientation="h",
                color="Spearman rho",
                color_continuous_scale=["#C8443B", "#E8ECF1", TEAL],
                range_color=(-1, 1),
                title="Rank correlation across zone-hours (association only)",
            )
            corr_fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            show(corr_fig, 340)
        else:
            st.info("Not enough overlapping zone-hour observations to calculate correlations.")
    else:
        st.info("Not enough data to compare the streams by zone and hour.")

st.markdown(
    '<div class="note"><b>Interpretation:</b> correlations describe co-movement in the selected '
    'zone-hour observations. They do not establish causation. Video flags are AI detections; '
    'only reviewed and confirmed flags should be treated as confirmed detections. Client-recorded '
    'welfare events are a separate source.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 3. Video analysis
# ---------------------------------------------------------------------
section("3 | Video detection and confirmation analysis")
c1, c2, c3 = st.columns([1, 1.2, 1])
with c1:
    funnel = go.Figure(go.Funnel(
        y=["Detected", "Reviewed", "Confirmed"],
        x=[len(video), reviewed_count, confirmed_count],
        marker_color=["#8FB8DE", TEAL, AMBER],
        textinfo="value+percent initial",
    ))
    funnel.update_layout(title="Detection and review funnel")
    show(funnel, 320)
with c2:
    if not video.empty:
        video_view = video.copy()
        video_view["outcome"] = np.where(
            video_view["reviewed"],
            video_view["review_outcome"].replace("", "No outcome"),
            "Not reviewed",
        )
        outcome_counts = video_view.groupby(["event_type", "outcome"]).size().reset_index(name="n")
        fig = px.bar(outcome_counts, x="event_type", y="n", color="outcome",
                     title="Flags by event type and review outcome")
        fig.update_layout(xaxis_title="", yaxis_title="Flags")
        show(fig, 320)
    else:
        st.info("No video flags match the selected filters.")
with c3:
    if not video.empty:
        video_view = video.copy()
        video_view["outcome"] = np.where(
            video_view["reviewed"],
            video_view["review_outcome"].replace("", "No outcome"),
            "Not reviewed",
        )
        fig = px.box(video_view, x="outcome", y="confidence", color="outcome",
                     title="Model confidence by review outcome")
        fig.update_layout(showlegend=False, xaxis_title="")
        show(fig, 320)
    else:
        st.info("No video confidence data is available.")

# ---------------------------------------------------------------------
# 4. Welfare events and stated causes
# ---------------------------------------------------------------------
section("4 | Recorded welfare events and client-stated causes")
c1, c2, c3 = st.columns(3)
with c1:
    if not events.empty:
        grouped = events.groupby(["category", "zone"]).size().reset_index(name="Events")
        fig = px.bar(grouped, x="category", y="Events", color="zone",
                     title="Events by category and zone")
        fig.update_layout(xaxis_title="")
        show(fig)
    else:
        st.info("No welfare events match the selected filters.")
with c2:
    if not events.empty:
        causes = (events.groupby("stated_cause").size().sort_values().tail(10)
                  .reset_index(name="Events"))
        fig = px.bar(causes, x="Events", y="stated_cause", orientation="h",
                     title="Top client-stated causes", color_discrete_sequence=[NAVY])
        fig.update_layout(yaxis_title="")
        show(fig)
    else:
        st.info("No stated causes are available.")
with c3:
    comparison = zone_summary[["Video flags", "Confirmed flags", "Welfare events"]].reset_index()
    if not comparison.empty:
        comparison = comparison.melt(id_vars="zone", var_name="Stream", value_name="Count")
        fig = px.bar(comparison, x="zone", y="Count", color="Stream", barmode="group",
                     title="Video flags vs confirmed flags vs events")
        fig.update_layout(xaxis_title="")
        show(fig)
    else:
        st.info("No zone comparison is available.")

if not events.empty:
    cause_category = events.groupby(["category", "stated_cause"]).size().reset_index(name="Count")
    if not cause_category.empty:
        tree = px.treemap(
            cause_category, path=["category", "stated_cause"], values="Count",
            title="Root-cause exploration (category → stated cause)",
            color_discrete_sequence=PALETTE,
        )
        show(tree, 340)

# ---------------------------------------------------------------------
# 5. Ear-tag traceability
# ---------------------------------------------------------------------
section("5 | Batch and transport insights")
st.markdown(
    '<div class="note">The ear-tag dataset does not currently contain a shared animal or batch key '
    'in the other monitoring streams. Therefore, batches are not linked to individual video flags '
    'or welfare events; this section describes arrivals and transport only.</div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    if not tags.empty:
        unique_animals = tags.drop_duplicates("animal_id")
        fig = px.histogram(unique_animals, x="transport_hours", nbins=20,
                           title="Transport duration distribution",
                           color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title="Transport hours", yaxis_title="Animals")
        show(fig)
    else:
        st.info("No ear-tag arrivals match the selected dates.")
with c2:
    if not tags.empty:
        farms = (tags.groupby("origin_farm")
                 .agg(animals=("animal_id", "nunique"),
                      avg_transport_hours=("transport_hours", "mean"))
                 .sort_values("animals").tail(10).reset_index())
        fig = px.bar(farms, x="animals", y="origin_farm", orientation="h",
                     color="avg_transport_hours",
                     color_continuous_scale=["#EAF3F3", NAVY],
                     title="Origin farms (colour = average transport hours)")
        fig.update_layout(yaxis_title="", xaxis_title="Animals")
        show(fig)
    else:
        st.info("No origin-farm data is available.")
with c3:
    if not tags.empty:
        daily = (tags.groupby(tags["arrival_ts"].dt.date)
                 .agg(animals=("animal_id", "nunique"),
                      avg_transport_hours=("transport_hours", "mean")).reset_index())
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=daily["arrival_ts"], y=daily["animals"],
                    name="Animals arriving", marker_color="#8FB8DE")
        fig.add_scatter(x=daily["arrival_ts"], y=daily["avg_transport_hours"],
                        name="Avg transport hours", line=dict(color=AMBER, width=2),
                        secondary_y=True)
        fig.update_layout(title="Daily arrivals and average transport time")
        show(fig)
    else:
        st.info("No daily arrival data is available.")

# ---------------------------------------------------------------------
# 6. Data quality and coverage
# ---------------------------------------------------------------------
section("6 | Data quality and monitoring coverage")
c1, c2 = st.columns(2)
with c1:
    if not quality.empty:
        quality_plot = quality.melt(
            id_vars="dataset",
            value_vars=["final_records", "rejected_records", "duplicate_records"],
            var_name="Record type",
            value_name="Records",
        )
        fig = px.bar(quality_plot, x="dataset", y="Records", color="Record type",
                     title="Records by dataset",
                     color_discrete_sequence=[TEAL, RED, AMBER])
        fig.update_layout(xaxis_title="", yaxis_title="Records")
        show(fig)
    else:
        st.info("The data-quality report is empty.")
with c2:
    coverage_parts = []
    for frame, timestamp_col, stream_name in [
        (video, "ts_start", "Video"),
        (acoustic, "ts", "Acoustic"),
        (process, "ts", "Process"),
        (events, "ts_logged", "Welfare events"),
    ]:
        if not frame.empty:
            counts = frame.groupby(frame[timestamp_col].dt.date).size().rename(stream_name)
            coverage_parts.append(counts)
    if coverage_parts:
        coverage = pd.concat(coverage_parts, axis=1).fillna(0).astype(int).T
        fig = go.Figure(go.Heatmap(
            z=coverage.values,
            x=[str(x) for x in coverage.columns],
            y=coverage.index,
            colorscale=[[0, "#FBE9E7"], [0.02, "#EAF3F3"], [1, TEAL]],
            text=coverage.values,
            texttemplate="%{text}",
            showscale=False,
        ))
        fig.update_layout(title="Monitoring coverage: records per stream per day")
        show(fig, 340)
    else:
        st.info("No monitoring coverage records match the selected filters.")

if not quality.empty:
    with st.expander("View data-quality report"):
        st.dataframe(quality, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Automatic read-out and methodology note
# ---------------------------------------------------------------------
section("Automatic read-out")
if not zone_summary.empty and zone_summary["Video flags"].sum() > 0 and zone_summary["Welfare events"].sum() > 0:
    highest_wait_zone = zone_summary["Avg wait"].idxmax()
    highest_flags_zone = zone_summary["Video flags"].idxmax()
    highest_events_zone = zone_summary["Welfare events"].idxmax()
    same_zone = len({highest_wait_zone, highest_flags_zone, highest_events_zone}) == 1
    message = (
        f"Highest average waiting time: **{highest_wait_zone}**. "
        f"Most video flags: **{highest_flags_zone}**. "
        f"Most recorded welfare events: **{highest_events_zone}**. "
    )
    message += (
        "All three indicators point to the same zone, which may deserve closer review. "
        "This pattern alone does not establish a cause."
        if same_zone else
        "The indicators point to different zones, illustrating why integrated monitoring "
        "can provide more context than a single data stream."
    )
    st.info(message)
else:
    st.info("There are not enough filtered video flags and welfare events to generate this read-out.")

st.caption(
    "Proof of concept. Interpret results in the context of data completeness, review protocols "
    "and operational knowledge. Associations are not evidence of causation."
)
