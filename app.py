# app.py
import io
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict

# =========================
# App Config
# =========================
st.set_page_config(page_title="Board Meeting Dashboard", page_icon="📊", layout="wide")

REGIONS = ["KSA", "UAE", "Qatar", "Kuwait", "Oman", "Bahrain"]
METRICS = ["Sales", "EBITDA", "EBITDA Margin", "Units", "ASP"]
SEED = 42


# =========================
# Mock Data (cached)
# =========================
@dataclass
class MockConfig:
    start_q: str = "2019Q1"
    periods: int = 24
    regions: List[str] = None

    def __post_init__(self):
        if self.regions is None:
            self.regions = REGIONS

@st.cache_data(show_spinner=False)
def get_mock_data(cfg: MockConfig = MockConfig(), seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.period_range(cfg.start_q, periods=cfg.periods, freq="Q")
    rows = []
    base_sales = {"KSA": 95, "UAE": 70, "Qatar": 40, "Kuwait": 35, "Oman": 28, "Bahrain": 22}
    growth = {r: rng.normal(0.015, 0.005) for r in cfg.regions}
    margin_targets = {"KSA": 0.27, "UAE": 0.25, "Qatar": 0.24, "Kuwait": 0.22, "Oman": 0.20, "Bahrain": 0.19}

    for r in cfg.regions:
        level = base_sales[r]
        for q in idx:
            seasonality = [0.98, 1.00, 1.03, 1.07][q.quarter - 1]
            level = level * (1 + growth[r]) + rng.normal(0, 2.5)
            sales = max(level * seasonality, 5)

            asp = rng.normal(2.30, 0.08)            # '000 SAR
            units = max(sales / asp, 1.0)
            margin = np.clip(rng.normal(margin_targets[r], 0.02), 0.12, 0.35)
            ebitda = sales * margin

            rows.append({
                "Region": r,
                "Quarter": q.to_timestamp(how="end"),
                "Sales": round(sales, 2),
                "Units": round(units * 1_000, 0),
                "ASP": round(asp * 1_000, 2),       # SAR
                "EBITDA Margin": round(margin, 4),
                "EBITDA": round(ebitda, 2),
                "EBITDA Target": margin_targets[r]
            })
    return pd.DataFrame(rows)


# =========================
# Upload & Parse (cached)
# =========================
@st.cache_data(show_spinner=False)
def parse_uploaded_bytes(file_bytes: bytes, ext: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if ext == ".csv":
        df = pd.read_csv(bio)
    else:
        df = pd.read_excel(bio)
    if "Quarter" in df.columns:
        try:
            df["Quarter"] = pd.to_datetime(df["Quarter"])
        except Exception:
            pass
    return df

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    ext = ".csv" if name.endswith(".csv") else ".xlsx"
    return parse_uploaded_bytes(uploaded_file.getvalue(), ext)


# =========================
# Forecasting: Lite & HW
# =========================
def _seasonal_indices_4q(ser: pd.Series) -> np.ndarray:
    """Get last 4 seasonal multipliers via 4Q moving-average method."""
    if ser.size < 8:
        return np.array([1.0, 1.0, 1.0, 1.0])
    ma4 = ser.rolling(4, min_periods=4, center=True).mean()
    ratio = (ser / ma4).fillna(method="bfill").fillna(method="ffill").fillna(1.0)
    return ratio.tail(4).to_numpy()

def _median_qoq_growth(ser: pd.Series, lookback: int = 8) -> float:
    if ser.size < 2:
        return 0.0
    qoq = ser.pct_change().dropna()
    return float(qoq.tail(lookback).median()) if not qoq.empty else 0.0

def forecast_sales_lite(df_region: pd.DataFrame, horizon_q: int = 6) -> pd.DataFrame:
    """Fast seasonal-naive with drift."""
    ser = df_region.set_index("Quarter")["Sales"].asfreq("Q").interpolate(limit_direction="both")
    last_level = ser.iloc[-1]
    seas = _seasonal_indices_4q(ser)
    g = _median_qoq_growth(ser, lookback=8)

    future_idx = pd.period_range(ser.index[-1], periods=horizon_q+1, freq="Q")[1:].to_timestamp(how="end")
    fc_vals = []
    for h, _ in enumerate(future_idx, start=1):
        fc = last_level * ((1.0 + g) ** h) * seas[(ser.index[-1].quarter - 1 + h) % 4]
        fc_vals.append(fc)

    return pd.concat([ser.rename("Sales"), pd.Series(fc_vals, index=future_idx, name="Forecast")], axis=1)

def forecast_sales(df_region: pd.DataFrame, horizon_q: int = 6, engine: str = "hw") -> pd.DataFrame:
    """
    engine: 'hw' (Holt-Winters via statsmodels) or 'lite' (seasonal-naive with drift)
    """
    if engine == "lite":
        return forecast_sales_lite(df_region, horizon_q=horizon_q)

    # Lazy import to speed initial app render
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ser = df_region.set_index("Quarter")["Sales"].asfreq("Q").interpolate(limit_direction="both")
    model = ExponentialSmoothing(
        ser, trend="add", seasonal="add", seasonal_periods=4, initialization_method="estimated"
    ).fit(optimized=True)
    fc = model.forecast(horizon_q)
    return pd.concat([ser.rename("Sales"), fc.rename("Forecast")], axis=1)


# =========================
# Narrative
# =========================
def build_narrative(data: pd.DataFrame, regions: List[str]) -> Dict[str, str]:
    d = data[data["Region"].isin(regions)].sort_values("Quarter")
    recent = d[d["Quarter"] == d["Quarter"].max()]
    total_sales = recent["Sales"].sum()
    total_ebitda = recent["EBITDA"].sum()
    blended_margin = total_ebitda / max(total_sales, 1e-9)
    q_latest = d["Quarter"].max()
    q_yago = (pd.Period(q_latest, freq="Q") - 4).to_timestamp(how="end")
    sales_latest = d[d["Quarter"] == q_latest]["Sales"].sum()
    sales_yago = d[d["Quarter"] == q_yago]["Sales"].sum()
    yoy = ((sales_latest - sales_yago) / max(sales_yago, 1e-9)) * 100 if sales_yago else np.nan
    return {
        "performance": f"Sales {q_latest:%b %Y}: SAR {total_sales:,.1f}m, EBITDA SAR {total_ebitda:,.1f}m ({blended_margin*100:.1f}%), YoY {yoy:.1f}%",
        "risks": "Margin pressure from costs and pricing mix; slower growth in selected regions.",
        "actions": "Optimise mix, renegotiate procurement, targeted promotions, SG&A control.",
        "outlook": "Mid-single-digit growth expected; margin recovery initiatives underway."
    }

def render_narrative(narr: Dict[str, str]) -> str:
    return (
        "## Narrative Summary\n"
        "### Performance\n" + narr["performance"] + "\n\n"
        "### Risks\n" + narr["risks"] + "\n\n"
        "### Actions\n" + narr["actions"] + "\n\n"
        "### Outlook\n" + narr["outlook"] + "\n"
    )


# =========================
# Plotly Charts
# =========================
def plot_timeseries(df, regions, metric):
    d = df[df["Region"].isin(regions)].sort_values("Quarter")
    agg = d.groupby("Quarter")[metric].sum().reset_index()
    fig = go.Figure([go.Scatter(x=agg["Quarter"], y=agg[metric], mode="lines+markers", name=metric)])
    fig.update_layout(title=f"{metric} — Historical Trend", hovermode="x unified",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_forecast_chart(df, regions, engine: str):
    d = df[df["Region"].isin(regions)].sort_values("Quarter")
    agg = d.groupby("Quarter")["Sales"].sum().reset_index()
    fc = forecast_sales(agg, horizon_q=6, engine=engine)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Sales"], name="Sales"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Forecast"],
                             name=("Forecast - Lite" if engine == "lite" else "Forecast - HW"),
                             line=dict(dash="dash")))
    fig.update_layout(title=f"Sales Forecast — Next 6 Quarters ({'Lite' if engine=='lite' else 'Holt-Winters'})",
                      hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_margin_vs_target(df, regions):
    last_q = df["Quarter"].max()
    d = df[(df["Region"].isin(regions)) & (df["Quarter"] == last_q)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Region"], y=d["EBITDA Margin"]*100, name="Actual"))
    fig.add_trace(go.Scatter(x=d["Region"], y=d["EBITDA Target"]*100, mode="lines+markers", name="Target"))
    fig.update_layout(title=f"EBITDA Margin vs Target — {last_q:%b %Y}",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_region_compare(df, regions, metric):
    last_q = df["Quarter"].max()
    metric_cmp = metric if metric != "EBITDA Margin" else "EBITDA"
    d = df[(df["Region"].isin(regions)) & (df["Quarter"] == last_q)]
    fig = go.Figure([go.Bar(x=d["Region"], y=d[metric_cmp], name=metric_cmp)])
    fig.update_layout(title=f"Regional Comparison — {metric_cmp} ({last_q:%b %Y})",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# =========================
# Excel Exporters
# =========================
def make_excel_png_pack(dff, figs, narrative_md):
    # Only used when clicking download; safe to try Kaleido then fall back
    try:
        pio.kaleido.scope.default_width = 1200
        pio.kaleido.scope.default_height = 650
    except Exception:
        pass

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dff.to_excel(writer, sheet_name="Data", index=False)

        ws_narr = writer.book.add_worksheet("Narrative")
        writer.sheets["Narrative"] = ws_narr
        ws_narr.write(0, 0, "Board Narrative")
        ws_narr.write(2, 0, narrative_md)

        ws_ch = writer.book.add_worksheet("Charts")
        writer.sheets["Charts"] = ws_ch
        row = 1
        for name, fig in figs.items():
            try:
                img = fig.to_image(format="png", scale=2)
                ws_ch.write(row, 1, name)
                ws_ch.insert_image(row + 1, 1, f"{name}.png", {"image_data": io.BytesIO(img)})
                row += 35
            except Exception as e:
                ws_ch.write(row, 1, f"{name} (image export unavailable: {e})")
                row += 2

    output.seek(0)
    return output.getvalue()

def make_excel_native_pack(dff, regions, metric, narrative_md, engine: str):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        dff.to_excel(writer, sheet_name="Data", index=False)

        ws_narr = wb.add_worksheet("Narrative"); writer.sheets["Narrative"] = ws_narr
        ws_narr.write(0, 0, "Board Narrative"); ws_narr.write(2, 0, narrative_md)

        ws_cd = wb.add_worksheet("ChartData"); writer.sheets["ChartData"] = ws_cd

        # Chart data for all 4 charts
        df_sel = dff[dff["Region"].isin(regions)]
        agg = df_sel.groupby("Quarter").agg(Sales=("Sales", "sum")).reset_index()
        q_str = agg["Quarter"].dt.to_period("Q").astype(str)

        # A) Sales trend
        ws_cd.write_row(0, 0, ["Quarter", "Sales"])
        for i, (qs, s) in enumerate(zip(q_str, agg["Sales"])):
            ws_cd.write_row(i + 1, 0, [qs, float(s)])

        # B) Sales forecast (respect engine)
        fc = forecast_sales(agg, horizon_q=6, engine=engine)
        q_fc_str = fc.index.to_period("Q").astype(str)
        start_fc_row = len(agg) + 3
        ws_cd.write_row(start_fc_row, 0, ["Quarter", "Sales", "Forecast"])
        for i, (qs, s, f) in enumerate(zip(q_fc_str, fc["Sales"], fc["Forecast"])):
            ws_cd.write_row(start_fc_row + i + 1, 0,
                            [qs, float(s) if pd.notna(s) else None, float(f) if pd.notna(f) else None])

        # C) Margin vs Target (last quarter)
        last_q = df_sel["Quarter"].max()
        d_mg = df_sel[df_sel["Quarter"] == last_q]
        start_mg_row = start_fc_row + len(fc) + 3
        ws_cd.write_row(start_mg_row, 0, ["Region", "Actual Margin (%)", "Target Margin (%)"])
        for i, (r, a, t) in enumerate(zip(d_mg["Region"], d_mg["EBITDA Margin"] * 100, d_mg["EBITDA Target"] * 100)):
            ws_cd.write_row(start_mg_row + i + 1, 0, [r, float(a), float(t)])

        # D) Regional comparison
        metric_cmp = metric if metric != "EBITDA Margin" else "EBITDA"
        d_cmp = df_sel[df_sel["Quarter"] == last_q]
        start_cmp_row = start_mg_row + len(d_mg) + 3
        ws_cd.write_row(start_cmp_row, 0, ["Region", metric_cmp])
        for i, (r, v) in enumerate(zip(d_cmp["Region"], d_cmp[metric_cmp])):
            ws_cd.write_row(start_cmp_row + i + 1, 0, [r, float(v)])

        # Charts sheet
        ws_ch = wb.add_worksheet("Charts"); writer.sheets["Charts"] = ws_ch

        # 1) Sales trend
        ch1 = wb.add_chart({"type": "line"})
        ch1.add_series({
            "name": "Sales",
            "categories": f"ChartData!$A$2:$A${len(agg) + 1}",
            "values":     f"ChartData!$B$2:$B${len(agg) + 1}",
        })
        ch1.set_title({"name": "Historical Trend"}); ch1.set_x_axis({"name": "Quarter"}); ch1.set_y_axis({"name": "Sales (SAR m)"})
        ws_ch.insert_chart("B2", ch1, {"x_scale": 1.2, "y_scale": 1.0})

        # 2) Sales forecast
        ch2 = wb.add_chart({"type": "line"})
        ch2.add_series({
            "name": "Sales",
            "categories": f"ChartData!$A${start_fc_row + 2}:$A${start_fc_row + 1 + len(fc)}",
            "values":     f"ChartData!$B${start_fc_row + 2}:$B${start_fc_row + 1 + len(fc)}",
        })
        ch2.add_series({
            "name": "Forecast",
            "categories": f"ChartData!$A${start_fc_row + 2}:$A${start_fc_row + 1 + len(fc)}",
            "values":     f"ChartData!$C${start_fc_row + 2}:$C${start_fc_row + 1 + len(fc)}",
        })
        ch2.set_title({"name": f"Sales Forecast (Next 6 Quarters) — {'Lite' if engine=='lite' else 'HW'}"})
        ch2.set_x_axis({"name": "Quarter"}); ch2.set_y_axis({"name": "Sales (SAR m)"})
        ws_ch.insert_chart("B20", ch2, {"x_scale": 1.2, "y_scale": 1.0})

        # 3) Margin vs Target (combo)
        ch3 = wb.add_chart({"type": "column"})
        ch3.add_series({
            "name": "Actual Margin (%)",
            "categories": f"ChartData!$A${start_mg_row + 2}:$A${start_mg_row + 1 + len(d_mg)}",
            "values":     f"ChartData!$B${start_mg_row + 2}:$B${start_mg_row + 1 + len(d_mg)}",
        })
        ch3_line = wb.add_chart({"type": "line"})
        ch3_line.add_series({
            "name": "Target Margin (%)",
            "categories": f"ChartData!$A${start_mg_row + 2}:$A${start_mg_row + 1 + len(d_mg)}",
            "values":     f"ChartData!$C${start_mg_row + 2}:$C${start_mg_row + 1 + len(d_mg)}",
            "y2_axis": True
        })
        ch3.set_title({"name": "EBITDA Margin vs Target"}); ch3.set_y_axis({"name": "Margin (%)"}); ch3_line.set_y2_axis({"name": "Target (%)"})
        ch3.combine(ch3_line)
        ws_ch.insert_chart("B38", ch3, {"x_scale": 1.2, "y_scale": 1.0})

        # 4) Regional comparison
        ch4 = wb.add_chart({"type": "column"})
        ch4.add_series({
            "name": metric_cmp,
            "categories": f"ChartData!$A${start_cmp_row + 2}:$A${start_cmp_row + 1 + len(d_cmp)}",
            "values":     f"ChartData!$B${start_cmp_row + 2}:$B${start_cmp_row + 1 + len(d_cmp)}",
        })
        ch4.set_title({"name": f"Regional Comparison — {metric_cmp}"}); ch4.set_y_axis({"name": metric_cmp})
        ws_ch.insert_chart("B56", ch4, {"x_scale": 1.2, "y_scale": 1.0})

    output.seek(0)
    return output.getvalue()


# =========================
# NEW: Sample Data Template Generator
# =========================
def make_sample_template(regions: List[str] = REGIONS, blank_rows: int = 500) -> bytes:
    """
    Builds an XLSX with:
      - Template: headers + blank rows + data validation on Region
      - Lists (hidden): region list feeding validation
      - Instructions: usage notes + column definitions
      - Example: a few mocked rows to illustrate structure
    """
    cols = ["Region", "Quarter", "Sales", "Units", "ASP", "EBITDA Margin", "EBITDA", "EBITDA Target"]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        # Template (empty table)
        df_empty = pd.DataFrame(columns=cols)
        df_empty.to_excel(writer, sheet_name="Template", index=False)
        ws_t = writer.sheets["Template"]
        ws_t.set_column("A:A", 14)  # Region
        ws_t.set_column("B:B", 18)  # Quarter
        ws_t.set_column("C:H", 16)

        # Pre-create blank rows to help users
        for r in range(1, blank_rows + 1):
            pass  # worksheet already has blank space; validations will apply to range

        # Lists sheet for Region validation
        ws_l = wb.add_worksheet("Lists")
        writer.sheets["Lists"] = ws_l
        for i, reg in enumerate(regions):
            ws_l.write(i, 0, reg)
        ws_l.hide()

        # Data validations
        ws_t.data_validation(1, 0, blank_rows, 0, {  # A2:A{blank_rows+1}
            "validate": "list",
            "source": "=Lists!$A$1:$A$%d" % len(regions),
            "input_title": "Pick a Region",
            "input_message": "Choose from the dropdown."
        })
        # Quarter as date format (no strict validation, just format)
        date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})
        ws_t.set_column("B:B", 18, date_fmt)

        # Instructions
        ws_i = wb.add_worksheet("Instructions")
        writer.sheets["Instructions"] = ws_i
        ws_i.set_column("A:A", 100)
        notes = (
            "How to use this template:\n"
            "1) Enter one row per Region & Quarter.\n"
            "2) Quarter: any date within the quarter (e.g., 2025-03-31) – the app will treat it as quarterly.\n"
            "3) Sales, Units, ASP in absolute terms; EBITDA Margin as a decimal (e.g., 0.25 for 25%).\n"
            "4) EBITDA and EBITDA Target are optional; if provided, they will be used for charts and targets.\n"
            "5) Keep column names unchanged.\n\n"
            "Columns:\n"
            "- Region (dropdown)\n"
            "- Quarter (date)\n"
            "- Sales (numeric)\n"
            "- Units (numeric)\n"
            "- ASP (numeric)\n"
            "- EBITDA Margin (0–1)\n"
            "- EBITDA (numeric)\n"
            "- EBITDA Target (0–1)"
        )
        ws_i.write(0, 0, notes)

        # Example data
        ws_e = wb.add_worksheet("Example")
        writer.sheets["Example"] = ws_e
        example_df = get_mock_data(MockConfig(periods=8)).head(12)  # small sample
        example_df.to_excel(writer, sheet_name="Example", index=False)

    output.seek(0)
    return output.getvalue()


# =========================
# Main App
# =========================
def main():
    st.markdown("<h2 style='margin-top:0'>📊 Board Meeting Dashboard</h2>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Data Source")
        # NEW: sample template download button
        st.download_button(
            "📄 Download sample data template (.xlsx)",
            data=make_sample_template(),
            file_name="board_data_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
        if uploaded:
            df = load_uploaded_file(uploaded)
            st.success("✅ Uploaded file loaded.")
        else:
            st.info("Using mock data (upload to override).")
            df = get_mock_data()

        st.header("Controls")
        regions = st.multiselect("Regions", options=REGIONS, default=REGIONS)
        metric = st.selectbox("Metric", options=METRICS, index=0)
        quarters_to_show = st.slider("Timeframe (quarters)", 8, 24, 16)

        engine_label = st.radio("Forecast engine", ["Holt-Winters (accurate)", "Lite (no statsmodels)"], index=0)
        engine_key = "hw" if engine_label.startswith("Holt") else "lite"

    if df.empty:
        st.warning("No data available.")
        return

    max_q = df["Quarter"].max()
    recent_cut = (pd.Period(max_q, freq="Q") - (quarters_to_show - 1)).to_timestamp(how="end")
    dff = df[df["Quarter"] >= recent_cut]

    if not regions:
        st.warning("Select at least one region.")
        return

    if engine_key == "lite":
        st.info("⚡ Lite forecast mode enabled (seasonal-naive with drift). Switch to Holt-Winters for higher accuracy.")

    # Charts
    fig_ts = plot_timeseries(dff, regions, "Sales" if metric == "Sales" else metric)
    fig_fc = plot_forecast_chart(dff, regions, engine=engine_key)
    fig_mg = plot_margin_vs_target(dff, regions)
    fig_cmp = plot_region_compare(dff, regions, metric)

    st.plotly_chart(fig_ts, use_container_width=True)
    st.plotly_chart(fig_fc, use_container_width=True)
    st.plotly_chart(fig_mg, use_container_width=True)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Narrative
    narr_md = render_narrative(build_narrative(dff, regions))
    st.markdown(narr_md)
    st.dataframe(dff.sort_values(["Quarter", "Region"]))

    # Exports
    figs = {
        "Historical Trend": fig_ts,
        "Sales Forecast": fig_fc,
        "Margin vs Target": fig_mg,
        "Regional Comparison": fig_cmp
    }
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Excel Pack (PNG Charts)",
            data=make_excel_png_pack(dff, figs, narr_md),
            file_name="Board_Pack_PNG.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with c2:
        st.download_button(
            "📊 Excel Pack (Editable Charts)",
            data=make_excel_native_pack(dff, regions, metric, narr_md, engine=engine_key),
            file_name="Board_Pack_Editable.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
