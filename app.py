# app.py
import io
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------- Config ----------
st.set_page_config(page_title="Board Meeting Dashboard", page_icon="📊", layout="wide")

REGIONS = ["KSA", "UAE", "Qatar", "Kuwait", "Oman", "Bahrain"]
METRICS = ["Sales", "EBITDA", "EBITDA Margin", "Units", "ASP"]
SEED = 42

# ---------- Data generation ----------
@dataclass
class MockConfig:
    start_q: str = "2019Q1"
    periods: int = 24
    regions: List[str] = None
    def __post_init__(self):
        if self.regions is None:
            self.regions = REGIONS

def generate_mock_data(cfg: MockConfig, seed: int = SEED) -> pd.DataFrame:
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
            asp = rng.normal(2.30, 0.08)
            units = max(sales / asp, 1.0)
            margin = np.clip(rng.normal(margin_targets[r], 0.02), 0.12, 0.35)
            ebitda = sales * margin
            rows.append({
                "Region": r,
                "Quarter": q.to_timestamp(how="end"),
                "Sales": round(sales, 2),
                "Units": round(units * 1_000, 0),
                "ASP": round(asp * 1_000, 2),
                "EBITDA Margin": round(margin, 4),
                "EBITDA": round(ebitda, 2),
                "EBITDA Target": margin_targets[r]
            })
    return pd.DataFrame(rows)

# ---------- File upload ----------
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    if "Quarter" in df.columns:
        try:
            df["Quarter"] = pd.to_datetime(df["Quarter"])
        except Exception:
            pass
    return df

# ---------- Forecasting ----------
def forecast_sales(df_region: pd.DataFrame, horizon_q: int = 6) -> pd.DataFrame:
    ser = df_region.set_index("Quarter")["Sales"].asfreq("Q")
    ser = ser.interpolate(limit_direction="both")
    model = ExponentialSmoothing(
        ser, trend="add", seasonal="add", seasonal_periods=4, initialization_method="estimated"
    ).fit(optimized=True)
    fc = model.forecast(horizon_q)
    return pd.concat([ser.rename("Sales"), fc.rename("Forecast")], axis=1)

# ---------- Narrative ----------
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
        "risks": "Margin pressure from costs and pricing mix; slower growth in some regions.",
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

# ---------- Charts ----------
def plot_timeseries(df, regions, metric):
    d = df[df["Region"].isin(regions)].sort_values("Quarter")
    agg = d.groupby("Quarter")[metric].sum().reset_index()
    return go.Figure([go.Scatter(x=agg["Quarter"], y=agg[metric], mode="lines+markers", name=metric)])

def plot_forecast_chart(df, regions):
    d = df[df["Region"].isin(regions)].sort_values("Quarter")
    agg = d.groupby("Quarter")["Sales"].sum().reset_index()
    fc = forecast_sales(agg)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Sales"], name="Sales"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Forecast"], name="Forecast", line=dict(dash="dash")))
    return fig

def plot_margin_vs_target(df, regions):
    last_q = df["Quarter"].max()
    d = df[(df["Region"].isin(regions)) & (df["Quarter"] == last_q)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Region"], y=d["EBITDA Margin"]*100, name="Actual"))
    fig.add_trace(go.Scatter(x=d["Region"], y=d["EBITDA Target"]*100, mode="lines+markers", name="Target"))
    return fig

def plot_region_compare(df, regions, metric):
    last_q = df["Quarter"].max()
    metric_cmp = metric if metric != "EBITDA Margin" else "EBITDA"
    d = df[(df["Region"].isin(regions)) & (df["Quarter"] == last_q)]
    return go.Figure([go.Bar(x=d["Region"], y=d[metric_cmp], name=metric_cmp)])

# ---------- Excel Exporters ----------
def make_excel_png_pack(dff, figs, narrative_md):
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
                ws_ch.insert_image(row+1, 1, f"{name}.png", {"image_data": io.BytesIO(img)})
                row += 35
            except:
                ws_ch.write(row, 1, f"{name} (no image)")
                row += 2
    output.seek(0)
    return output.getvalue()

def make_excel_native_pack(dff, regions, metric, narrative_md):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        dff.to_excel(writer, sheet_name="Data", index=False)
        ws_narr = wb.add_worksheet("Narrative")
        writer.sheets["Narrative"] = ws_narr
        ws_narr.write(0, 0, "Board Narrative")
        ws_narr.write(2, 0, narrative_md)
        ws_cd = wb.add_worksheet("ChartData")
        writer.sheets["ChartData"] = ws_cd

        # Prepare chart data for all 4 charts
        df_sel = dff[dff["Region"].isin(regions)]
        agg = df_sel.groupby("Quarter").agg(Sales=("Sales","sum")).reset_index()
        q_str = agg["Quarter"].dt.to_period("Q").astype(str)

        # Sales trend
        ws_cd.write_row(0,0,["Quarter","Sales"])
        for i,(qs,s) in enumerate(zip(q_str, agg["Sales"])):
            ws_cd.write_row(i+1,0,[qs,float(s)])

        # Sales forecast
        fc = forecast_sales(agg)
        q_fc_str = fc.index.to_period("Q").astype(str)
        start_fc_row = len(agg)+3
        ws_cd.write_row(start_fc_row,0,["Quarter","Sales","Forecast"])
        for i,(qs,s,f) in enumerate(zip(q_fc_str, fc["Sales"], fc["Forecast"])):
            ws_cd.write_row(start_fc_row+i+1,0,[qs,float(s) if pd.notna(s) else None,float(f) if pd.notna(f) else None])

        # Margin vs target
        last_q = df_sel["Quarter"].max()
        d_mg = df_sel[df_sel["Quarter"]==last_q]
        start_mg_row = start_fc_row+len(fc)+3
        ws_cd.write_row(start_mg_row,0,["Region","Actual Margin","Target Margin"])
        for i,(r,a,t) in enumerate(zip(d_mg["Region"], d_mg["EBITDA Margin"]*100, d_mg["EBITDA Target"]*100)):
            ws_cd.write_row(start_mg_row+i+1,0,[r,a,t])

        # Regional comparison
        metric_cmp = metric if metric != "EBITDA Margin" else "EBITDA"
        d_cmp = df_sel[df_sel["Quarter"]==last_q]
        start_cmp_row = start_mg_row+len(d_mg)+3
        ws_cd.write_row(start_cmp_row,0,["Region",metric_cmp])
        for i,(r,v) in enumerate(zip(d_cmp["Region"], d_cmp[metric_cmp])):
            ws_cd.write_row(start_cmp_row+i+1,0,[r,float(v)])

        # Charts
        ws_ch = wb.add_worksheet("Charts")
        writer.sheets["Charts"] = ws_ch
        # Sales trend chart
        ch1 = wb.add_chart({"type":"line"})
        ch1.add_series({"categories":f"ChartData!$A$2:$A${len(agg)+1}",
                        "values":f"ChartData!$B$2:$B${len(agg)+1}",
                        "name":"Sales"})
        ws_ch.insert_chart("B2", ch1)
        # Forecast chart
        ch2 = wb.add_chart({"type":"line"})
        ch2.add_series({"categories":f"ChartData!$A${start_fc_row+2}:$A${start_fc_row+1+len(fc)}",
                        "values":f"ChartData!$B${start_fc_row+2}:$B${start_fc_row+1+len(fc)}",
                        "name":"Sales"})
        ch2.add_series({"categories":f"ChartData!$A${start_fc_row+2}:$A${start_fc_row+1+len(fc)}",
                        "values":f"ChartData!$C${start_fc_row+2}:$C${start_fc_row+1+len(fc)}",
                        "name":"Forecast"})
        ws_ch.insert_chart("B20", ch2)
        # Margin vs target chart
        ch3 = wb.add_chart({"type":"column"})
        ch3.add_series({"categories":f"ChartData!$A${start_mg_row+2}:$A${start_mg_row+1+len(d_mg)}",
                        "values":f"ChartData!$B${start_mg_row+2}:$B${start_mg_row+1+len(d_mg)}",
                        "name":"Actual Margin"})
        ch3_t = wb.add_chart({"type":"line"})
        ch3_t.add_series({"categories":f"ChartData!$A${start_mg_row+2}:$A${start_mg_row+1+len(d_mg)}",
                          "values":f"ChartData!$C${start_mg_row+2}:$C${start_mg_row+1+len(d_mg)}",
                          "name":"Target Margin","y2_axis":True})
        ch3.combine(ch3_t)
        ws_ch.insert_chart("B38", ch3)
        # Regional comparison chart
        ch4 = wb.add_chart({"type":"column"})
        ch4.add_series({"categories":f"ChartData!$A${start_cmp_row+2}:$A${start_cmp_row+1+len(d_cmp)}",
                        "values":f"ChartData!$B${start_cmp_row+2}:$B${start_cmp_row+1+len(d_cmp)}",
                        "name":metric_cmp})
        ws_ch.insert_chart("B56", ch4)

    output.seek(0)
    return output.getvalue()

# ---------- Main ----------
def main():
    st.title("📊 Board Meeting Dashboard")
    with st.sidebar:
        st.header("Data Source")
        uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx","xls","csv"])
        if uploaded:
            df = load_uploaded_file(uploaded)
            st.success("✅ Uploaded file loaded.")
        else:
            st.info("Using mock data.")
            df = generate_mock_data(MockConfig())
        st.header("Controls")
        regions = st.multiselect("Regions", options=REGIONS, default=REGIONS)
        metric = st.selectbox("Metric", options=METRICS, index=0)
        quarters_to_show = st.slider("Timeframe (quarters)", 8, 24, 16)
        max_q = df["Quarter"].max()
        recent_cut = (pd.Period(max_q, freq="Q") - (quarters_to_show - 1)).to_timestamp(how="end")
        dff = df[df["Quarter"] >= recent_cut]

    if not regions:
        st.warning("Select at least one region.")
        return

    fig_ts = plot_timeseries(dff, regions, "Sales" if metric=="Sales" else metric)
    fig_fc = plot_forecast_chart(dff, regions)
    fig_mg = plot_margin_vs_target(dff, regions)
    fig_cmp = plot_region_compare(dff, regions, metric)

    st.plotly_chart(fig_ts, use_container_width=True)
    st.plotly_chart(fig_fc, use_container_width=True)
    st.plotly_chart(fig_mg, use_container_width=True)
    st.plotly_chart(fig_cmp, use_container_width=True)

    narr = build_narrative(dff, regions)
    narr_md = render_narrative(narr)
    st.markdown(narr_md)
    st.dataframe(dff)

    figs = {
        "Historical Trend": fig_ts,
        "Sales Forecast": fig_fc,
        "Margin vs Target": fig_mg,
        "Regional Comparison": fig_cmp
    }
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Excel Pack (PNG Charts)",
                           data=make_excel_png_pack(dff, figs, narr_md),
                           file_name="Board_Pack_PNG.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2:
        st.download_button("📊 Excel Pack (Editable Charts)",
                           data=make_excel_native_pack(dff, regions, metric, narr_md),
                           file_name="Board_Pack_Editable.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
