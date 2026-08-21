import os
import io
import shutil
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq, curve_fit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Equipment Degradation and Prognostic Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Equipment Degradation & Prognostic Analysis Tool")
st.markdown("This application models physical asset degradation trends and uses curve-fitting models to project Remaining Useful Life (RUL) and forecasted dates for **Alert** and **Danger** threshold breaches.")

# ==========================================
# 1. USER INPUTS & PARAMETERS (SIDEBAR)
# ==========================================
st.sidebar.header("1. Asset Information")
complex_name = st.sidebar.text_input("Complex Name", value="Complex A")
equipment_name = st.sidebar.text_input("Equipment Name", value="Pump-101")
analysis_title = st.sidebar.text_input("Analysis Title / Parameter", value="Filter Fouling Trend")

st.sidebar.header("2. Data Source")
data_source = st.sidebar.selectbox(
    "Choose Analysis Mode:",
    ["Upload Excel File", "Copy & Paste Bulk Data", "Sample Data"]
)

df = None

if data_source == "Upload Excel File":
    uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset (.xlsx / .xls)", type=["xlsx", "xls"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        st.info("👈 Please upload an Excel file in the sidebar to proceed.")
        st.stop()

elif data_source == "Copy & Paste Bulk Data":
    st.sidebar.markdown("**Paste Data Below** (Format: Two columns: `Timestamp`, `Value`)")
    paste_data = st.sidebar.text_area("Paste tab or comma-separated data:", height=150, 
                                     placeholder="12/05/2026\t0.312\n12/05/2026\t0.311\n13/05/2026\t0.316")
    if paste_data.strip():
        try:
            df = pd.read_csv(io.StringIO(paste_data), sep=None, engine='python', header=None)
        except Exception as e:
            st.error(f"Error parsing pasted data: {e}")
            st.stop()
    else:
        st.info("👈 Please paste tabular data in the text area to proceed.")
        st.stop()

else:  # Sample Data Mode
    st.sidebar.success("✅ Running with Synthetic Asset Degradation Data")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=20, freq='15D')
    days_passed = np.arange(20) * 15
    synthetic_degradation = 5.0 + 0.08 * (days_passed ** 1.2) + np.random.normal(0, 1.2, 20)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "value": synthetic_degradation
    })

# Engineering Parameters Inputs
st.sidebar.header("3. Engineering Parameters")
param_unit = st.sidebar.text_input("Measurement Unit", value="bar")

trend_direction = st.sidebar.selectbox(
    "Degradation Trend Direction:",
    ["Progressive Upwards (High is Bad)", "Progressive Downwards (High is Good)"]
)
is_increasing = (trend_direction == "Progressive Upwards (High is Bad)")

# Set default thresholds based on direction mode (Formatted to 3 decimal places)
default_alert = 0.400 if is_increasing else 0.200
default_danger = 0.500 if is_increasing else 0.100

ALERT_THRESHOLD = st.sidebar.number_input(f"Alert Threshold [{param_unit}]", value=default_alert, step=0.001, format="%.3f")
DANGER_THRESHOLD = st.sidebar.number_input(f"Danger Threshold [{param_unit}]", value=default_danger, step=0.001, format="%.3f")
CONFIDENCE_PCT = st.sidebar.number_input("Confidence Level Analysis [%]", value=95.0, min_value=50.0, max_value=99.9, step=1.0)

# Setup Output Directory
OUTPUT_DIR = "Prognosis_Output_Files"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
if len(df.columns) >= 2:
    df = df.iloc[:, :2]
    df.columns = ["timestamp", "value"]

df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna().sort_values("timestamp").reset_index(drop=True)

if df.empty:
    st.error("❌ No valid data points found. Please check your input date format or values.")
    st.stop()

t0 = df["timestamp"].iloc[0]
days = (df["timestamp"] - t0).dt.total_seconds().values / 86400.0
degradation_val = df["value"].values
latest_val = df["value"].iloc[-1]

# ==========================================
# 3. REGRESSION MODELING SUITE (DIRECTION-AWARE)
# ==========================================
def _lin(x, a, b): return a * x + b
def _quad(x, a, b, c): return a * x**2 + b * x + c
def _power(x, a, b): return a * np.power(np.maximum(x, 1e-6), b)
def _expo(x, a, b): return a * np.exp(np.clip(b * x, -100, 100))
def _logf(x, a, b): return a * np.log(x + 1.0) + b
def _lognorm(x, a, shape, scale): return a * stats.lognorm.cdf(np.maximum(x, 1e-6), s=shape, scale=scale)
def _weibull(x, a, beta, eta): return a * (1.0 - np.exp(-1.0 * (np.maximum(x, 1e-6) / eta)**beta))
def _loglogis(x, a, alpha, beta): return a * (1.0 / (1.0 + (np.maximum(x, 1e-6) / alpha)**(-beta)))

max_v = max(np.max(degradation_val) * 2.5, 100.0)
mean_d = max(np.mean(days), 1.0)

if is_increasing:
    lin_bounds = ([0, -np.inf], [np.inf, np.inf])
    quad_bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
    expo_bounds = ([0, 0], [np.inf, np.inf])
    log_bounds = ([0, -np.inf], [np.inf, np.inf])
else:
    lin_bounds = ([-np.inf, -np.inf], [0, np.inf])
    quad_bounds = ([-np.inf, -np.inf, -np.inf], [0, 0, np.inf])
    expo_bounds = ([0, -np.inf], [np.inf, 0])
    log_bounds = ([-np.inf, -np.inf], [0, np.inf])

MODELS = {
    "Linear": (_lin, [0.001 if is_increasing else -0.001, np.mean(degradation_val)], lin_bounds),
    "Quadratic": (_quad, [0.0001 if is_increasing else -0.0001, 0.001 if is_increasing else -0.001, np.mean(degradation_val)], quad_bounds),
    "Exponential": (_expo, [np.mean(degradation_val), 0.001 if is_increasing else -0.001], expo_bounds),
    "Logarithmic": (_logf, [0.01 if is_increasing else -0.01, np.mean(degradation_val)], log_bounds),
}

if is_increasing:
    MODELS["Power Law"] = (_power, [0.1, 1.2], (0, np.inf))
    MODELS["Log-Normal CDF"] = (_lognorm, [max_v, 1.0, mean_d], ([0, 0.01, 0.1], [max_v * 5, 10.0, 50000]))
    MODELS["Weibull CDF"] = (_weibull, [max_v, 1.5, mean_d], ([0, 0.1, 0.1], [max_v * 5, 10.0, 50000]))
    MODELS["Log-Logistic CDF"] = (_loglogis, [max_v, mean_d, 1.5], ([0, 0.1, 0.1], [max_v * 5, 50000, 10.0]))

model_results = {}
for name, (func, p0, bnds) in MODELS.items():
    try:
        popt, _ = curve_fit(func, days, degradation_val, p0=p0, bounds=bnds, maxfev=30000)
        y_pred = func(days, *popt)
        ss_res = np.sum((degradation_val - y_pred)**2)
        ss_tot = np.sum((degradation_val - np.mean(degradation_val))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        dof = max(len(days) - len(popt), 1)
        resid_std = np.sqrt(ss_res / dof)
        model_results[name] = {
            "func": func, "popt": popt, "r2": r2,
            "resid_std": resid_std, "dof": dof, "ss_res": ss_res
        }
    except Exception:
        pass

if not model_results:
    st.error("❌ Unable to fit regression models with the selected degradation direction. Check threshold orientation.")
    st.stop()

# --- MODEL SELECTION BY USER ---
st.sidebar.header("4. Model Selection")
auto_best = max(model_results, key=lambda k: model_results[k]["r2"])
model_options = ["Auto (Select Best R²)"] + list(model_results.keys())
selected_model_option = st.sidebar.selectbox("Regression Model Choice:", model_options)

if selected_model_option == "Auto (Select Best R²)":
    best_name = auto_best
else:
    best_name = selected_model_option

best = model_results[best_name]

# ==========================================
# 4. EXPLANATION & FIT METRICS DISPLAY
# ==========================================
st.subheader("📊 Model Comparison & Fit Metrics")

with st.expander("💡 Technical Guidance: Metrics & RUL Definition"):
    st.markdown("""
    * **Remaining Useful Life (RUL):** The calculated operational time remaining before degradation reaches or passes an Alert or Danger limit.
      $$\\text{RUL (Days)} = \\text{Expected Breach Date} - \\text{Last Observed Data Date}$$
    * **Degradation Direction Mode:** 
      * **Progressive Upwards:** Values increase toward limits (e.g., Vibration, Temperature, Fouling Delta-P).
      * **Progressive Downwards:** Values decrease toward limits (e.g., Component Thickness, Pressure, Flow Rate).
    * **$R^2$ Score (Coefficient of Determination):** Measures goodness-of-fit ($1.0$ indicates a perfect mathematical fit).
    """)

model_comparison_data = []
for name, res in model_results.items():
    model_comparison_data.append({
        "Model Name": name,
        "R² Score": f"{res['r2']:.4f}",
        "Residual Std": f"{res['resid_std']:.3f} {param_unit}".strip(),
        "Status": "✅ Selected" if name == best_name else ("Best Fit" if name == auto_best else "Candidate")
    })
st.dataframe(pd.DataFrame(model_comparison_data), use_container_width=True)

# ==========================================
# 5. METRICS & PROGNOSTIC BREACH SUMMARY
# ==========================================
m1, m2, m3 = st.columns(3)
m1.metric("Selected Model", f"{best_name}", f"R² = {best['r2']:.4f}")
m2.metric("Current Data Value", f"{latest_val:.3f} {param_unit}".strip())
m3.metric("Direction Mode", "Upwards ⬆️" if is_increasing else "Downwards ⬇️")

def solve_crossing(model, target_val, conf_pct, max_days=36500):
    func, popt, dof, std = model["func"], model["popt"], model["dof"], model["resid_std"]
    t_val = stats.t.ppf((1 + conf_pct / 100.0) / 2, dof) if dof >= 1 else 0.0
    band = t_val * std

    def get_date(f):
        xs = np.linspace(0, max_days, 30000)
        ys = f(xs)
        idx = np.where(np.diff(np.sign(ys)) != 0)[0]
        if len(idx) == 0: return None
        try: return brentq(f, xs[idx[0]], xs[idx[0]+1])
        except: return None

    if is_increasing:
        early = get_date(lambda x: func(x, *popt) + band - target_val)
        central = get_date(lambda x: func(x, *popt) - target_val)
        late = get_date(lambda x: func(x, *popt) - band - target_val)
    else:
        early = get_date(lambda x: func(x, *popt) - band - target_val)
        central = get_date(lambda x: func(x, *popt) - target_val)
        late = get_date(lambda x: func(x, *popt) + band - target_val)

    return early, central, late

max_horizon = max(days[-1] * 10, 36500)
f_Alert = solve_crossing(best, ALERT_THRESHOLD, CONFIDENCE_PCT, max_horizon)
f_Danger = solve_crossing(best, DANGER_THRESHOLD, CONFIDENCE_PCT, max_horizon)

st.subheader("📋 Prognostic Breach & RUL Projection Summary")
prognosis_data = []
latest_day = days[-1]

targets_info = [
    ("Alert Limit", ALERT_THRESHOLD, f_Alert),
    ("Danger Limit", DANGER_THRESHOLD, f_Danger)
]

for label, threshold_val, (e, c, l) in targets_info:
    c_date = (t0 + timedelta(days=c)).strftime('%Y-%m-%d') if c else "N/A"
    e_date = (t0 + timedelta(days=e)).strftime('%Y-%m-%d') if e else "N/A"
    l_date = (t0 + timedelta(days=l)).strftime('%Y-%m-%d') if l else "N/A"
    rul_days = f"{int(c - latest_day)} Days" if c and c >= latest_day else "Exceeded / N/A"

    prognosis_data.append({
        "Threshold Level": label,
        "Threshold Value": f"{threshold_val:.3f} {param_unit}".strip(),
        "Earliest Date": e_date,
        "Expected Date": c_date,
        "Latest Date": l_date,
        "RUL (Days)": rul_days
    })

st.table(pd.DataFrame(prognosis_data))

# ==========================================
# 6. VISUALIZATION
# ==========================================
candidate_days = [d for d in [f_Alert[1], f_Danger[1]] if d is not None]
x_max_plot = max(candidate_days) * 1.15 if candidate_days else max(days[-1] * 3, 30)
x_plot = np.linspace(0, x_max_plot, 500)
y_plot = best["func"](x_plot, *best["popt"])
dates_plot = [t0 + timedelta(days=d) for d in x_plot]

t_val = stats.t.ppf((1 + CONFIDENCE_PCT / 100.0) / 2, best["dof"]) if best["dof"] >= 1 else 0.0
band_val = t_val * best["resid_std"]

trend_title = f"{complex_name} - {equipment_name}: {analysis_title}"
y_label_text = f"Value [{param_unit}]" if param_unit else "Value"

# --- STATIC GRAPH WITH BREACH CALLOUTS (MATPLOTLIB / PNG) ---
fig_static, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
ax.scatter(df["timestamp"], df["value"], color="#1f77b4", s=30, alpha=0.8, label="Measured Data")
ax.plot(dates_plot, y_plot, color="#d62728", linewidth=2, label=f"Model Fit ({best_name})")
ax.fill_between(dates_plot, y_plot - band_val, y_plot + band_val, color="#d62728", alpha=0.15, label=f"{CONFIDENCE_PCT:.0f}% CI")

ax.axhline(ALERT_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Alert Limit ({ALERT_THRESHOLD:.3f} {param_unit})".strip())
ax.axhline(DANGER_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.5, label=f"Danger Limit ({DANGER_THRESHOLD:.3f} {param_unit})".strip())

# Add Callout Annotations to Static Plot (Matplotlib)
callout_points_mpl = [
    ("Alert", f_Alert[1], ALERT_THRESHOLD, "#ff7f0e", (15, 20) if is_increasing else (15, -25)),
    ("Danger", f_Danger[1], DANGER_THRESHOLD, "#d62728", (-100, 25) if is_increasing else (-100, -30))
]

for label, day_val, threshold_val, color_hex, text_offset in callout_points_mpl:
    if day_val is not None and day_val >= 0:
        breach_dt = t0 + timedelta(days=day_val)
        date_str = breach_dt.strftime('%d %b %Y')
        rul_val = int(day_val - latest_day)
        
        # Diamond marker at point of reach
        ax.scatter([breach_dt], [threshold_val], color=color_hex, s=60, marker='D', zorder=5, edgecolor='white')
        
        # Text Callout Box
        ax.annotate(
            f"Expected {label}:\n{date_str} ({rul_val}d RUL)",
            xy=(breach_dt, threshold_val),
            xytext=text_offset,
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            color=color_hex,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_hex, lw=1.2, alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color_hex, lw=1.2)
        )

ax.set_ylabel(y_label_text)
ax.set_xlabel("Date")
ax.set_title(trend_title, fontweight="bold")
ax.legend(loc="upper left" if is_increasing else "lower left", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)
fig_static.autofmt_xdate()
fig_static.tight_layout()

plot_img_path = os.path.join(OUTPUT_DIR, "prognostic_trend_plot.png")
fig_static.savefig(plot_img_path, dpi=150, bbox_inches="tight")
plt.close(fig_static)

# --- INTERACTIVE GRAPH (PLOTLY) WITH BREACH CALLOUTS ---
fig_interactive = go.Figure()

# Measured Data Points
fig_interactive.add_trace(go.Scatter(
    x=df["timestamp"], y=df["value"], mode='markers', name='Measured Data', marker=dict(color='#1f77b4', size=8)
))

# Confidence Interval Lower Bound
fig_interactive.add_trace(go.Scatter(
    x=dates_plot, y=y_plot - band_val, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo="skip"
))

# Confidence Interval Upper Bound Fill
fig_interactive.add_trace(go.Scatter(
    x=dates_plot, y=y_plot + band_val, mode='lines', fill='tonexty', fillcolor='rgba(214, 39, 40, 0.15)',
    line=dict(color='rgba(255,255,255,0)'), name=f"{CONFIDENCE_PCT:.0f}% Confidence Interval", hoverinfo="skip"
))

# Best Fit Curve
fig_interactive.add_trace(go.Scatter(
    x=dates_plot, y=y_plot, mode='lines', name=f'Model Fit ({best_name})', line=dict(color='#d62728', width=2)
))

# Threshold Lines
fig_interactive.add_hline(
    y=ALERT_THRESHOLD, line_dash="dash", line_color="#ff7f0e", 
    annotation_text=f"Alert ({ALERT_THRESHOLD:.3f} {param_unit})".strip(), annotation_position="bottom right"
)
fig_interactive.add_hline(
    y=DANGER_THRESHOLD, line_dash="dash", line_color="#d62728", 
    annotation_text=f"Danger ({DANGER_THRESHOLD:.3f} {param_unit})".strip(), annotation_position="bottom right"
)

# Callout Markers & Annotations for Expected Breach Dates
callout_points_plotly = [
    ("Alert", f_Alert[1], ALERT_THRESHOLD, "#ff7f0e", "bottom center" if is_increasing else "top center"),
    ("Danger", f_Danger[1], DANGER_THRESHOLD, "#d62728", "top center" if is_increasing else "bottom center")
]

for label, day_val, threshold_val, color_hex, text_pos in callout_points_plotly:
    if day_val is not None and day_val >= 0:
        breach_dt = t0 + timedelta(days=day_val)
        date_str = breach_dt.strftime('%d %b %Y')
        rul_val = int(day_val - latest_day)
        
        # Add a prominent marker at the intersection point
        fig_interactive.add_trace(go.Scatter(
            x=[breach_dt],
            y=[threshold_val],
            mode='markers+text',
            name=f'Expected {label} Reach',
            marker=dict(color=color_hex, size=12, symbol='diamond', line=dict(color='white', width=1.5)),
            text=[f"<b>Expected {label}: {date_str}</b><br>({rul_val} Days RUL)"],
            textposition=text_pos,
            textfont=dict(color=color_hex, size=11),
            hoverinfo='text'
        ))

fig_interactive.update_layout(
    title=dict(text=f"<b>{trend_title}</b>", x=0.5),
    xaxis_title="Date",
    yaxis_title=y_label_text,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=50, b=60),
    template="plotly_white"
)

st.plotly_chart(fig_interactive, use_container_width=True)

# ==========================================
# 7. REPORT GENERATION & DOWNLOAD
# ==========================================
pdf_file_path = os.path.join(OUTPUT_DIR, f"{complex_name}_{equipment_name}_Prognostic_Report.pdf")

def generate_pdf_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # TEMA WARNA BAHARU (Steel Blue & Slate)
    PRIMARY_COLOR = colors.HexColor('#00AE9E')    
    SECONDARY_COLOR = colors.HexColor('#00509E')  
    ALT_ROW_COLOR = colors.HexColor('#F8FAFC')     # Light Gray Background
    BORDER_COLOR = colors.HexColor('#CBD5E1')      # Slate Border

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=13, textColor=PRIMARY_COLOR, alignment=1, spaceAfter=12)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#FFFFFF'), backColor=SECONDARY_COLOR, spaceBefore=10, spaceAfter=6, leftIndent=6)

    hdr_style = ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.whitesmoke, alignment=1)
    hdr_style_l = ParagraphStyle('THL', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.whitesmoke, alignment=0)
    body_style = ParagraphStyle('TD', fontName='Helvetica', fontSize=8, leading=10, alignment=1)
    body_style_l = ParagraphStyle('TDL', fontName='Helvetica', fontSize=8, leading=10, alignment=0)

    story.append(Paragraph(f"DEGRADATION AND PROGNOSTIC ANALYSIS REPORT<br/>{complex_name.upper()} - {equipment_name.upper()}", title_style))
    story.append(Spacer(1, 6))

    # SECTION 1: SPECIFICATIONS
    story.append(Paragraph("TECHNICAL SPECIFICATIONS & THRESHOLDS", section_style))
    spec_data = [
        [Paragraph("Parameter", hdr_style_l), Paragraph("Value", hdr_style)],
        [Paragraph("Complex Name", body_style_l), Paragraph(complex_name, body_style)],
        [Paragraph("Equipment Name", body_style_l), Paragraph(equipment_name, body_style)],
        [Paragraph("Analysis Title / Parameter", body_style_l), Paragraph(analysis_title, body_style)],
        [Paragraph("Degradation Mode", body_style_l), Paragraph(trend_direction, body_style)],
        [Paragraph("Measurement Unit", body_style_l), Paragraph(param_unit if param_unit else "N/A", body_style)],
        [Paragraph("Alert Threshold", body_style_l), Paragraph(f"{ALERT_THRESHOLD:.3f} {param_unit}".strip(), body_style)],
        [Paragraph("Danger Threshold", body_style_l), Paragraph(f"{DANGER_THRESHOLD:.3f} {param_unit}".strip(), body_style)],
        [Paragraph("Confidence Level", body_style_l), Paragraph(f"{CONFIDENCE_PCT:.1f} %", body_style)]
    ]
    t_spec = Table(spec_data, colWidths=[300, 200])
    t_spec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_COLOR]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_spec)
    story.append(Spacer(1, 8))

    # SECTION 2: MODEL COMPARISON
    story.append(Paragraph("MODEL COMPARISON & FIT METRICS", section_style))
    comp_headers = [Paragraph("Model Name", hdr_style_l), Paragraph("R² Score", hdr_style), Paragraph("Residual Std", hdr_style), Paragraph("Status", hdr_style)]
    comp_table_data = [comp_headers]
    for row in model_comparison_data:
        comp_table_data.append([
            Paragraph(row["Model Name"], body_style_l),
            Paragraph(row["R² Score"], body_style),
            Paragraph(row["Residual Std"], body_style),
            Paragraph(row["Status"], body_style)
        ])

    t_comp = Table(comp_table_data, colWidths=[130, 100, 120, 150])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_COLOR]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    # SECTION 3: PROGNOSTIC SUMMARY
    story.append(Paragraph("PROGNOSTIC BREACH PROJECTION SUMMARY", section_style))
    prog_headers = [
        Paragraph("Threshold Level", hdr_style_l), Paragraph("Threshold Value", hdr_style),
        Paragraph("Earliest Date", hdr_style), Paragraph("Expected Date", hdr_style), Paragraph("Latest Date", hdr_style), Paragraph("RUL (Days)", hdr_style)
    ]
    prog_table_data = [prog_headers]
    for row in prognosis_data:
        prog_table_data.append([
            Paragraph(row["Threshold Level"], body_style_l), Paragraph(row["Threshold Value"], body_style),
            Paragraph(row["Earliest Date"], body_style), Paragraph(row["Expected Date"], body_style),
            Paragraph(row["Latest Date"], body_style), Paragraph(row["RUL (Days)"], body_style)
        ])

    t_prog = Table(prog_table_data, colWidths=[90, 85, 80, 80, 85, 80])
    t_prog.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_COLOR]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_prog)

    story.append(PageBreak())

    # SECTION 4: TREND CHART
    story.append(Paragraph("PROGNOSTIC TREND VISUALISATION", section_style))
    story.append(Spacer(1, 10))
    story.append(RLImage(plot_img_path, width=500, height=250))

    doc.build(story)

generate_pdf_report(pdf_file_path)

# Archive ZIP Package
zip_base_name = os.path.join(os.getcwd(), "Prognosis_Output_Files_Package")
zip_archive_path = shutil.make_archive(zip_base_name, 'zip', OUTPUT_DIR)

with open(pdf_file_path, "rb") as f:
    pdf_bytes = f.read()

with open(zip_archive_path, "rb") as f:
    zip_bytes = f.read()

st.subheader("📥 Download Prognosis Reports")
dcol1, dcol2 = st.columns(2)
dcol1.download_button(
    label="📄 Download PDF Report",
    data=pdf_bytes,
    file_name=os.path.basename(pdf_file_path),
    mime="application/pdf"
)
dcol2.download_button(
    label="📦 Download Complete Package (ZIP)",
    data=zip_bytes,
    file_name=f"{complex_name}_{equipment_name}_Prognosis_Files.zip",
    mime="application/zip"
)
