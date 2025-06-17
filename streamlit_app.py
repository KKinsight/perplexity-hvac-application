import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
logo_file = st.sidebar.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"])
# Title and project input
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)

# --- Display Logo and Title ---
if logo_file:
    st.image(logo_file, width=200)
st.title(page_title)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# --- Helper Functions ---
def parse_headers(headers):
    mapping = {
        'suctionPressures': [],
        'dischargePressures': [],
        'suctionTemps': [],
        'supplyAirTemps': [],
        'date': None
    }
    for i, h in enumerate(headers):
        lower = h.lower()
        if "suction pressure" in lower:
            mapping['suctionPressures'].append(i)
        if "discharge pressure" in lower:
            mapping['dischargePressures'].append(i)
        if "suction temp" in lower:
            mapping['suctionTemps'].append(i)
        if "supply air temp" in lower:
            mapping['supplyAirTemps'].append(i)
        if "date" in lower and mapping['date'] is None:
            mapping['date'] = i
    return mapping

def format_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT

def analyze_data(data, headers):
    issues = []
    # Missing values
    for idx, row in data.iterrows():
        for colIdx, cell in enumerate(row):
            if pd.isna(cell) or cell == "":
                issues.append({
                    "severity": "high",
                    "message": f"Missing value in row {idx + 2}, column \"{headers[colIdx]}\"",
                    "explanation": "Missing data can lead to incorrect analysis and may indicate sensor or logging issues.",
                    "suggestions": ["Check sensor connections.", "Ensure data logger is functioning.", "Manually review and fill missing entries."]
                })
    # Mixed data types
    for colIdx, header in enumerate(headers):
        col_data = data.iloc[:, colIdx]
        num_count = pd.to_numeric(col_data, errors='coerce').notna().sum()
        if 0 < num_count < len(col_data):
            issues.append({
                "severity": "medium",
                "message": f"Mixed data types in column \"{header}\"",
                "explanation": "Columns should contain consistent data types for accurate analysis.",
                "suggestions": ["Standardize data entry.", "Remove or correct non-numeric entries.", "Validate sensor outputs."]
            })
    # Duplicate rows
    duplicates = data.duplicated()
    for idx, is_dup in enumerate(duplicates):
        if is_dup:
            issues.append({
                "severity": "low",
                "message": f"Duplicate row at line {idx + 2}",
                "explanation": "Duplicate records can skew results and should be removed.",
                "suggestions": ["Remove duplicate rows.", "Check for repeated data uploads."]
            })
    # Statistical outliers summarized
    for colIdx, header in enumerate(headers):
        col_data = pd.to_numeric(data.iloc[:, colIdx], errors='coerce')
        nums = col_data.dropna()
        if len(nums) > 0:
            q1 = np.percentile(nums, 25)
            q3 = np.percentile(nums, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = nums[(nums < lower) | (nums > upper)]
            if not outliers.empty:
                issues.append({
                    "severity": "medium",
                    "message": f"Statistical outliers detected in \"{header}\"",
                    "explanation": "Outliers may indicate faulty sensors or abnormal operating conditions.",
                    "suggestions": ["Inspect sensor calibration.", "Review abnormal events.", "Filter outliers for trend analysis."],
                    "outlier_count": len(outliers),
                    "outlier_range": (outliers.min(), outliers.max())
                })
    return issues

# --- Main App Logic ---
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    headers = df.columns.tolist()
    mapping = parse_headers(headers)
    issues = analyze_data(df, headers)

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    st.subheader("Diagnostics")
    for issue in issues:
        st.markdown(f"**Severity:** {issue['severity'].capitalize()}")
        st.markdown(f"**Issue:** {issue['message']}")
        st.markdown(f"**Explanation:** {issue['explanation']}")
        st.markdown("**Suggestions:**")
        for s in issue['suggestions']:
            st.markdown(f"- {s}")
        if "outlier_count" in issue:
            st.markdown(f"**Outlier Count:** {issue['outlier_count']}")
            st.markdown(f"**Outlier Range:** {issue['outlier_range'][0]} to {issue['outlier_range'][1]}")
        st.markdown("---")

    # Time-series plot
    if mapping['date'] is not None:
        df['__date__'] = df.iloc[:, mapping['date']].apply(format_date)
        df = df[df['__date__'].notna()]
        st.subheader("Time-Series Plot")
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for idx in mapping['suctionPressures']:
            ax2.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), label=headers[idx], color='blue')
        for idx in mapping['dischargePressures']:
            ax2.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), label=headers[idx], color='navy')
        for idx in mapping['suctionTemps']:
            ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), label=headers[idx], color='red')
        for idx in mapping['supplyAirTemps']:
            ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), label=headers[idx], color='orange')
        ax1.set_ylabel("Temperature")
        ax2.set_ylabel("Pressure")
        ax1.set_xlabel("Date")
        fig.autofmt_xdate(rotation=45)
        fig.legend(loc="upper right")
        st.pyplot(fig)

    # Download report
    report = f"Project: {page_title}\n\nDiagnostics:\n\n"
    for issue in issues:
        report += f"Severity: {issue['severity']}\n"
        report += f"Issue: {issue['message']}\n"
        report += f"Explanation: {issue['explanation']}\n"
        report += f"Suggestions: {'; '.join(issue['suggestions'])}\n"
        if "outlier_count" in issue:
            report += f"Outlier Count: {issue['outlier_count']}\n"
            report += f"Outlier Range: {issue['outlier_range'][0]} to {issue['outlier_range'][1]}\n"
        report += "\n"
    st.download_button("Download Diagnostics Report", report, file_name="diagnostics_report.txt")

    st.subheader("HVAC Diagnostic Possibilities")
    st.markdown("""
- **Low Suction Temperature**: Check refrigerant charge, compressor, and expansion valve.
- **Dirty or Clogged Filters**: Inspect and replace air filters regularly.
- **Malfunctioning Thermostat**: Test thermostat accuracy and wiring.
- **Inadequate Airflow**: Check fans, ducts, and vents for obstructions.
- **Uneven Heating or Cooling**: Balance airflow, inspect dampers and zoning.
- **Unit Not Turning On**: Check power supply, controls, and safety switches.
- **Blown Fuses/Tripped Breakers**: Inspect electrical panel and wiring.
""")
