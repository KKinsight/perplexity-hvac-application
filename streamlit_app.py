import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
logo_file = st.sidebar.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"])

# --- Display Logo and Title ---
if logo_file:
    st.image(logo_file, width=200)

# Title and project input
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)

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
    report = f"Project: {project_title}\n\nDiagnostics:\n\n"
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

    st.subheader("Comprehensive HVAC Diagnostic Reference")
    st.markdown("### 🔧 **Refrigeration System Issues**")
    
    with st.expander("Low Refrigerant Charge"):
        st.markdown("""
        **Symptoms:** Low suction pressure, high superheat, poor cooling capacity
        **Causes:** Refrigerant leaks, improper charging, system evacuation issues
        **Diagnostic Steps:**
        - Check superheat and subcooling values
        - Perform leak detection with electronic detector or soap bubbles
        - Verify refrigerant type and proper charging procedures
        **Solutions:** Locate and repair leaks, properly evacuate and recharge system
        """)
    
    with st.expander("Overcharged Refrigerant"):
        st.markdown("""
        **Symptoms:** High discharge pressure, low superheat, liquid slugging
        **Causes:** Excessive refrigerant added, moisture in system, non-condensables
        **Diagnostic Steps:**
        - Check subcooling values (typically too low)
        - Monitor compressor amp draw
        - Check for liquid refrigerant at compressor suction
        **Solutions:** Recover excess refrigerant, check for moisture and non-condensables
        """)
    
    with st.expander("Compressor Failure"):
        st.markdown("""
        **Symptoms:** No cooling, unusual noises, high amp draw, tripped breakers
        **Causes:** Electrical issues, mechanical wear, liquid slugging, overheating
        **Diagnostic Steps:**
        - Check compressor windings with ohmmeter
        - Measure amp draw and compare to nameplate
        - Check oil condition and level
        - Test starting components (contactors, capacitors)
        **Solutions:** Replace compressor, address root cause, check system cleanliness
        """)

    st.markdown("### 🌡️ **Temperature Control Problems**")
    
    with st.expander("Inconsistent Temperature Control"):
        st.markdown("""
        **Symptoms:** Temperature swings, short cycling, uneven cooling/heating
        **Causes:** Faulty thermostat, improper sensor placement, control calibration
        **Diagnostic Steps:**
        - Calibrate thermostat with accurate thermometer
        - Check sensor location and wiring
        - Verify control settings and deadband
        **Solutions:** Replace or recalibrate thermostat, relocate sensors, adjust controls
        """)
    
    with st.expander("Frozen Evaporator Coil"):
        st.markdown("""
        **Symptoms:** Reduced airflow, ice buildup, poor cooling performance
        **Causes:** Dirty filters, low refrigerant, blocked airflow, defrost issues
        **Diagnostic Steps:**
        - Check air filter condition
        - Measure airflow across coil
        - Check refrigerant pressures and superheat
        - Inspect defrost system operation
        **Solutions:** Replace filters, clean coil, repair refrigerant leaks, fix defrost system
        """)

    st.markdown("### 💨 **Airflow and Ventilation Issues**")
    
    with st.expander("Inadequate Airflow"):
        st.markdown("""
        **Symptoms:** Poor cooling/heating, high energy consumption, comfort complaints
        **Causes:** Dirty filters, blocked ducts, fan problems, undersized ductwork
        **Diagnostic Steps:**
        - Measure static pressure across system
        - Check filter condition and size
        - Inspect ductwork for obstructions or damage
        - Verify fan operation and belt condition
        **Solutions:** Clean/replace filters, clear obstructions, repair ducts, adjust fan speed
        """)
    
    with st.expander("Dirty Air Filters"):
        st.markdown("""
        **Symptoms:** Reduced airflow, increased energy usage, poor indoor air quality
        **Causes:** Lack of maintenance, wrong filter type, excessive contaminants
        **Diagnostic Steps:**
        - Visual inspection of filter condition
        - Measure pressure drop across filter
        - Check filter size and MERV rating
        **Solutions:** Establish regular filter replacement schedule, use appropriate filter type
        """)
    
    with st.expander("Ductwork Problems"):
        st.markdown("""
        **Symptoms:** Uneven temperatures, high energy bills, excessive noise
        **Causes:** Leaky ducts, poor insulation, improper sizing, crushed ducts
        **Diagnostic Steps:**
        - Perform duct blaster test for leakage
        - Check insulation condition and R-value
        - Measure airflow at registers
        - Visual inspection for damage
        **Solutions:** Seal duct leaks, add insulation, resize ducts, repair damage
        """)

    st.markdown("### ⚡ **Electrical System Faults**")
    
    with st.expander("Electrical Control Failures"):
        st.markdown("""
        **Symptoms:** Unit won't start, intermittent operation, blown fuses
        **Causes:** Faulty contactors, bad capacitors, loose connections, control board issues
        **Diagnostic Steps:**
        - Check voltage at all connection points
        - Test contactors and relays
        - Measure capacitor values
        - Inspect control board for damage
        **Solutions:** Replace faulty components, tighten connections, update control boards
        """)
    
    with st.expander("Motor Problems"):
        st.markdown("""
        **Symptoms:** No fan operation, unusual noises, high amp draw, overheating
        **Causes:** Bearing wear, winding failure, capacitor problems, mechanical binding
        **Diagnostic Steps:**
        - Check motor amp draw vs nameplate
        - Test motor windings for continuity
        - Check capacitor values
        - Inspect for mechanical obstructions
        **Solutions:** Replace motor, repair capacitors, lubricate bearings, clear obstructions
        """)

    st.markdown("### 🏠 **System Design and Installation Issues**")
    
    with st.expander("Undersized Equipment"):
        st.markdown("""
        **Symptoms:** Cannot maintain setpoint, continuous operation, high energy bills
        **Causes:** Incorrect load calculations, building modifications, extreme weather
        **Diagnostic Steps:**
        - Perform proper load calculation (Manual J)
        - Monitor runtime and temperature differential
        - Check equipment capacity vs actual load
        **Solutions:** Upgrade equipment size, improve building envelope, add supplemental units
        """)
    
    with st.expander("Oversized Equipment"):
        st.markdown("""
        **Symptoms:** Short cycling, poor humidity control, temperature swings
        **Causes:** Incorrect sizing, overly conservative calculations
        **Diagnostic Steps:**
        - Monitor cycle times and frequency
        - Check actual vs design loads
        - Measure humidity levels
        **Solutions:** Install variable capacity equipment, add staging controls, resize system
        """)
    
    with st.expander("Poor System Balance"):
        st.markdown("""
        **Symptoms:** Hot/cold spots, varying airflow between rooms
        **Causes:** Improper damper settings, poor duct design, missing balancing
        **Diagnostic Steps:**
        - Measure airflow at each register
        - Check damper positions
        - Verify duct sizing calculations
        **Solutions:** Balance airflow, adjust dampers, install balancing dampers
        """)

    st.markdown("### 🧼 **Maintenance-Related Problems**")
    
    with st.expander("Dirty Condenser Coil"):
        st.markdown("""
        **Symptoms:** High discharge pressure, reduced cooling capacity, high energy usage
        **Causes:** Lack of maintenance, environmental contamination, poor location
        **Diagnostic Steps:**
        - Visual inspection of coil condition
        - Check discharge pressure vs ambient temperature
        - Measure amp draw of condensing unit
        **Solutions:** Clean coil with appropriate methods, establish maintenance schedule
        """)
    
    with st.expander("Dirty Evaporator Coil"):
        st.markdown("""
        **Symptoms:** Reduced airflow, ice formation, poor heat transfer
        **Causes:** Poor filtration, lack of maintenance, biological growth
        **Diagnostic Steps:**
        - Visual inspection through access panels
        - Check temperature split across coil
        - Measure airflow and static pressure
        **Solutions:** Clean coil professionally, improve filtration, treat for biologicals
        """)
    
    with st.expander("Clogged Condensate Drain"):
        st.markdown("""
        **Symptoms:** Water leaks, high humidity, musty odors, water damage
        **Causes:** Algae growth, debris buildup, improper slope, frozen traps
        **Diagnostic Steps:**
        - Check drain pan for standing water
        - Test drain flow with water
        - Inspect trap and drain line
        **Solutions:** Clear blockages, install drain cleaners, improve drain slope
        """)

    st.markdown("### 🔄 **Advanced System Issues**")
    
    with st.expander("Heat Recovery Problems"):
        st.markdown("""
        **Symptoms:** Inefficient operation, bypass issues, contamination
        **Causes:** Heat exchanger fouling, damper problems, control failures
        **Diagnostic Steps:**
        - Check heat exchanger effectiveness
        - Verify damper operation
        - Monitor temperature differentials
        **Solutions:** Clean heat exchangers, repair dampers, calibrate controls
        """)
    
    with st.expander("Variable Frequency Drive (VFD) Issues"):
        st.markdown("""
        **Symptoms:** Erratic fan speeds, motor overheating, harmonic distortion
        **Causes:** Parameter settings, electrical interference, heat buildup
        **Diagnostic Steps:**
        - Check VFD parameters and settings
        - Monitor input/output voltages
        - Check for electrical noise
        **Solutions:** Reprogram VFD, add line reactors, improve ventilation
        """)
    
    with st.expander("Building Automation System (BAS) Problems"):
        st.markdown("""
        **Symptoms:** Poor system coordination, inefficient operation, control conflicts
        **Causes:** Programming errors, communication failures, sensor drift
        **Diagnostic Steps:**
        - Review control sequences
        - Check communication networks
        - Calibrate sensors and actuators
        **Solutions:** Update programming, repair networks, replace faulty components
        """)
    
    with st.expander("Zoning System Malfunctions"):
        st.markdown("""
        **Symptoms:** Uneven temperatures between zones, damper problems
        **Causes:** Faulty zone dampers, control panel issues, sensor problems
        **Diagnostic Steps:**
        - Test zone damper operation
        - Check zone control panel
        - Verify temperature sensors
        **Solutions:** Replace dampers, repair control panels, calibrate sensors
        """)
    
    with st.expander("Indoor Air Quality Issues"):
        st.markdown("""
        **Symptoms:** Occupant complaints, odors, health issues, poor ventilation
        **Causes:** Inadequate ventilation, contamination sources, filtration problems
        **Diagnostic Steps:**
        - Measure ventilation rates
        - Test for common contaminants
        - Check filtration effectiveness
        **Solutions:** Increase ventilation, eliminate sources, upgrade filtration
        """)
    
    with st.expander("Refrigerant Line Issues"):
        st.markdown("""
        **Symptoms:** Pressure drops, oil logging, capacity loss
        **Causes:** Improper sizing, installation errors, insulation problems
        **Diagnostic Steps:**
        - Check line sizing calculations
        - Inspect insulation condition
        - Monitor pressure drops
        **Solutions:** Resize lines, repair insulation, add oil separators
        """)

    st.markdown("---")
    st.markdown("**💡 Pro Tip:** Always start diagnostics with basic checks (power, filters, settings) before moving to complex system analysis. Document all findings and maintain detailed service records for trend analysis.")

else:
    st.info("Please upload a CSV file to begin HVAC diagnostics analysis.")
    st.markdown("### How to Use This Tool")
    st.markdown("""
    1. **Upload Logo** (optional): Add your company logo in the sidebar
    2. **Enter Project Title**: Customize the report title
    3. **Upload CSV Data**: Include columns for:
       - Date/timestamp
       - Suction pressures
       - Discharge pressures  
       - Suction temperatures
       - Supply air temperatures
    4. **Review Analysis**: Check for data quality issues and outliers
    5. **Reference Diagnostics**: Use the comprehensive problem guide below
    6. **Download Report**: Generate a text report of findings
    """)
