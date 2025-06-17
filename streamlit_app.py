import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

# --- Helper Functions (Define before use) ---
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
        if "suction" in lower and "pressure" in lower:
            mapping['suctionPressures'].append(i)
        elif "discharge" in lower and "pressure" in lower:
            mapping['dischargePressures'].append(i)
        elif "suction" in lower and "temp" in lower:
            mapping['suctionTemps'].append(i)
        elif ("supply" in lower or "discharge" in lower) and "temp" in lower:
            mapping['supplyAirTemps'].append(i)
        elif "date" in lower and mapping['date'] is None:
            mapping['date'] = i
    return mapping

def format_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT

def analyze_hvac_data(data, headers):
    issues = []
    
    # HVAC-specific analysis based on actual data patterns
    for colIdx, header in enumerate(headers):
        col_data = pd.to_numeric(data.iloc[:, colIdx], errors='coerce').dropna()
        if len(col_data) == 0:
            continue
            
        header_lower = header.lower()
        
        # Suction Pressure Analysis
        if "suction" in header_lower and "pressure" in header_lower:
            if col_data.mean() < 60:  # Low suction pressure (typical range 60-80 psi for R-410A)
                issues.append({
                    "severity": "high",
                    "message": f"Low suction pressure detected in {header}",
                    "explanation": "Low suction pressure typically indicates refrigerant undercharge, restriction in liquid line, or evaporator issues.",
                    "suggestions": ["Check for refrigerant leaks", "Verify proper refrigerant charge", "Inspect liquid line for restrictions", "Check evaporator coil condition"]
                })
            elif col_data.mean() > 90:  # High suction pressure
                issues.append({
                    "severity": "medium",
                    "message": f"High suction pressure detected in {header}",
                    "explanation": "High suction pressure may indicate overcharge, compressor issues, or excessive heat load.",
                    "suggestions": ["Check refrigerant charge level", "Inspect compressor operation", "Verify cooling load calculations", "Check for non-condensables"]
                })
        
        # Discharge Pressure Analysis  
        elif "discharge" in header_lower and "pressure" in header_lower:
            if col_data.mean() > 400:  # High discharge pressure (varies by refrigerant)
                issues.append({
                    "severity": "high", 
                    "message": f"High discharge pressure detected in {header}",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coil", "Check condenser fan operation", "Verify proper airflow", "Check for overcharge"]
                })
            elif col_data.mean() < 200:  # Low discharge pressure
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure detected in {header}",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"]
                })
        
        # Temperature Analysis
        elif "temp" in header_lower:
            temp_range = col_data.max() - col_data.min()
            if "suction" in header_lower:
                if col_data.mean() > 60:  # High suction temp
                    issues.append({
                        "severity": "medium",
                        "message": f"High suction temperature in {header}",
                        "explanation": "High suction temperature indicates low refrigerant charge or expansion valve problems.",
                        "suggestions": ["Check superheat settings", "Verify refrigerant charge", "Inspect expansion valve", "Check for restrictions"]
                    })
                elif col_data.mean() < 32:  # Risk of freezing
                    issues.append({
                        "severity": "high",
                        "message": f"Low suction temperature risk in {header}",
                        "explanation": "Very low suction temperature risks liquid refrigerant returning to compressor.",
                        "suggestions": ["Check superheat immediately", "Verify proper airflow", "Inspect expansion valve", "Check for flooding"]
                    })
            elif "supply" in header_lower or "discharge" in header_lower:
                if col_data.mean() > 120:  # High discharge temp
                    issues.append({
                        "severity": "high",
                        "message": f"High discharge temperature in {header}",
                        "explanation": "High discharge temperature indicates compressor stress, poor heat rejection, or overcharge.",
                        "suggestions": ["Check condenser operation", "Verify proper airflow", "Check refrigerant charge", "Inspect compressor condition"]
                    })
            
            # Temperature stability analysis
            if temp_range > 20:  # High temperature variation
                issues.append({
                    "severity": "medium", 
                    "message": f"High temperature variation in {header}",
                    "explanation": "Large temperature swings indicate cycling issues, control problems, or system instability.",
                    "suggestions": ["Check thermostat operation", "Verify control settings", "Inspect for short cycling", "Check system sizing"]
                })
        
        # General outlier detection with HVAC context
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
        if len(outliers) > len(col_data) * 0.1:  # More than 10% outliers
            issues.append({
                "severity": "medium",
                "message": f"Frequent unusual readings in {header}",
                "explanation": "Multiple abnormal readings suggest equipment malfunction, sensor drift, or operating condition changes.",
                "suggestions": ["Calibrate sensors", "Check equipment operation during outlier periods", "Review maintenance logs", "Monitor for patterns"],
                "outlier_count": len(outliers)
            })
    
    return issues

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
logo_file = st.sidebar.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"])

# --- Display Logo and Title ---
if logo_file:
    st.image(logo_file, width=200)

# Title and project input
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)
page_title = st.sidebar.text_input("Webpage Title", "Air Carolinas Data Analysis")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and process the uploaded file
        content = uploaded_file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        headers = df.columns.tolist()
        mapping = parse_headers(headers)
        issues = analyze_hvac_data(df, headers)

        # --- Main App Logic ---
        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        st.subheader("HVAC Data Analysis")
        if issues:
            for issue in issues:
                if issue['severity'] == 'high':
                    st.error(f"🔴 **{issue['message']}**")
                elif issue['severity'] == 'medium':
                    st.warning(f"🟡 **{issue['message']}**")
                else:
                    st.info(f"🔵 **{issue['message']}**")
                
                st.markdown(f"**Why this matters:** {issue['explanation']}")
                st.markdown("**Recommended actions:**")
                for s in issue['suggestions']:
                    st.markdown(f"• {s}")
                if "outlier_count" in issue:
                    st.markdown(f"**Affected readings:** {issue['outlier_count']}")
                st.markdown("---")
        else:
            st.success("✅ No immediate HVAC issues detected in the data analysis.")

        # Time-series plot
        if mapping['date'] is not None:
            df['__date__'] = df.iloc[:, mapping['date']].apply(format_date)
            df = df[df['__date__'].notna()]
            st.subheader("Time-Series Plot")
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot pressures on secondary y-axis
            for idx in mapping['suctionPressures']:
                ax2.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=headers[idx], color='blue', linestyle='-')
            for idx in mapping['dischargePressures']:
                ax2.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=headers[idx], color='navy', linestyle='-')
            
            # Plot temperatures on primary y-axis
            for idx in mapping['suctionTemps']:
                ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=headers[idx], color='red', linestyle='--')
            for idx in mapping['supplyAirTemps']:
                ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=headers[idx], color='orange', linestyle='--')
            
            ax1.set_ylabel("Temperature (°F)", color='red')
            ax2.set_ylabel("Pressure (PSI)", color='blue')
            ax1.set_xlabel("Date")
            ax1.tick_params(axis='y', labelcolor='red')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # Format x-axis dates
            fig.autofmt_xdate(rotation=45)
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            st.pyplot(fig)

        # Download report
        report_lines = [f"HVAC Diagnostic Report - {project_title}", "="*50, "", "DATA ANALYSIS FINDINGS:", ""]
        
        if issues:
            for issue in issues:
                report_lines.extend([
                    f"SEVERITY: {issue['severity'].upper()}",
                    f"ISSUE: {issue['message']}",
                    f"EXPLANATION: {issue['explanation']}",
                    f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                ])
                if "outlier_count" in issue:
                    report_lines.append(f"AFFECTED READINGS: {issue['outlier_count']}")
                report_lines.extend(["", "-"*40, ""])
        else:
            report_lines.append("No immediate HVAC issues detected in data analysis.")
        
        report = "\n".join(report_lines)
        st.download_button("Download Diagnostics Report", report, 
                          file_name=f"hvac_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted and contains valid data.")

else:
    st.info("👆 Please upload a CSV file to begin HVAC data analysis")

# --- HVAC Diagnostic Reference (Always visible) ---
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
