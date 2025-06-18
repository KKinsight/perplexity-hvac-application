import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime
import base64

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
                    "suggestions": ["Check for refrigerant leaks", "Verify proper refrigerant charge", "Inspect liquid line for restrictions", "Check evaporator coil condition"],
                    "issue_type": "refrigerant_system"
                })
            elif col_data.mean() > 90:  # High suction pressure
                issues.append({
                    "severity": "medium",
                    "message": f"High suction pressure detected in {header}",
                    "explanation": "High suction pressure may indicate overcharge, compressor issues, or excessive heat load.",
                    "suggestions": ["Check refrigerant charge level", "Inspect compressor operation", "Verify cooling load calculations", "Check for non-condensables"],
                    "issue_type": "refrigerant_system"
                })
        
        # Discharge Pressure Analysis  
        elif "discharge" in header_lower and "pressure" in header_lower:
            if col_data.mean() > 400:  # High discharge pressure (varies by refrigerant)
                issues.append({
                    "severity": "high", 
                    "message": f"High discharge pressure detected in {header}",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coil", "Check condenser fan operation", "Verify proper airflow", "Check for overcharge"],
                    "issue_type": "condenser_system"
                })
            elif col_data.mean() < 200:  # Low discharge pressure
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure detected in {header}",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"],
                    "issue_type": "compressor_system"
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
                        "suggestions": ["Check superheat settings", "Verify refrigerant charge", "Inspect expansion valve", "Check for restrictions"],
                        "issue_type": "refrigerant_system"
                    })
                elif col_data.mean() < 32:  # Risk of freezing
                    issues.append({
                        "severity": "high",
                        "message": f"Low suction temperature risk in {header}",
                        "explanation": "Very low suction temperature risks liquid refrigerant returning to compressor.",
                        "suggestions": ["Check superheat immediately", "Verify proper airflow", "Inspect expansion valve", "Check for flooding"],
                        "issue_type": "refrigerant_system"
                    })
            elif "supply" in header_lower or "discharge" in header_lower:
                if col_data.mean() > 120:  # High discharge temp
                    issues.append({
                        "severity": "high",
                        "message": f"High discharge temperature in {header}",
                        "explanation": "High discharge temperature indicates compressor stress, poor heat rejection, or overcharge.",
                        "suggestions": ["Check condenser operation", "Verify proper airflow", "Check refrigerant charge", "Inspect compressor condition"],
                        "issue_type": "compressor_system"
                    })
            
            # Temperature stability analysis
            if temp_range > 20:  # High temperature variation
                issues.append({
                    "severity": "medium", 
                    "message": f"High temperature variation in {header}",
                    "explanation": "Large temperature swings indicate cycling issues, control problems, or system instability.",
                    "suggestions": ["Check thermostat operation", "Verify control settings", "Inspect for short cycling", "Check system sizing"],
                    "issue_type": "control_system"
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
                "outlier_count": len(outliers),
                "issue_type": "sensor_system"
            })
    
    return issues

def generate_diagnostic_reference(detected_issues):
    """Generate diagnostic reference based on detected issues"""
    issue_types = set()
    for issue in detected_issues:
        issue_types.add(issue.get('issue_type', 'general'))
    
    reference_content = {}
    
    if 'refrigerant_system' in issue_types:
        reference_content['Refrigerant System Issues'] = {
            'Low Refrigerant Charge': {
                'symptoms': 'Low suction pressure, high superheat, poor cooling capacity',
                'causes': 'Refrigerant leaks, improper charging, system evacuation issues',
                'diagnostics': ['Check superheat and subcooling values', 'Perform leak detection', 'Verify refrigerant type and charging procedures'],
                'solutions': ['Locate and repair leaks', 'Properly evacuate and recharge system']
            },
            'High Suction Temperature': {
                'symptoms': 'Elevated suction line temperature, poor cooling performance',
                'causes': 'Low refrigerant charge, expansion valve problems, restrictions',
                'diagnostics': ['Measure superheat at evaporator outlet', 'Check expansion valve operation', 'Inspect for line restrictions'],
                'solutions': ['Adjust superheat settings', 'Replace expansion valve', 'Clear restrictions']
            }
        }
    
    if 'compressor_system' in issue_types:
        reference_content['Compressor System Issues'] = {
            'Compressor Performance Problems': {
                'symptoms': 'Unusual pressures, high discharge temperature, poor efficiency',
                'causes': 'Mechanical wear, electrical issues, refrigerant problems',
                'diagnostics': ['Check compressor amp draw', 'Test valve operation', 'Monitor discharge temperature'],
                'solutions': ['Replace worn components', 'Repair electrical connections', 'Address refrigerant issues']
            },
            'High Discharge Temperature': {
                'symptoms': 'Compressor overheating, reduced efficiency, potential failure',
                'causes': 'Poor heat rejection, overcharge, compressor wear',
                'diagnostics': ['Check condenser operation', 'Verify refrigerant charge', 'Test compressor condition'],
                'solutions': ['Clean condenser', 'Adjust refrigerant charge', 'Replace compressor if needed']
            }
        }
    
    if 'condenser_system' in issue_types:
        reference_content['Condenser System Issues'] = {
            'High Discharge Pressure': {
                'symptoms': 'Excessive head pressure, poor cooling, high energy consumption',
                'causes': 'Dirty condenser coil, fan problems, airflow restrictions',
                'diagnostics': ['Check condenser coil condition', 'Test fan operation', 'Measure airflow'],
                'solutions': ['Clean condenser coil', 'Repair fan motor', 'Clear airflow restrictions']
            }
        }
    
    if 'control_system' in issue_types:
        reference_content['Control System Issues'] = {
            'Temperature Control Problems': {
                'symptoms': 'Temperature swings, short cycling, poor comfort',
                'causes': 'Faulty thermostat, sensor issues, control calibration',
                'diagnostics': ['Check thermostat calibration', 'Test sensor accuracy', 'Verify control settings'],
                'solutions': ['Replace thermostat', 'Calibrate sensors', 'Adjust control parameters']
            }
        }
    
    if 'sensor_system' in issue_types:
        reference_content['Sensor and Monitoring Issues'] = {
            'Sensor Drift and Calibration': {
                'symptoms': 'Inconsistent readings, frequent outliers, poor system response',
                'causes': 'Sensor aging, environmental factors, calibration drift',
                'diagnostics': ['Compare readings with calibrated instruments', 'Check sensor wiring', 'Review historical data'],
                'solutions': ['Recalibrate sensors', 'Replace aged sensors', 'Improve sensor protection']
            }
        }
    
    return reference_content

def get_base64_image(uploaded_file):
    """Convert uploaded image to base64 string"""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        base64_string = base64.b64encode(bytes_data).decode()
        return base64_string
    return None

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

        st.subheader("HVAC Diagnostic Analysis")
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

        # Time-series plot with improved suction temperature handling
        if mapping['date'] is not None:
            df['__date__'] = df.iloc[:, mapping['date']].apply(format_date)
            df = df[df['__date__'].notna()]
            st.subheader("Time-Series Analysis")
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot pressures on secondary y-axis
            for idx in mapping['suctionPressures']:
                ax2.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=f"{headers[idx]} (Pressure)", color='blue', linestyle='-', linewidth=2)
            for idx in mapping['dischargePressures']:
                ax2.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=f"{headers[idx]} (Pressure)", color='navy', linestyle='-', linewidth=2)
            
            # Plot temperatures on primary y-axis - ensure suction temps are included
            temp_plotted = False
            for idx in mapping['suctionTemps']:
                ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=f"{headers[idx]} (Temperature)", color='red', linestyle='--', linewidth=2, marker='o', markersize=3)
                temp_plotted = True
            for idx in mapping['supplyAirTemps']:
                ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                        label=f"{headers[idx]} (Temperature)", color='orange', linestyle='--', linewidth=2, marker='s', markersize=3)
                temp_plotted = True
            
            # If no temperature data was found, look for any column with 'temp' in the name
            if not temp_plotted:
                for idx, header in enumerate(headers):
                    if 'temp' in header.lower():
                        ax1.plot(df['__date__'], pd.to_numeric(df.iloc[:, idx], errors='coerce'), 
                                label=f"{header} (Temperature)", color='red', linestyle='--', linewidth=2, marker='o', markersize=3)
                        temp_plotted = True
            
            ax1.set_ylabel("Temperature (°F)", color='red', fontsize=12, fontweight='bold')
            ax2.set_ylabel("Pressure (PSI)", color='blue', fontsize=12, fontweight='bold')
            ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='red')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # Format x-axis dates
            fig.autofmt_xdate(rotation=45)
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines1 or lines2:
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
            
            plt.title(f"HVAC System Performance - {project_title}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        # Generate relevant diagnostic reference based on detected issues
        if issues:
            st.subheader("Relevant Diagnostic Reference")
            st.markdown("*Based on issues detected in your system data*")
            
            diagnostic_ref = generate_diagnostic_reference(issues)
            
            for category, problems in diagnostic_ref.items():
                st.markdown(f"### 🔧 **{category}**")
                
                for problem_name, details in problems.items():
                    with st.expander(problem_name):
                        st.markdown(f"**Symptoms:** {details['symptoms']}")
                        st.markdown(f"**Causes:** {details['causes']}")
                        st.markdown("**Diagnostic Steps:**")
                        for step in details['diagnostics']:
                            st.markdown(f"• {step}")
                        st.markdown("**Solutions:**")
                        for solution in details['solutions']:
                            st.markdown(f"• {solution}")

        # Enhanced Download report with logo and title
        report_lines = [
            f"{project_title}",
            "="*len(project_title),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if logo_file:
            logo_base64 = get_base64_image(logo_file)
            if logo_base64:
                report_lines.extend([
                    f"Company Logo: [Base64 Encoded - {len(logo_base64)} characters]",
                    ""
                ])
        
        report_lines.extend([
            "HVAC DIAGNOSTIC ANALYSIS REPORT",
            "="*50,
            "",
            "SYSTEM DATA ANALYSIS FINDINGS:",
            ""
        ])
        
        if issues:
            high_issues = [i for i in issues if i['severity'] == 'high']
            medium_issues = [i for i in issues if i['severity'] == 'medium']
            low_issues = [i for i in issues if i['severity'] == 'low']
            
            if high_issues:
                report_lines.extend(["HIGH PRIORITY ISSUES:", "-"*20])
                for issue in high_issues:
                    report_lines.extend([
                        f"ISSUE: {issue['message']}",
                        f"EXPLANATION: {issue['explanation']}",
                        f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                        ""
                    ])
            
            if medium_issues:
                report_lines.extend(["MEDIUM PRIORITY ISSUES:", "-"*22])
                for issue in medium_issues:
                    report_lines.extend([
                        f"ISSUE: {issue['message']}",
                        f"EXPLANATION: {issue['explanation']}",
                        f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                        ""
                    ])
            
            if low_issues:
                report_lines.extend(["LOW PRIORITY ISSUES:", "-"*19])
                for issue in low_issues:
                    report_lines.extend([
                        f"ISSUE: {issue['message']}",
                        f"EXPLANATION: {issue['explanation']}",
                        f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                        ""
                    ])
        else:
            report_lines.append("✅ No immediate HVAC issues detected in data analysis.")
        
        report_lines.extend([
            "",
            "="*50,
            f"Report generated by {project_title} Analysis System",
            f"For technical support, please contact your HVAC service provider."
        ])
        
        report = "\n".join(report_lines)
        st.download_button(
            "📄 Download Comprehensive Diagnostics Report", 
            report, 
            file_name=f"{project_title.replace(' ', '_')}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted and contains valid data.")

else:
    st.info("👆 Please upload a CSV file to begin HVAC data analysis")
    st.markdown("### 📋 **Expected Data Format**")
    st.markdown("""
    Your CSV file should contain columns with names that include:
    - **Date/Time** information (e.g., 'Date', 'Timestamp')
    - **Suction Pressure** data (e.g., 'Suction Pressure', 'Suction PSI')
    - **Discharge Pressure** data (e.g., 'Discharge Pressure', 'Head Pressure')
    - **Temperature** readings (e.g., 'Suction Temp', 'Supply Air Temp', 'Discharge Temp')
    
    The system will automatically detect and analyze these parameters based on column names.
    """)
