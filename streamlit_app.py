import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from datetime import datetime
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import altair as alt

# --- Enhanced Helper Functions ---
def parse_headers_enhanced(headers):
    """Enhanced header parsing to handle multiple CSV formats"""
    mapping = {
        'suctionPressures': [],
        'dischargePressures': [],
        'suctionTemps': [],
        'supplyAirTemps': [],
        'dischargeTemps': [],
        'outdoorAirTemps': [],
        'coolingSetpoints': [],
        'heatingSetpoints': [],
        'relativeHumidity': [],
        'indoorTemps': [],
        'date': None,
        'time': None
    }
    
    for i, h in enumerate(headers):
        h_clean = str(h).strip()
        lower = h_clean.lower()
       
        # Relative Humidity detection
        if any(keyword in lower for keyword in ['rel hum', 'rel. hum', 'relative humidity', 'rh']):
            mapping['relativeHumidity'].append(i)
        # Indoor Temperature detection
        if any(keyword in lower for keyword in ['indoor temp', 'indoor temperature', 'room temp', 'space temp']):
            mapping['indoorTemps'].append(i)
        
        # Date and Time detection
        if any(keyword in lower for keyword in ['date']) and mapping['date'] is None:
            mapping['date'] = i
        elif any(keyword in lower for keyword in ['time']) and mapping['time'] is None:
            mapping['time'] = i
        
        # Enhanced pressure detection
        elif any(keyword in lower for keyword in ['sucpr', 'suc pr', 'suction pr', 'suction_pr']) or \
             (('suc' in lower or 'suction' in lower) and ('pr' in lower or 'pressure' in lower)):
            mapping['suctionPressures'].append(i)
        
        elif any(keyword in lower for keyword in ['dischg', 'dis chg', 'discharge pr', 'head pr', 'headpr']) or \
             (('discharge' in lower or 'head' in lower) and ('pr' in lower or 'pressure' in lower)):
            mapping['dischargePressures'].append(i)
        
        # Enhanced temperature detection
        elif any(keyword in lower for keyword in ['suctmp', 'suc tmp', 'suction tmp', 'suction_tmp', 'suction temp']):
            mapping['suctionTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['sat ', 'supply air', 'supply_air', 'discharge temp']):
            mapping['supplyAirTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['dischg', 'dis chg', 'discharge']) and 'temp' in lower:
            mapping['dischargeTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['oat', 'outdoor', 'outside']) and ('temp' in lower or 'air' in lower):
            mapping['outdoorAirTemps'].append(i)
        
        # Setpoint detection
        elif any(keyword in lower for keyword in ['csp', 'cool', 'cooling']) and ('sp' in lower or 'setpoint' in lower):
            mapping['coolingSetpoints'].append(i)
        
        elif any(keyword in lower for keyword in ['hsp', 'heat', 'heating']) and ('sp' in lower or 'setpoint' in lower):
            mapping['heatingSetpoints'].append(i)

        # Humidity
        elif 'rh' in lower or 'humidity' in lower:
            mapping['relativeHumidity'].append(i)
    
    return mapping

def format_date_enhanced(date_val, time_val=None):
    """Enhanced date formatting function"""
    try:
        if pd.isna(date_val):
            return None
            
        # Handle different date formats
        if isinstance(date_val, str):
            # Try parsing common date formats
            for fmt in ['%d-%b', '%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y']:
                try:
                    if time_val is not None and not pd.isna(time_val):
                        datetime_str = f"{date_val} {time_val}"
                        return pd.to_datetime(datetime_str, format=f"{fmt} %H:%M")
                    else:
                        return pd.to_datetime(date_val, format=fmt)
                except:
                    continue
        
        # If direct conversion works
        if time_val is not None and not pd.isna(time_val):
            return pd.to_datetime(f"{date_val} {time_val}")
        else:
            return pd.to_datetime(date_val)
            
    except:
        return None

def check_comfort_conditions(df, headers, mapping):
    """Check indoor comfort conditions"""
    results = []
    
    # Check relative humidity
    comfort_results = check_comfort_conditions(df, headers, mapping)
    st.markdown("## 🏠 Indoor Comfort Check")
    if comfort_results:
        for result in comfort_results:
            if result["type"] == "Relative Humidity":
                msg = ('✅ Within ideal range (≤60%)' if result['compliant'] 
                       else f'⚠️ {result["percent_over"]:.1f}% of values above 60%')
                st.write(f"**{result['column']}** (Avg: {result['average']:.1f}%) - {msg}")
            elif result["type"] == "Indoor Temperature":
                msg = ('✅ Within ideal range (70–75°F)' if result['compliant'] 
                       else f'⚠️ {result["percent_outside"]:.1f}% of values outside 70–75°F')
                st.write(f"**{result['column']}** (Avg: {result['average']:.1f}°F) - {msg}")
    else:
        st.info("No relative humidity or indoor temperature columns detected in this file.")

    # Check indoor temperature
    for idx in mapping.get('indoorTemps', []):  # Use .get() for safety
        temp_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(temp_data) > 0:
            below_70 = (temp_data < 70).sum()
            above_75 = (temp_data > 75).sum()
            percent_outside = ((below_70 + above_75) / len(temp_data)) * 100
            avg_temp = temp_data.mean()
            results.append({
                'type': 'Indoor Temperature',
                'column': headers[idx],
                'average': avg_temp,
                'percent_outside': percent_outside,
                'compliant': percent_outside == 0
            })
    
    return results

def analyze_hvac_data_enhanced(data, headers, mapping):
    """Enhanced HVAC analysis with improved detection logic"""
    issues = []
    
    # HVAC-specific analysis based on actual data patterns
    for colIdx, header in enumerate(headers):
        col_data = pd.to_numeric(data.iloc[:, colIdx], errors='coerce').dropna()
        if len(col_data) == 0:
            continue
            
        header_lower = str(header).lower()
        
        # Suction Pressure Analysis (Enhanced)
        if colIdx in mapping['suctionPressures']:
            avg_pressure = col_data.mean()
    # Check suction pressures
    for idx in mapping['suctionPressures']:
        col_data = pd.to_numeric(df.iloc[:, i], errors='coerce').dropna()
        if len(col_data) > 0:
            if col_data.mean() > 200:
                issues.append({
                    'message': f'High suction pressure detected in {headers[i]}',
                    'severity': 'high',
                    'explanation': 'High suction pressure may indicate system overcharge or restricted airflow',
                    'suggestions': ['Check refrigerant levels', 'Inspect air filters', 'Verify ductwork']
                })
            elif col_data.mean() < 50:
                issues.append({
                    'message': f'Low suction pressure detected in {headers[idx]}',
                    'severity': 'medium',
                    'explanation': 'Low suction pressure may indicate refrigerant leak or expansion valve issues',
                    'suggestions': ['Check for refrigerant leaks', 'Inspect expansion valve', 'Verify system charge']
                })
        
        # Discharge Pressure Analysis (Enhanced)
        elif colIdx in mapping['dischargePressures']:
            avg_pressure = col_data.mean()
            if avg_pressure > 400:  # High discharge pressure
                issues.append({
                    "severity": "high", 
                    "message": f"High discharge pressure detected in {header} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coil", "Check condenser fan operation", "Verify proper airflow", "Check for overcharge"],
                    "issue_type": "condenser_system"
                })
            elif avg_pressure < 150:  # Low discharge pressure
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure detected in {header} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"],
                    "issue_type": "compressor_system"
                })
        
        # Enhanced Temperature Analysis
        elif colIdx in mapping['suctionTemps']:
            avg_temp = col_data.mean()
            if avg_temp > 65:  # High suction temp
                issues.append({
                    "severity": "medium",
                    "message": f"High suction temperature in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High suction temperature indicates low refrigerant charge or expansion valve problems.",
                    "suggestions": ["Check superheat settings", "Verify refrigerant charge", "Inspect expansion valve", "Check for restrictions"],
                    "issue_type": "refrigerant_system"
                })
            elif avg_temp < 35:  # Risk of freezing
                issues.append({
                    "severity": "high",
                    "message": f"Low suction temperature risk in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Very low suction temperature risks liquid refrigerant returning to compressor.",
                    "suggestions": ["Check superheat immediately", "Verify proper airflow", "Inspect expansion valve", "Check for flooding"],
                    "issue_type": "refrigerant_system"
                })
        
        elif colIdx in mapping['supplyAirTemps'] or colIdx in mapping['dischargeTemps']:
            avg_temp = col_data.mean()
            if avg_temp > 120:  # High discharge temp
                issues.append({
                    "severity": "high",
                    "message": f"High discharge temperature in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High discharge temperature indicates compressor stress, poor heat rejection, or overcharge.",
                    "suggestions": ["Check condenser operation", "Verify proper airflow", "Check refrigerant charge", "Inspect compressor condition"],
                    "issue_type": "compressor_system"
                })
            elif avg_temp < 50:  # Very low supply air temp
                issues.append({
                    "severity": "medium",
                    "message": f"Very low supply air temperature in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Extremely low supply air temperature may indicate overcooling or control issues.",
                    "suggestions": ["Check thermostat settings", "Verify cooling load", "Inspect damper operation", "Check for overcooling"],
                    "issue_type": "control_system"
                })
        
        # Temperature stability analysis for all temperature readings
        if colIdx in (mapping['suctionTemps'] + mapping['supplyAirTemps'] + mapping['dischargeTemps'] + mapping['outdoorAirTemps']):
            temp_range = col_data.max() - col_data.min()
            if temp_range > 25:  # High temperature variation
                issues.append({
                    "severity": "medium", 
                    "message": f"High temperature variation in {header} (Range: {temp_range:.1f}°F)",
                    "explanation": "Large temperature swings indicate cycling issues, control problems, or system instability.",
                    "suggestions": ["Check thermostat operation", "Verify control settings", "Inspect for short cycling", "Check system sizing"],
                    "issue_type": "control_system"
                })
        
        # General outlier detection with HVAC context
        if len(col_data) > 5:  # Only analyze if we have enough data points
            q1, q3 = np.percentile(col_data, [25, 75])
            iqr = q3 - q1
            if iqr > 0:  # Avoid division by zero
                outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                if len(outliers) > len(col_data) * 0.15:  # More than 15% outliers
                    issues.append({
                        "severity": "medium",
                        "message": f"Frequent unusual readings in {header}",
                        "explanation": "Multiple abnormal readings suggest equipment malfunction, sensor drift, or operating condition changes.",
                        "suggestions": ["Calibrate sensors", "Check equipment operation during outlier periods", "Review maintenance logs", "Monitor for patterns"],
                        "outlier_count": len(outliers),
                        "issue_type": "sensor_system"
                    })

         # Check temperature differentials
        if mapping['suctionTemps'] and mapping['supplyAirTemps']:
            for suc_idx in mapping['suctionTemps']:
                for sat_idx in mapping['supplyAirTemps']:
                    suc_data = pd.to_numeric(df.iloc[:, suc_idx], errors='coerce').dropna()
                    sat_data = pd.to_numeric(df.iloc[:, sat_idx], errors='coerce').dropna()
                    
                    if len(suc_data) > 0 and len(sat_data) > 0:
                        temp_diff = abs(sat_data.mean() - suc_data.mean())
                        if temp_diff < 15:
                            issues.append({
                                'message': f'Low temperature differential between {headers[suc_idx]} and {headers[sat_idx]}',
                                'severity': 'medium',
                                'explanation': 'Low temperature differential may indicate poor heat transfer or airflow issues',
                                'suggestions': ['Check evaporator coil', 'Verify airflow rates', 'Inspect refrigerant levels']
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

def generate_pdf_report(project_title, logo_file, issues, df_summary=None):
    """Generate a comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=12,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=8,
        textColor=colors.darkred
    )
    
    normal_style = styles['Normal']
    normal_style.alignment = TA_JUSTIFY
    
    # Build the PDF content
    story = []
    
    # Add logo if provided
    if logo_file:
        try:
            logo_file.seek(0)  # Reset file pointer
            logo = Image(logo_file, width=2*inch, height=1*inch)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 12))
        except:
            pass
    
    # Title
    story.append(Paragraph(project_title, title_style))
    story.append(Paragraph("HVAC Diagnostic Analysis Report", heading_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    if issues:
        high_count = len([i for i in issues if i['severity'] == 'high'])
        medium_count = len([i for i in issues if i['severity'] == 'medium'])
        low_count = len([i for i in issues if i['severity'] == 'low'])
        
        summary_text = f"""
        This report analyzes HVAC system performance data and identifies {len(issues)} total issues requiring attention:
        <br/>• {high_count} High Priority Issues (require immediate attention)
        <br/>• {medium_count} Medium Priority Issues (should be addressed soon)
        <br/>• {low_count} Low Priority Issues (monitor and plan maintenance)
        """
        story.append(Paragraph(summary_text, normal_style))
    else:
        story.append(Paragraph("System analysis shows no immediate issues detected. All parameters appear to be within normal operating ranges.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Detailed Findings
    story.append(Paragraph("Detailed Findings", heading_style))
    
    if issues:
        # Group issues by severity
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        low_issues = [i for i in issues if i['severity'] == 'low']
        
        # High Priority Issues
        if high_issues:
            story.append(Paragraph("🔴 HIGH PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(high_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                
                if "outlier_count" in issue:
                    story.append(Paragraph(f"<b>Affected Readings:</b> {issue['outlier_count']}", normal_style))
                story.append(Spacer(1, 12))
        
        # Medium Priority Issues
        if medium_issues:
            story.append(Paragraph("🟡 MEDIUM PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(medium_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                
                if "outlier_count" in issue:
                    story.append(Paragraph(f"<b>Affected Readings:</b> {issue['outlier_count']}", normal_style))
                story.append(Spacer(1, 12))
        
        # Low Priority Issues
        if low_issues:
            story.append(Paragraph("🔵 LOW PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(low_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                
                if "outlier_count" in issue:
                    story.append(Paragraph(f"<b>Affected Readings:</b> {issue['outlier_count']}", normal_style))
                story.append(Spacer(1, 12))
    
    # Add data summary if provided
    if df_summary is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Data Summary Statistics", heading_style))
        
        # Create a simple table with basic stats
        try:
            numeric_df = df_summary.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_data = [['Parameter', 'Mean', 'Min', 'Max', 'Std Dev']]
                for col in numeric_df.columns[:10]:  # Limit to first 10 columns
                    stats_data.append([
                        col,
                        f"{numeric_df[col].mean():.2f}",
                        f"{numeric_df[col].min():.2f}",
                        f"{numeric_df[col].max():.2f}",
                        f"{numeric_df[col].std():.2f}"
                    ])
                
                table = Table(stats_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
        except:
            story.append(Paragraph("Data summary statistics could not be generated.", normal_style))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Report Notes", heading_style))
    story.append(Paragraph("""
    This automated diagnostic report is based on pattern analysis of Air Carolinas HVAC system data. 
    All recommendations should be verified by qualified HVAC technicians before implementation. 
    Regular maintenance and professional inspections are essential for optimal system performance.
    """, normal_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by {project_title} Analysis System", normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.set_page_config(page_title="Enhanced HVAC Data Analysis", layout="wide")

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
logo_file = st.sidebar.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"])

# --- Display Logo and Title ---
if logo_file:
    st.image(logo_file, width=200)

# Title and project input
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)

def read_csv_with_encoding(file_obj):
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    for encoding in encodings_to_try:
        try:
            file_obj.seek(0)
            content = file_obj.read().decode(encoding)
            return pd.read_csv(StringIO(content)), content
        except Exception:
            continue
    file_obj.seek(0)
    content = file_obj.read().decode('utf-8', errors='replace')
    return pd.read_csv(StringIO(content)), content

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more CSV or Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

def read_csv_with_encoding(uploaded_file):
    # Try reading with utf-8, fallback to latin1 if needed
    try:
        df = pd.read_csv(uploaded_file)
        content = uploaded_file.getvalue()
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
        content = uploaded_file.getvalue()
    return df, content

if uploaded_files:
    all_dataframes = []
    all_issues = []
    
    # Process each file
    for uploaded_file in uploaded_files:
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(
                    uploaded_file,
                    engine='openpyxl' if file_extension == 'xlsx' else 'xlrd'
                )
                st.success(f"✅ Excel file '{uploaded_file.name}' successfully read")
            else:
                df, content = read_csv_with_encoding(uploaded_file)
                st.success(f"✅ CSV file '{uploaded_file.name}' successfully read")
            
            # Now you can process df as needed:
            headers = list(df.columns)
            mapping = parse_headers_enhanced(headers)
            
            # Example: Show first few rows
            st.write(f"**Preview of {uploaded_file.name}:**")
            st.dataframe(df.head())
            
            # Example: Analyze and collect issues for this file
            issues = analyze_hvac_data_enhanced(df, headers, mapping)
            all_dataframes.append(df)
            all_issues.append(issues)
            
            # Example: Show issues
            if issues:
                st.error(f"Issues detected in {uploaded_file.name}:")
                for issue in issues:
                    st.write(f"- {issue['message']}")
            else:
                st.success(f"No major issues found in {uploaded_file.name}.")
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            
            # Add source file identifier
            df['source_file'] = uploaded_file.name
            all_dataframes.append(df)
            
            # Analyze each file
            headers = df.columns.tolist()
            mapping = parse_headers_enhanced(headers)
            
            # Show detected columns
            st.subheader(f"🔍 Detected Columns in {uploaded_file.name}")
            if mapping['suctionPressures']:
                st.write(f"**Suction Pressures:** {[headers[i] for i in mapping['suctionPressures']]}")
            if mapping['dischargePressures']:
                st.write(f"**Discharge Pressures:** {[headers[i] for i in mapping['dischargePressures']]}")
            if mapping['suctionTemps']:
                st.write(f"**Suction Temps:** {[headers[i] for i in mapping['suctionTemps']]}")
            if mapping['supplyAirTemps']:
                st.write(f"**Supply Air Temps:** {[headers[i] for i in mapping['supplyAirTemps']]}")
            if mapping['outdoorAirTemps']:
                st.write(f"**Outdoor Air Temps:** {[headers[i] for i in mapping['outdoorAirTemps']]}")
            
            # Analyze issues for this file
            issues = analyze_hvac_data_enhanced(df, headers, mapping)
            all_issues.extend(issues)
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        st.subheader("📊 Combined Data Overview")
        st.dataframe(combined_df.head(10))
        
        # Show basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data Points", len(combined_df))
        with col2:
            st.metric("Number of Files", len(all_dataframes))
        with col3:
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Parameters", len(numeric_cols))
        
        # Combined Analysis
        st.subheader("🔧 Combined HVAC Diagnostic Analysis")
        
        if all_issues:
            # Show summary counts
            high_count = len([i for i in all_issues if i['severity'] == 'high'])
            medium_count = len([i for i in all_issues if i['severity'] == 'medium'])
            low_count = len([i for i in all_issues if i['severity'] == 'low'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 High Priority", high_count)
            with col2:
                st.metric("🟡 Medium Priority", medium_count)
            with col3:
                st.metric("🔵 Low Priority", low_count)
            
            # Display all issues
            for issue in all_issues:
                if issue['severity'] == 'high':
                    st.error(f"🔴 **{issue['message']}**")
                elif issue['severity'] == 'medium':
                    st.warning(f"🟡 **{issue['message']}**")
                else:
                    st.info(f"**{issue['message']}**")
                
                st.markdown(f"**Why this matters:** {issue['explanation']}")
                st.markdown("**Recommended actions:**")
                for s in issue['suggestions']:
                    st.markdown(f"• {s}")
                st.markdown("---")
        else:
            st.success("✅ No immediate HVAC issues detected in the combined data analysis.")
        
        # Create visualizations using the combined data
        # Find numeric columns that likely represent HVAC parameters
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        
        # Try to identify key columns by name patterns
        pressure_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['pressure', 'psi', 'sucpr', 'headpr'])]
        temp_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['temp', 'sat', 'oat', 'discharge'])]
        
        if len(pressure_cols) > 0 or len(temp_cols) > 0:
            st.subheader("📈 Combined System Performance Analysis")

            def plot_temperature_vs_time(df, mapping, headers, time_col):
                plt.figure(figsize=(12, 6))
                temp_cols = (
                    mapping['supplyAirTemps']
                    + mapping['indoorTemps']  # assuming 'space temp' is mapped here
                    + mapping.get('spaceDewpoint', [])
                    + mapping['outdoorAirTemps']
                    + mapping.get('outdoorAirDewpoint', [])
                    + mapping['suctionTemps']
                )
                for idx in temp_cols:
                    plt.plot(df[time_col], pd.to_numeric(df.iloc[:, idx], errors='coerce'), label=headers[idx])
                    plt.xlabel('Time')
                    plt.ylabel('Temperature (°F)')
                    plt.title('Temperature vs. Time')
                    plt.legend()
                    plt.tight_layout()
                    st.pyplot(plt)

def plot_pressure_vs_time(df, mapping, headers, time_col):
    plt.figure(figsize=(12, 6))
    pressure_cols = (
        mapping['suctionPressures']
        + mapping['dischargePressures']  # assuming both discharge pressures are mapped here
    )
    for idx in pressure_cols:
        plt.plot(df[time_col], pd.to_numeric(df.iloc[:, idx], errors='coerce'), label=headers[idx])
    plt.xlabel('Time')
    plt.ylabel('Pressure (PSI)')
    plt.title('Pressure vs. Time')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    
        # Enhanced Download report as PDF
    from datetime import datetime

    st.subheader("📄 Generate Professional Report")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Generate PDF Report", type="primary"):
            try:
                issues = analyze_hvac_data_enhanced(df, headers, mapping)
                pdf_buffer = generate_pdf_report(
                    project_title="HVAC Diagnostic Report",
                    logo_file=None,
                    issues=issues,
                    df=df
                )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name="hvac_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                st.info("PDF generation requires additional libraries. Falling back to text report.")
    
                # Fallback to text report
                project_title = "HVAC Diagnostic Report"
                report_lines = [
                    f"{project_title}",
                    "="*len(project_title),
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "HVAC DIAGNOSTIC ANALYSIS REPORT",
                    "="*50,
                    "",
                    "SYSTEM DATA ANALYSIS FINDINGS:",
                    ""
                ]
    
                if issues:
                    high_issues = [i for i in issues if i.get('severity') == 'high']
                    medium_issues = [i for i in issues if i.get('severity') == 'medium']
                    low_issues = [i for i in issues if i.get('severity') == 'low']
    
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
                    "📄 Download Text Report (Fallback)",
                    report,
                    file_name=f"{project_title.replace(' ', '_')}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
    
    with col2:
        st.info(
            "📋 **PDF Report Includes:**\n"
            "- Executive Summary\n"
            "- Detailed Issue Analysis\n"
            "- Recommendations\n"
            "- Data Statistics\n"
            "- Professional Formatting"
        )
        
else:
    st.info("👆 Please upload CSV or XLSX files to begin HVAC data analysis")
    st.markdown("### 📋 **Expected Data Format**")
    st.markdown("""
    Your CSV and XLSX files should contain columns with names that include:
    - **Date/Time** information (e.g., 'Date', 'Timestamp')
    - **Suction Pressure** data (e.g., 'Suction Pressure', 'Suction PSI')
    - **Discharge Pressure** data (e.g., 'Discharge Pressure', 'Head Pressure')
    - **Temperature** readings (e.g., 'Suction Temp', 'Supply Air Temp', 'Discharge Temp')
    
    The system will automatically detect and analyze these parameters based on column names.
    """)
