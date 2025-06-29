import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO, BytesIO
from datetime import datetime
import base64

# Only import reportlab if available
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("ReportLab not available. PDF generation will be limited to text reports.")

# --- Enhanced Helper Functions ---

def get_key_hvac_parameters(df, headers):
    """Identify and extract key HVAC parameters for focused summary"""
    key_params = {}
    
    # Define key parameter mappings with multiple possible column names
    parameter_mappings = {
        'Outside Air Temperature (OAT)': ['oat', 'outdoor air temp', 'outside air temp', 'outdoor temp'],
        'Supply Air Temperature (SAT)': ['sat', 'supply air temp', 'supply temp', 'discharge temp'],
        'Wet Bulb Temperature': ['oa wb', 'wet bulb', 'wb temp', 'oa_wb', 'outside air wb'],
        'Discharge Pressure 1': ['1dischg1', '1 dischg 1', 'discharge pressure 1', 'dischg1', 'dis chg 1'],
        'Suction Pressure 1': ['1sucpr1', '1 suc pr 1', 'suction pressure 1', 'sucpr1', 'suc pr 1'],
        'Head Pressure 1': ['1headpr1', '1 head pr 1', 'head pressure 1', 'headpr1', 'head pr 1'],
        'Condenser Pressure 1': ['1cond1', '1 cond 1', 'condenser pressure 1', 'cond1'],
        'Suction Temperature 1': ['1suctmp1', '1 suc tmp 1', 'suction temp 1', 'suctmp1', 'suc tmp 1'],
        'Outdoor Air RH': ['oa rh', 'outdoor air rh', 'outside air rh', 'oa_rh', 'outdoor rh'],
        'Cooling Setpoint (CSP)': ['csp', 'cooling setpoint', 'cool sp', 'cooling sp'],
        'Heating Setpoint (HSP)': ['hsp', 'heating setpoint', 'heat sp', 'heating sp'],
        'Supply Air Temp Setpoint': ['satsp', 'sat sp', 'supply air setpoint', 'supply temp sp'],
        'Coil Setpoint': ['coilsp', 'coil sp', '1coilsp', '2coilsp'],
        'Building Pressure': ['bldpr', 'building pressure', 'bld pr'],
        'Static Pressure': ['static', 'static pressure', 'static pr'],
        'Fan VFD': ['fanvfd', 'fan vfd', 'fan speed'],
        'Economizer': ['econo', 'economizer', 'econ']
    }
    
    # Search for each parameter
    for param_name, search_terms in parameter_mappings.items():
        for i, header in enumerate(headers):
            header_clean = str(header).strip().lower()
            for term in search_terms:
                if term.lower() in header_clean:
                    # Convert to numeric and get basic stats
                    try:
                        col_data = pd.to_numeric(df.iloc[:, i], errors='coerce').dropna()
                        if len(col_data) > 0 and not (col_data == 0).all():  # Skip all-zero columns
                            key_params[param_name] = {
                                'column_name': header,
                                'column_index': i,
                                'data': col_data,
                                'mean': col_data.mean(),
                                'min': col_data.min(),
                                'max': col_data.max(),
                                'std': col_data.std(),
                                'count': len(col_data)
                            }
                            break  # Found this parameter, move to next
                    except:
                        continue
                if param_name in key_params:
                    break  # Already found this parameter
            if param_name in key_params:
                break  # Already found this parameter
    
    return key_params

def create_key_parameter_summary(key_params):
    """Create a focused summary of key HVAC parameters"""
    if not key_params:
        return None
    
    # Create summary dataframe
    summary_data = []
    for param_name, param_info in key_params.items():
        summary_data.append({
            'Parameter': param_name,
            'Column': param_info['column_name'],
            'Mean': f"{param_info['mean']:.2f}",
            'Min': f"{param_info['min']:.2f}",
            'Max': f"{param_info['max']:.2f}",
            'Std Dev': f"{param_info['std']:.2f}",
            'Data Points': param_info['count']
        })
    
    return pd.DataFrame(summary_data)

def analyze_key_parameter_issues(key_params):
    """Analyze key parameters for potential issues"""
    issues = []
    
    for param_name, param_info in key_params.items():
        data = param_info['data']
        mean_val = param_info['mean']
        
        # Temperature analysis
        if 'Temperature' in param_name:
            if 'Outside Air' in param_name or 'OAT' in param_name:
                # OAT analysis - flag extreme temperatures
                if mean_val > 95:
                    issues.append({
                        'severity': 'medium',
                        'message': f'High average outdoor air temperature: {mean_val:.1f}°F',
                        'explanation': 'High outdoor temperatures increase cooling load and system stress',
                        'suggestions': ['Monitor system efficiency', 'Check condenser operation', 'Verify adequate airflow'],
                        'issue_type': 'environmental'
                    })
                elif mean_val < 32:
                    issues.append({
                        'severity': 'medium',
                        'message': f'Low average outdoor air temperature: {mean_val:.1f}°F',
                        'explanation': 'Low outdoor temperatures may affect heating system operation',
                        'suggestions': ['Check heating system operation', 'Monitor for freeze protection', 'Verify economizer operation'],
                        'issue_type': 'environmental'
                    })
            
            elif 'Supply Air' in param_name or 'SAT' in param_name:
                # SAT analysis
                if mean_val > 65:
                    issues.append({
                        'severity': 'medium',
                        'message': f'High supply air temperature: {mean_val:.1f}°F',
                        'explanation': 'High supply air temperature may indicate insufficient cooling or controls issues',
                        'suggestions': ['Check cooling coil operation', 'Verify thermostat settings', 'Inspect damper operation'],
                        'issue_type': 'cooling_system'
                    })
                elif mean_val < 45:
                    issues.append({
                        'severity': 'medium',
                        'message': f'Very low supply air temperature: {mean_val:.1f}°F',
                        'explanation': 'Very low supply air temperature may indicate overcooling or control issues',
                        'suggestions': ['Check cooling controls', 'Verify setpoints', 'Inspect mixing dampers'],
                        'issue_type': 'control_system'
                    })
        
        # Pressure analysis
        elif 'Pressure' in param_name:
            if 'Discharge' in param_name or 'Head' in param_name:
                if mean_val > 400:
                    issues.append({
                        'severity': 'high',
                        'message': f'High discharge pressure in {param_info["column_name"]}: {mean_val:.1f} PSI',
                        'explanation': 'High discharge pressure indicates condenser problems or system overcharge',
                        'suggestions': ['Clean condenser coil', 'Check condenser fan', 'Verify refrigerant charge'],
                        'issue_type': 'condenser_system'
                    })
                elif mean_val < 150:
                    issues.append({
                        'severity': 'medium',
                        'message': f'Low discharge pressure in {param_info["column_name"]}: {mean_val:.1f} PSI',
                        'explanation': 'Low discharge pressure may indicate undercharge or compressor issues',
                        'suggestions': ['Check refrigerant charge', 'Test compressor operation', 'Inspect for leaks'],
                        'issue_type': 'compressor_system'
                    })
            
            elif 'Suction' in param_name:
                if mean_val > 200:
                    issues.append({
                        'severity': 'high',
                        'message': f'High suction pressure in {param_info["column_name"]}: {mean_val:.1f} PSI',
                        'explanation': 'High suction pressure may indicate overcharge or restricted airflow',
                        'suggestions': ['Check refrigerant levels', 'Inspect air filters', 'Verify ductwork'],
                        'issue_type': 'refrigerant_system'
                    })
                elif mean_val < 50:
                    issues.append({
                        'severity': 'medium',
                        'message': f'Low suction pressure in {param_info["column_name"]}: {mean_val:.1f} PSI',
                        'explanation': 'Low suction pressure may indicate refrigerant leak or expansion valve issues',
                        'suggestions': ['Check for refrigerant leaks', 'Inspect expansion valve', 'Verify system charge'],
                        'issue_type': 'refrigerant_system'
                    })
        
        # Humidity analysis
        elif 'RH' in param_name and 'Outdoor' in param_name:
            if mean_val > 70:
                issues.append({
                    'severity': 'medium',
                    'message': f'High outdoor relative humidity: {mean_val:.1f}%',
                    'explanation': 'High outdoor humidity increases latent cooling load',
                    'suggestions': ['Monitor dehumidification performance', 'Check condensate drainage', 'Verify coil operation'],
                    'issue_type': 'humidity_control'
                })
    
    return issues

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
        'time': None,
        'datetime': None
    }

    for i, h in enumerate(headers):
        h_clean = str(h).strip()
        header_lower = h_clean.lower()

        # Date/Time detection first
        if any(keyword in header_lower for keyword in ['timestamp', 'datetime']):
            mapping['datetime'] = i
        elif any(keyword in header_lower for keyword in ['date']) and mapping['date'] is None:
            mapping['date'] = i
        elif any(keyword in header_lower for keyword in ['time']) and mapping['time'] is None:
            mapping['time'] = i

        # Relative Humidity detection
        elif any(keyword in header_lower for keyword in ['rel hum', 'rel. hum', 'relative humidity', 'rh']):
            # Skip if "oa rh", "Outside Air", or any outside air indicator is in the header
            if 'oa rh' not in header_lower:
                mapping['relativeHumidity'].append(i)

        # Indoor Temperature detection
        elif any(keyword in header_lower for keyword in ['indoor temp', 'indoor temperature', 'room temp', 'spacetemp','space temp','space-temp']):
            mapping['indoorTemps'].append(i)

        # Enhanced pressure detection
        elif any(keyword in header_lower for keyword in ['1sucpr1','suction', 'sucpr','suc pr', 'suction pr', 'suction_pr']) or \
             (('suc' in header_lower or 'suction' in header_lower) and ('pr' in header_lower or 'pressure' in header_lower)):
            mapping['suctionPressures'].append(i)

        elif any(keyword in header_lower for keyword in ['1dischg1','dischg', 'dis chg', 'discharge pr', 'head pr', 'headpr', '1cond1', '1headpr1']) or \
             (('discharge' in header_lower or 'head' in header_lower or 'cond' in header_lower) and ('pr' in header_lower or 'pressure' in header_lower)):
            mapping['dischargePressures'].append(i)

        # Enhanced temperature detection
        elif any(keyword in header_lower for keyword in ['1suctmp1','suctmp', 'suc tmp', 'suction tmp', 'suction_tmp', 'suction temp']):
            mapping['suctionTemps'].append(i)

        elif any(keyword in header_lower for keyword in ['sat', 'supply air', 'supply_air', 'discharge temp']):
            mapping['supplyAirTemps'].append(i)

        elif any(keyword in header_lower for keyword in ['dischg', 'dis chg', 'discharge']) and 'temp' in header_lower:
            mapping['dischargeTemps'].append(i)

        elif any(keyword in header_lower for keyword in ['oat', 'outdoor', 'outside']) and ('temp' in header_lower or 'air' in header_lower):
            mapping['outdoorAirTemps'].append(i)

        # Setpoint detection
        elif any(keyword in header_lower for keyword in ['csp', 'cool', 'cooling']) and ('sp' in header_lower or 'setpoint' in header_lower):
            mapping['coolingSetpoints'].append(i)

        elif any(keyword in header_lower for keyword in ['hsp', 'heat', 'heating']) and ('sp' in header_lower or 'setpoint' in header_lower):
            mapping['heatingSetpoints'].append(i)

    return mapping

def create_datetime_column(df, mapping):
    """Create a datetime column from date/time or datetime columns, with support for '31-May' format"""
    try:
        if mapping['datetime'] is not None:
            df['parsed_datetime'] = pd.to_datetime(df.iloc[:, mapping['datetime']], errors='coerce')
        elif mapping['date'] is not None and mapping['time'] is not None:
            date_col = df.iloc[:, mapping['date']].astype(str).str.strip()
            time_col = df.iloc[:, mapping['time']].astype(str).str.strip()

            # Convert '31-May' to '2024-05-31'
            def convert_date(date_str):
                if pd.isna(date_str) or date_str == 'nan':
                    return None
                if '-' in date_str and len(date_str.split('-')) == 2:
                    parts = date_str.split('-')
                    if parts[0].isdigit():
                        day = parts[0]
                        month = parts[1]
                        # Convert month name to number
                        month_map = {
                            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                        }
                        month_num = month_map.get(month.lower()[:3], month)
                        return f"2024-{month_num}-{day.zfill(2)}"
                return date_str

            date_col = date_col.apply(convert_date)
            datetime_str = date_col + ' ' + time_col
            df['parsed_datetime'] = pd.to_datetime(datetime_str, errors='coerce')

        elif mapping['date'] is not None:
            date_col = df.iloc[:, mapping['date']].astype(str).str.strip()
            date_col = date_col.apply(lambda x: convert_date(x) if callable(convert_date) else x)
            df['parsed_datetime'] = pd.to_datetime(date_col, errors='coerce')
        else:
            df['parsed_datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

        return df

    except Exception as e:
        st.warning(f"Could not parse datetime: {e}. Using sequential index.")
        df['parsed_datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        return df

def check_comfort_conditions(df, headers, mapping):
    """Check indoor comfort conditions"""
    results = []

    # Check relative humidity
    for idx in mapping.get('indoorrh', []):
        humidity_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(humidity_data) > 0:
            above_60 = (humidity_data > 60).sum()
            percent_over = (above_60 / len(humidity_data)) * 100
            avg_humidity = humidity_data.mean()

            results.append({
                'type': 'Indoor Relative Humidity',
                'column': headers[idx],
                'average': avg_humidity,
                'percent_over': percent_over,
                'compliant': percent_over == 0
            })

    # Check indoor temperature
    for idx in mapping.get('indoorTemps', []):
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

def analyze_hvac_data_enhanced(df, headers, mapping):
    """Enhanced HVAC analysis with improved detection logic"""
    issues = []

    # Check suction pressures
    for idx in mapping['suctionPressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_pressure = col_data.mean()
            if avg_pressure > 200:
                issues.append({
                    'message': f'High suction pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)',
                    'severity': 'high',
                    'explanation': 'High suction pressure may indicate system overcharge or restricted airflow',
                    'suggestions': ['Check refrigerant levels', 'Inspect air filters', 'Verify ductwork'],
                    'issue_type': 'refrigerant_system'
                })
            elif avg_pressure < 50:
                issues.append({
                    'message': f'Low suction pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)',
                    'severity': 'medium',
                    'explanation': 'Low suction pressure may indicate refrigerant leak or expansion valve issues',
                    'suggestions': ['Check for refrigerant leaks', 'Inspect expansion valve', 'Verify system charge'],
                    'issue_type': 'refrigerant_system'
                })

    # Check discharge pressures
    for idx in mapping['dischargePressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_pressure = col_data.mean()
            if avg_pressure > 400:
                issues.append({
                    "severity": "high",
                    "message": f"High discharge pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coil", "Check condenser fan operation", "Verify proper airflow", "Check for overcharge"],
                    "issue_type": "condenser_system"
                })
            elif avg_pressure < 150:
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"],
                    "issue_type": "compressor_system"
                })

    # Check suction temperatures
    for idx in mapping['suctionTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_temp = col_data.mean()
            if avg_temp > 65:
                issues.append({
                    "severity": "medium",
                    "message": f"High suction temperature in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High suction temperature indicates low refrigerant charge or expansion valve problems.",
                    "suggestions": ["Check superheat settings", "Verify refrigerant charge", "Inspect expansion valve", "Check for restrictions"],
                    "issue_type": "refrigerant_system"
                })
            elif avg_temp < 35:
                issues.append({
                    "severity": "high",
                    "message": f"Low suction temperature risk in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Very low suction temperature risks liquid refrigerant returning to compressor.",
                    "suggestions": ["Check superheat immediately", "Verify proper airflow", "Inspect expansion valve", "Check for flooding"],
                    "issue_type": "refrigerant_system"
                })

    # Check supply air and discharge temperatures
    for idx in mapping['supplyAirTemps'] + mapping['dischargeTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_temp = col_data.mean()
            if avg_temp > 120:
                issues.append({
                    "severity": "high",
                    "message": f"High discharge temperature in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High discharge temperature indicates compressor stress, poor heat rejection, or overcharge.",
                    "suggestions": ["Check condenser operation", "Verify proper airflow", "Check refrigerant charge", "Inspect compressor condition"],
                    "issue_type": "compressor_system"
                })
            elif avg_temp < 50:
                issues.append({
                    "severity": "medium",
                    "message": f"Very low supply air temperature in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Extremely low supply air temperature may indicate overcooling or control issues.",
                    "suggestions": ["Check thermostat settings", "Verify cooling load", "Inspect damper operation", "Check for overcooling"],
                    "issue_type": "control_system"
                })

    return issues

def generate_pdf_report(project_title, logo_file, issues, df_summary=None):
    """Generate a comprehensive PDF report"""
    if not REPORTLAB_AVAILABLE:
        return None

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
            logo_file.seek(0)
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
                story.append(Spacer(1, 12))

        # Medium Priority Issues
        if medium_issues:
            story.append(Paragraph("🟡 MEDIUM PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(medium_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                story.append(Spacer(1, 12))

        # Low Priority Issues
        if low_issues:
            story.append(Paragraph("🔵 LOW PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(low_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                story.append(Spacer(1, 12))

    # Add data summary if provided
    if df_summary is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Data Summary Statistics", heading_style))
        
        try:
            numeric_df = df_summary.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_data = [['Parameter', 'Mean', 'Min', 'Max', 'Std Dev']]
                for col in numeric_df.columns[:10]:
                    mean = numeric_df[col].mean()
                    min_val = numeric_df[col].min()
                    max_val = numeric_df[col].max()
                    std_dev = numeric_df[col].std()
                    
                    if not (mean == 0 and min_val == 0 and max_val == 0 and std_dev == 0):
                        stats_data.append([
                            col,
                            f"{mean:.2f}",
                            f"{min_val:.2f}",
                            f"{max_val:.2f}",
                            f"{std_dev:.2f}"
                        ])

                table = Table(stats_data)
                table.setStyle
