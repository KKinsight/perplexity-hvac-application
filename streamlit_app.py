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
        story.append(PageBreak())
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
    This automated diagnostic report is based on pattern analysis of HVAC system data. 
    All recommendations should be verified by qualified HVAC technicians before implementation. 
    Regular maintenance and professional inspections are essential for optimal system performance.
    """, normal_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by {project_title} Analysis System", normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

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

        # Enhanced Download report as PDF
        st.subheader("📄 Generate Professional Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate PDF Report", type="primary"):
                try:
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(project_title, logo_file, issues, df)
                    
                    # Offer download
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"{project_title.replace(' ', '_')}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("PDF generation requires additional libraries. Falling back to text report.")
                    
                    # Fallback to text report
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
                        "📄 Download Text Report (Fallback)", 
                        report, 
                        file_name=f"{project_title.replace(' ', '_')}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
        
        with col2:
            st.info("📋 **PDF Report Includes:**\n- Executive Summary\n- Detailed Issue Analysis\n- Recommendations\n- Data Statistics\n- Professional Formatting")

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
