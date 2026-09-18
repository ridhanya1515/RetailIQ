import os
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from config import Config
from utils.database import query_db

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import logging
logger = logging.getLogger(__name__)

def generate_pdf_report(kpis, segment_dist, forecast_df, insights, username="Analyst"):
    """
    Generate a highly styled, executive-ready PDF report of retail intelligence.
    Returns a BytesIO stream containing the PDF data.
    """
    buffer = BytesIO()
    
    # Page setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Premium SaaS look
    primary_color = colors.HexColor("#1e293b")  # Dark Slate
    accent_color = colors.HexColor("#3b82f6")   # Electric Blue
    text_color = colors.HexColor("#334155")     # Gray-700
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=30,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    bold_body_style = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # ------------------ TITLE PAGE / HEADER ------------------
    story.append(Paragraph("RETAILIQ EXECUTIVE SUMMARY", title_style))
    story.append(Paragraph(
        f"AI-Powered Retail Intelligence, Customer Segmentation & Demand Forecasting Report<br/>"
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Prepared by: {username}", 
        subtitle_style
    ))
    story.append(Spacer(1, 15))
    
    # ------------------ KPI DASHBOARD GRID ------------------
    story.append(Paragraph("Key Performance Indicators (KPIs)", h1_style))
    kpi_data = [
        [
            Paragraph("<b>Total Revenue</b>", body_style), 
            Paragraph(f"${kpis['total_revenue']:,.2f}", bold_body_style),
            Paragraph("<b>Total Orders</b>", body_style), 
            Paragraph(f"{kpis['total_orders']:,}", bold_body_style)
        ],
        [
            Paragraph("<b>Total Customers</b>", body_style), 
            Paragraph(f"{kpis['total_customers']:,}", bold_body_style),
            Paragraph("<b>Average Order Value</b>", body_style), 
            Paragraph(f"${kpis['avg_order_value']:,.2f}", bold_body_style)
        ],
        [
            Paragraph("<b>Revenue Growth (MoM)</b>", body_style), 
            Paragraph(f"{kpis['rev_growth_mom']:+.1f}%", bold_body_style),
            Paragraph("<b>Customer Growth (MoM)</b>", body_style), 
            Paragraph(f"{kpis['cust_growth_mom']:+.1f}%", bold_body_style)
        ]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[2.0*inch, 1.5*inch, 2.0*inch, 1.5*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (1,0), (1,-1), accent_color),
        ('TEXTCOLOR', (3,0), (3,-1), accent_color),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 25))
    
    # ------------------ BUSINESS INSIGHTS ------------------
    story.append(Paragraph("Automated Business Insights", h1_style))
    if not insights:
        story.append(Paragraph("No recent automated insights generated.", body_style))
    else:
        for ins in insights[:4]:  # Show top 4 insights
            importance_color = "#ef4444" if ins['importance'] == 'High' else ("#f59e0b" if ins['importance'] == 'Medium' else "#10b981")
            insight_text = (
                f"<b><font color='{importance_color}'>[{ins['importance']}]</font> "
                f"{ins['title']}</b> - <i>{ins['category']}</i><br/>"
                f"{ins['description']}"
            )
            story.append(Paragraph(insight_text, body_style))
            story.append(Spacer(1, 6))
            
    story.append(Spacer(1, 15))
    
    # ------------------ PAGE BREAK: SEGMENTATION ------------------
    story.append(PageBreak())
    story.append(Paragraph("Customer Segmentation Analysis", h1_style))
    story.append(Paragraph(
        "Customers are clustered into distinct business cohorts using KMeans Clustering on standardized RFM values (Recency, Frequency, Monetary).",
        body_style
    ))
    
    if segment_dist.empty:
        story.append(Paragraph("No customer segment data available.", body_style))
    else:
        # Table of Segment Distribution
        seg_table_data = [[
            Paragraph("<b>Segment</b>", bold_body_style),
            Paragraph("<b>Customers (%)</b>", bold_body_style),
            Paragraph("<b>Revenue Share</b>", bold_body_style),
            Paragraph("<b>Avg Recency</b>", bold_body_style),
            Paragraph("<b>Avg Frequency</b>", bold_body_style),
            Paragraph("<b>Avg Spend</b>", bold_body_style)
        ]]
        
        for _, row in segment_dist.iterrows():
            seg_table_data.append([
                Paragraph(f"{row['segment']}", body_style),
                Paragraph(f"{row['customer_count']:,} ({row['customer_share_pct']:.1f}%)", body_style),
                Paragraph(f"${row['total_revenue']:,.2f} ({row['revenue_share_pct']:.1f}%)", body_style),
                Paragraph(f"{row['average_recency']:.1f} days", body_style),
                Paragraph(f"{row['average_frequency']:.1f} orders", body_style),
                Paragraph(f"${row['average_spend']:,.2f}", body_style)
            ])
            
        seg_table = Table(seg_table_data, colWidths=[1.8*inch, 1.4*inch, 1.6*inch, 1.0*inch, 1.0*inch, 1.2*inch])
        seg_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (1,0), (-1,-1), 'LEFT'),
        ]))
        
        # Change headers font textcolor within table style is hard because they are Paragraphs, so we styled inside html bold tags
        story.append(seg_table)
        
    story.append(Spacer(1, 20))
    
    # ------------------ DEMAND FORECASTING ------------------
    story.append(Paragraph("Sales & Demand Forecasting (Next 30 Days)", h1_style))
    story.append(Paragraph(
        "Future transaction revenue is forecasted using statistical time series modeling (ARIMA/SARIMAX). The table below details upcoming demand trends.",
        body_style
    ))
    
    if forecast_df.empty:
        story.append(Paragraph("No demand forecasting results available.", body_style))
    else:
        # Show a summary: let's display every 5th day to save space or aggregate
        forecast_df_summary = forecast_df.head(10) # first 10 days
        
        fc_table_data = [[
            Paragraph("<b>Forecast Date</b>", bold_body_style),
            Paragraph("<b>Predicted Revenue</b>", bold_body_style),
            Paragraph("<b>Lower Confidence Limit</b>", bold_body_style),
            Paragraph("<b>Upper Confidence Limit</b>", bold_body_style)
        ]]
        
        for _, row in forecast_df_summary.iterrows():
            date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            fc_table_data.append([
                Paragraph(date_str, body_style),
                Paragraph(f"${row['Forecast']:,.2f}", bold_body_style),
                Paragraph(f"${row['Lower_CI']:,.2f}", body_style),
                Paragraph(f"${row['Upper_CI']:,.2f}", body_style)
            ])
            
        fc_table = Table(fc_table_data, colWidths=[2.0*inch, 1.8*inch, 1.6*inch, 1.6*inch])
        fc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(fc_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph("<i>*Values truncated to show the first 10 forecasted periods. Download the Excel report for full projections.</i>", body_style))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_excel_report(kpis, segment_dist, forecast_df, insights, rfm_df=None):
    """
    Generate a structured, multi-sheet Excel workbook using pandas & XlsxWriter.
    Returns a BytesIO stream containing the Excel workbook binary.
    """
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1E293B',
            'font_color': '#FFFFFF',
            'border': 1
        })
        
        money_format = workbook.add_format({'num_format': '$#,##0.00'})
        integer_format = workbook.add_format({'num_format': '#,##0'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        
        # --- Sheet 1: KPIs ---
        kpi_list = [
            {"Metric": "Total Revenue", "Value": kpis['total_revenue'], "Type": "Currency"},
            {"Metric": "Total Orders", "Value": kpis['total_orders'], "Type": "Integer"},
            {"Metric": "Total Customers", "Value": kpis['total_customers'], "Type": "Integer"},
            {"Metric": "Average Order Value", "Value": kpis['avg_order_value'], "Type": "Currency"},
            {"Metric": "Average Revenue per Customer", "Value": kpis['avg_revenue_per_customer'], "Type": "Currency"},
            {"Metric": "Revenue Growth MoM (%)", "Value": kpis['rev_growth_mom'] / 100.0, "Type": "Percentage"},
            {"Metric": "Customer Growth MoM (%)", "Value": kpis['cust_growth_mom'] / 100.0, "Type": "Percentage"},
        ]
        kpis_df = pd.DataFrame(kpi_list)
        kpis_df.to_excel(writer, sheet_name="KPI Summary", index=False)
        
        # Format KPI Sheet
        kpi_sheet = writer.sheets["KPI Summary"]
        kpi_sheet.set_column('A:A', 30)
        kpi_sheet.set_column('B:B', 15)
        
        for row_num in range(1, len(kpi_list) + 1):
            val_type = kpi_list[row_num - 1]["Type"]
            if val_type == "Currency":
                kpi_sheet.write_number(row_num, 1, kpi_list[row_num - 1]["Value"], money_format)
            elif val_type == "Integer":
                kpi_sheet.write_number(row_num, 1, kpi_list[row_num - 1]["Value"], integer_format)
            elif val_type == "Percentage":
                kpi_sheet.write_number(row_num, 1, kpi_list[row_num - 1]["Value"], percent_format)

        # --- Sheet 2: Segments Summary ---
        if not segment_dist.empty:
            segment_dist.to_excel(writer, sheet_name="Segments Distribution", index=False)
            seg_sheet = writer.sheets["Segments Distribution"]
            seg_sheet.set_column('A:A', 25)
            seg_sheet.set_column('B:B', 15, integer_format)
            seg_sheet.set_column('C:C', 20, money_format)
            seg_sheet.set_column('D:D', 20, money_format)
            seg_sheet.set_column('E:E', 15)
            seg_sheet.set_column('F:F', 15)
            seg_sheet.set_column('G:H', 15, percent_format)

        # --- Sheet 3: Demand Forecast ---
        if not forecast_df.empty:
            # Format date for Excel compatibility
            fc_excel = forecast_df.copy()
            fc_excel['Date'] = pd.to_datetime(fc_excel['Date']).dt.strftime('%Y-%m-%d')
            fc_excel.to_excel(writer, sheet_name="Demand Forecast", index=False)
            
            fc_sheet = writer.sheets["Demand Forecast"]
            fc_sheet.set_column('A:A', 15)
            fc_sheet.set_column('B:D', 18, money_format)

        # --- Sheet 4: Business Insights ---
        if insights:
            ins_df = pd.DataFrame(insights)
            ins_df = ins_df[['title', 'category', 'importance', 'description']]
            ins_df.to_excel(writer, sheet_name="Business Insights", index=False)
            
            ins_sheet = writer.sheets["Business Insights"]
            ins_sheet.set_column('A:A', 35)
            ins_sheet.set_column('B:C', 15)
            ins_sheet.set_column('D:D', 65)

        # --- Sheet 5: Customer Details (RFM) ---
        if rfm_df is not None and not rfm_df.empty:
            # Limit rows exported to Excel details to 20,000 for size limits
            rfm_export = rfm_df.head(20000).copy()
            rfm_export.to_excel(writer, sheet_name="Customer RFM Details", index=False)
            
            rfm_sheet = writer.sheets["Customer RFM Details"]
            rfm_sheet.set_column('A:A', 15)
            rfm_sheet.set_column('B:B', 12, integer_format)
            rfm_sheet.set_column('C:C', 12, integer_format)
            rfm_sheet.set_column('D:D', 18, money_format)
            rfm_sheet.set_column('E:F', 18, money_format)
            rfm_sheet.set_column('G:H', 18)

    buffer.seek(0)
    return buffer

def generate_csv_report(df, filename="report.csv"):
    """
    Generate a raw CSV export of dataframes.
    Returns a string containing the CSV data.
    """
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer
