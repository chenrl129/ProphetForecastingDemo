# BUILD PROPHET MODEL

# Imports
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the synthetic dataset from CSV 
data = pd.read_csv("synthetic_infusion_pump_utilization.csv")

# Group the data by date and compute the average utilization rate per day
daily_data = data.groupby('date').mean().reset_index()

# Select only the 'date' and 'utilization_rate' columns and rename them to be Prophet-friendly
daily_data = daily_data[['date', 'utilization_rate']]
daily_data.columns = ['ds', 'y']

# Fit the model to the data
model = Prophet()
model.fit(daily_data)

# Generate forecasts for the next 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot the forecast
fig1 = model.plot(forecast)
ax = fig1.gca()
ax.set_xlabel("Date")
ax.set_ylabel("Utilization Rate")
ax.set_ylim([0, 1])
plt.show()

# Plot the components
fig2 = model.plot_components(forecast)
ax = fig2.gca()
ax.set_xlabel("Date")
plt.show()


# Interactive Plot using Plotly 
from prophet.plot import plot_plotly, plot_components_plotly

plotly_fig1 = plot_plotly(model, forecast)
plotly_fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Utilization Rate",
    yaxis=dict(range=[0, 1])  # Set the y-axis range from 0 to 1
)
plotly_fig1.show()

plotly_fig2 = plot_components_plotly(model, forecast)
plotly_fig2.update_layout(
    xaxis_title="Date",
    yaxis=dict(range=[0, 1])  # Set the y-axis range from 0 to 1
)
plotly_fig2.show()



# GENERATE REPORT
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(forecast, filename):
    forecast_table_data = [['Date', 'Forecast', 'Lower Bound', 'Upper Bound']]
    for index, row in forecast.iterrows():
        forecast_table_data.append([row['ds'].strftime('%Y-%m-%d'), f"{row['yhat']:.2f}", f"{row['yhat_lower']:.2f}", f"{row['yhat_upper']:.2f}"])

    doc = SimpleDocTemplate(filename, pagesize=letter)

    # Define styles
    styles = getSampleStyleSheet()
    header_style = styles['Heading1']
    header_style.alignment = TA_CENTER
    header_style.fontSize = 24
    header_style.spaceAfter = 20

    body_style = ParagraphStyle(name='Body', fontSize=14, leading=18, spaceAfter=12)
    body_style.alignment = TA_JUSTIFY

    # Create document elements
    header = Paragraph("Utilization Forecast Report", header_style)
    intro = Paragraph("This report provides a forecast of equipment utilization rates for the next month. The table below presents the predicted utilization rates along with their lower and upper bounds.", body_style)
    forecast_table = Table(forecast_table_data, repeatRows=1)
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))

    # Add elements to the document
    content = [header, intro, Spacer(1, 0.5 * 72), forecast_table]
    doc.build(content)

pdf_filename = "equipment_forecast_report.pdf"
generate_pdf_report(forecast.tail(30)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], pdf_filename)