# Prophet Equipment Utilization Forecast

## Overview 📚
This project aims to predict future equipment utilization rates in a healthcare setting using the Prophet forecasting model. It uses a synthetic dataset containing equipment utilization rates and demonstrates how to preprocess the data, fit the model, generate forecasts, visualize the results, and create a PDF report.

## Data Description 📊

- This project uses a synthetic dataset containing equipment utilization rates.
- The dataset is in CSV format, named `synthetic_infusion_pump_utilization.csv`.

| Column            | Description                              |
| ----------------- | ---------------------------------------- |
| `date`            | The date when the utilization rate was recorded (YYYY-MM-DD). |
| `utilization_rate` | The equipment utilization rate, ranging from 0 to 1. |

## Project Workflow 🚀

1. Import the required libraries.
2. Load the historical data from a CSV file.
3. Preprocess the data for the Prophet model.
4. Fit the Prophet model to the data.
5. Generate forecasts for the next 90 days.
6. Visualize the forecast using Matplotlib and Plotly.

## Prerequisites 🛠️

To run the project, you'll need to have the following Python libraries installed:

- pandas
- fbprophet
- matplotlib
- plotly
- reportlab

You can install them using the following command:
```bash
pip install pandas fbprophet matplotlib plotly reportlab
```

## Usage Instructions 📚

1. Clone the repository.
2. Make sure the dataset is in the same directory as the script.
3. Run the script to generate the forecast, visualize it, and create a PDF report.

## Generating a PDF Report 📄

This project includes a function called `generate_pdf_report` that creates a PDF report of the equipment utilization forecast. The function takes two parameters:

- `forecast`: A pandas DataFrame containing the forecast data.
- `filename`: The name of the output PDF file.

To generate the report, simply call the function with the desired forecast data and output file name.

Example:

```python
pdf_filename = "equipment_forecast_report.pdf"
generate_pdf_report(forecast.tail(30)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], pdf_filename)
```

## Code Overview 📖

The project is organized into the following sections:

1. **Data Loading and Preprocessing**: Read the CSV file, group data by date, and compute the average utilization rate per day. Prepare the data for the Prophet model by selecting and renaming the required columns.
2. **Model Building**: Instantiate and fit the Prophet model using the preprocessed data.
3. **Forecast Generation**: Generate equipment utilization forecasts for the next 90 days.
4. **Visualization**: Create visualizations of the forecast using both Matplotlib and Plotly libraries. This includes an overall forecast plot and a components plot.
5. **Report Generation**: Create a PDF report of the equipment utilization forecast using the `reportlab` library. The `generate_pdf_report` function takes the forecast data and output file name as parameters and generates a PDF report containing a table of predicted utilization rates along with their lower and upper bounds.

An example forecast can be found here: https://chart-studio.plotly.com/create/?fid=rchen%3A7#/
