# Sunderland Carbon Intensity Dashboard

This Streamlit app provides real-time and historical data on carbon intensity in Sunderland, UK. The data is sourced from the UK Carbon Intensity API and displayed interactively, offering insights into the environmental impact of energy use in the region.

## Features

- **Real-Time Monitoring**: Get current carbon intensity data for Sunderland.
- **Historical Data Analysis**: Visualize past trends using interactive tools.
- **Interactive Date Range Selection**: Filter data based on a custom date range.
- **Detailed Carbon Intensity Statistics**: Analyze key statistics and visualizations.

## How It Works

1. **Data Collection**:
   - The app reads carbon intensity data from a local CSV file (`carbon.csv`) and fetches additional data from the UK Carbon Intensity API based on available dates.
   - The data is updated to reflect the latest information and cached for faster performance.

2. **Date Range Handling**:
   - The app calculates a date range from the earliest timestamp in the data to the current UK time, rounded to the nearest half hour.
   - Users can adjust the date range with a Streamlit slider to filter and visualize the carbon intensity data.

3. **Visualizations**:
   - The app displays an interactive line chart of carbon intensity levels over the selected time period.
   - Users can also explore summary statistics for the filtered data.

## Requirements

- **Python 3.6+**
- Required libraries:
  - `streamlit`
  - `pandas`
  - `requests`
  - `pytz`

Install the dependencies using:

```bash
pip install streamlit pandas requests pytz
