import streamlit as st
#import matplotlib.pyplot as plt  # Import for plotting
#import seaborn as sns  # Import for seaborn-based visualizations
import pandas as pd
from pathlib import Path
import pytz
from datetime import datetime, timedelta
import requests
#import matplotlib.pyplot as plt
# import plotly.express as px  # Import Plotly for plotting
import altair as alt
# Set the title and favicon for the browser tab
st.set_page_config(page_title='Sunderland Carbon Intensity', page_icon=':earth_africa:')

# Function to load and process Carbon Intensity data
@st.cache_data
def get_carbon_data():
    DATA_FILENAME = Path(__file__).parent / 'data/carbon.csv'
    if DATA_FILENAME.exists():
        raw_carbon_df = pd.read_csv(DATA_FILENAME, parse_dates=['from', 'to'], infer_datetime_format=True)
        raw_carbon_df['from'] = pd.to_datetime(raw_carbon_df['from'], utc=True)
        raw_carbon_df['to'] = pd.to_datetime(raw_carbon_df['to'], utc=True)
    else:
        raw_carbon_df = pd.DataFrame(columns=['from', 'to', 'forecast', 'index'])  # Empty DataFrame structure
    
    try: 
          raw_carbon_df = raw_carbon_df.drop('Unnammed:0', axis=1)
    except:
            print(raw_carbon_df.columns)
    return raw_carbon_df

# Function to get the last available timestamp from the DataFrame
def get_last_timestamp_from_df(df, timestamp_column):
    if not df.empty:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        return df[timestamp_column].max()
    else:
        return None

# Get current UK time rounded to the nearest half hour
def get_current_uk_time_rounded():
    uk_timezone = pytz.timezone('Europe/London')
    current_time = datetime.now(uk_timezone)
    
    # Round to the nearest half-hour
    minutes = current_time.minute
    if minutes < 15:
        rounded_time = current_time.replace(minute=0, second=0, microsecond=0)
    elif 15 <= minutes < 45:
        rounded_time = current_time.replace(minute=30, second=0, microsecond=0)
    else:
        rounded_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    return rounded_time

# Define intensity categories
def categorize_intensity(forecast):
    if forecast < 50:
        return "Very Low"
    elif 50 <= forecast < 100:
        return "Low"
    elif 100 <= forecast < 150:
        return "Moderate"
    elif 150 <= forecast < 200:
        return "High"
    else:
        return "Very High"
# Fetch data from the Carbon Intensity API
def fetch_data(start, end):
    headers = {'Accept': 'application/json'}
    url = f'https://api.carbonintensity.org.uk/regional/intensity/{start.strftime("%Y-%m-%dT%H:%MZ")}/{end.strftime("%Y-%m-%dT%H:%MZ")}/postcode/me4'
    response = requests.get(url, headers=headers)
    return response.json()
# Main function to generate date range for new data fetching
def generate_date_range_for_fetching(df, timestamp_column):
    last_saved_timestamp = get_last_timestamp_from_df(df, timestamp_column)
    
    if last_saved_timestamp is None:
        start_date = datetime(2021, 1, 1, tzinfo=pytz.UTC)  # Default start date if no data is available
    else:
        start_date = last_saved_timestamp + timedelta(minutes=30)  # Start fetching from the next time slot

    end_date = get_current_uk_time_rounded()
    
    return start_date, end_date

# Append new data to the CSV file
def append_new_data_to_csv(new_data, filename):
    if not new_data.empty:
        new_data.to_csv(filename, mode='a', header=False, index=False)
        
        
# Load the Carbon Intensity data
carbon_df = get_carbon_data()
# Determine the date range for fetching new data
start_date, end_date = generate_date_range_for_fetching(carbon_df, 'to')

# Fetch new data only if there's a gap between the last available timestamp and the current time
if start_date < end_date:
    all_records = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=1), end_date)  # Fetch data day by day
        data = fetch_data(current_start, current_end)
        try:
            entries = data['data']['data']
        except:
            continue

        for entry in entries:
            record = {
                'from': entry['from'],
                'to': entry['to'],
                'forecast': entry['intensity']['forecast'],
                'index': entry['intensity']['index'],
            }
            for mix in entry['generationmix']:
                record[mix['fuel']] = mix['perc']
            all_records.append(record)

        current_start += timedelta(days=1)

    # Convert list of records to DataFrame
    if all_records:
        new_data_df = pd.DataFrame(all_records)
        new_data_df['from'] = pd.to_datetime(new_data_df['from'], utc=True)
        new_data_df['to'] = pd.to_datetime(new_data_df['to'], utc=True)

        # Append new data to the existing DataFrame and CSV
        carbon_df = pd.concat([carbon_df, new_data_df], ignore_index=True)
        try: 
          carbon_df = carbon_df.drop('Unnammed:0', axis=1)
        except:
            print(carbon_df.columns)
                  
        append_new_data_to_csv(new_data_df, Path(__file__).parent / 'data/carbon.csv')
# Ensure 'from' column is in UTC and sort by time
carbon_df['from'] = pd.to_datetime(carbon_df['from'], utc=True)
carbon_df = carbon_df.sort_values(by='from', ascending=False)

try: 
          carbon_df = carbon_df.drop('Unnammed:0', axis=1)
except:
            print(carbon_df.columns)

# Get the latest data
latest_data = carbon_df.iloc[0]
latest_forecast = float(latest_data['forecast'])
latest_index = latest_data['index']

# Dashboard title and introduction
st.title(':earth_africa: Sunderland Carbon Intensity Dashboard')

# --- Latest Carbon Intensity Section ---
st.header('🔍 Latest Carbon Intensity')

# Show the forecast value with the category
st.metric("Carbon Intensity", f"{latest_forecast} gCO₂/kWh", f"Status: {latest_index}")

# Map status to colors for visual appeal
color_map = {
    "very low": "green",
    "low": "lightgreen",
    "moderate": "orange",
    "high": "red",
    "very high": "darkred"
}

# Use HTML and inline styles for colored text
st.markdown(f'<h3>Intensity Status: <span style="color:{color_map[latest_index]};">{latest_index}</span></h3>', unsafe_allow_html=True)

# Display a progress bar to visually indicate the intensity category
st.progress(int((latest_forecast / 250) * 100))  # Assuming 250 is the upper limit for "Very High"

# --- Carbon Intensity Over Time Section ---
st.header('📊 Carbon Intensity Over Time')

# Filter data to show the latest 48 hours (for example)
uk_timezone = pytz.timezone('Europe/London')
current_time = datetime.now(uk_timezone)
time_window = current_time - pd.Timedelta(hours=48)

recent_df = carbon_df[carbon_df['from'] >= time_window]

# Plot the line chart for recent carbon intensity
st.line_chart(recent_df, x='from', y='forecast')
# Add columns for 'day' and 'hour'
carbon_df['day'] = carbon_df['from'].dt.date
carbon_df['hour'] = carbon_df['from'].dt.hour

# Filter the data for the last two weeks
last_two_weeks = carbon_df[carbon_df['from'] >= (carbon_df['from'].max() - pd.Timedelta(days=14))]

# Boxplot by Day using Altair (for the last 2 weeks)
boxplot_day = alt.Chart(last_two_weeks).mark_boxplot().encode(
    x=alt.X('day:T', title='Day'),
    y=alt.Y('forecast:Q', title='Carbon Intensity (gCO₂/kWh)')
).properties(
    title='Carbon Intensity Forecast (gCO₂/kWh) by Day - Last 2 Weeks',
    width=600
)
st.header('📆 Carbon Intensity Forecast (gCO₂/kWh) by Day - Last 2 Weeks')
# Display day-based boxplot in Streamlit
st.altair_chart(boxplot_day)

# Boxplot by Hour using Altair
boxplot_hour = alt.Chart(last_two_weeks).mark_boxplot().encode(
    x=alt.X('hour:O', title='Hour'),
    y=alt.Y('forecast:Q', title='Carbon Intensity (gCO₂/kWh)')
).properties(
    # title='Carbon Intensity Forecast (gCO₂/kWh) by Hour - Last 2 Weeks',
    width=600
)
st.header('🕖 Carbon Intensity Forecast (gCO₂/kWh) by Hour - Last 2 Weeks')

# Display hour-based boxplot in Streamlit
st.altair_chart(boxplot_hour)
# --- Data Statistics and Filtering Section ---
st.header('📅 Select Date Range and View Data')

# Filter data based on selected date range
min_date = carbon_df['from'].min()
max_date = carbon_df['from'].max()

# Streamlit slider for date range selection
selected_dates = st.slider('Select the date range:', 
                           min_value=min_date.to_pydatetime(), 
                           max_value=max_date.to_pydatetime(), 
                           value=(min_date.to_pydatetime(), max_date.to_pydatetime()))

# Filter DataFrame based on selected date range
start_date = pd.to_datetime(selected_dates[0]).tz_convert('UTC')
end_date = pd.to_datetime(selected_dates[1]).tz_convert('UTC')
filtered_carbon_df = carbon_df[(carbon_df['from'] >= start_date) & (carbon_df['from'] <= end_date)]

# Display filtered data
st.write(filtered_carbon_df)

# Display summary statistics for filtered data
st.header('📈 Carbon Intensity Statistics')
st.write(filtered_carbon_df.describe())

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import pytz
# import requests
# from datetime import datetime, timedelta

# # Set the title and favicon for the browser tab
# st.set_page_config(page_title='Sunderland Carbon Intensity', page_icon=':earth_africa:')

# # Function to load and process Carbon Intensity data
# @st.cache_data
# def get_carbon_data():
#     """Load Carbon Intensity data from a CSV file with caching."""
#     DATA_FILENAME = Path(__file__).parent / 'data/carbon.csv'
#     if DATA_FILENAME.exists():
#         raw_carbon_df = pd.read_csv(DATA_FILENAME, parse_dates=['from', 'to'], infer_datetime_format=True)
#         raw_carbon_df['from'] = pd.to_datetime(raw_carbon_df['from'], utc=True)
#         raw_carbon_df['to'] = pd.to_datetime(raw_carbon_df['to'], utc=True)
#     else:
#         raw_carbon_df = pd.DataFrame(columns=['from', 'to', 'forecast', 'index'])  # Empty DataFrame structure
#     return raw_carbon_df

# # Function to get the last available timestamp from the DataFrame
# def get_last_timestamp_from_df(df, timestamp_column):
#     if not df.empty:
#         df[timestamp_column] = pd.to_datetime(df[timestamp_column])
#         return df[timestamp_column].max()
#     else:
#         return None

# # Get current UK time rounded to the nearest half hour
# def get_current_uk_time_rounded():
#     uk_timezone = pytz.timezone('Europe/London')
#     current_time = datetime.now(uk_timezone)
    
#     # Round to the nearest half-hour
#     minutes = current_time.minute
#     if minutes < 15:
#         rounded_time = current_time.replace(minute=0, second=0, microsecond=0)
#     elif 15 <= minutes < 45:
#         rounded_time = current_time.replace(minute=30, second=0, microsecond=0)
#     else:
#         rounded_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

#     return rounded_time






# # Define intensity categories
# def categorize_intensity(forecast):
#     if forecast < 50:
#         return "Very Low"
#     elif 50 <= forecast < 100:
#         return "Low"
#     elif 100 <= forecast < 150:
#         return "Moderate"
#     elif 150 <= forecast < 200:
#         return "High"
#     else:
#         return "Very High"


# # Load the Carbon Intensity data
# carbon_df = get_carbon_data()

# # Determine the date range for fetching new data
# start_date, end_date = generate_date_range_for_fetching(carbon_df, 'to')

# # Fetch new data only if there's a gap between the last available timestamp and the current time
# if start_date < end_date:
#     all_records = []
#     current_start = start_date
#     while current_start < end_date:
#         current_end = min(current_start + timedelta(days=1), end_date)  # Fetch data day by day
#         data = fetch_data(current_start, current_end)
#         try:
#             entries = data['data']['data']
#         except:
#             continue

#         for entry in entries:
#             record = {
#                 'from': entry['from'],
#                 'to': entry['to'],
#                 'forecast': entry['intensity']['forecast'],
#                 'index': entry['intensity']['index'],
#             }
#             for mix in entry['generationmix']:
#                 record[mix['fuel']] = mix['perc']
#             all_records.append(record)

#         current_start += timedelta(days=1)

#     # Convert list of records to DataFrame
#     if all_records:
#         new_data_df = pd.DataFrame(all_records)
#         new_data_df['from'] = pd.to_datetime(new_data_df['from'], utc=True)
#         new_data_df['to'] = pd.to_datetime(new_data_df['to'], utc=True)

#         # Append new data to the existing DataFrame and CSV
#         carbon_df = pd.concat([carbon_df, new_data_df], ignore_index=True)
#         append_new_data_to_csv(new_data_df, Path(__file__).parent / 'data/carbon.csv')

# # Dashboard content and visualizations
# # st.title(':earth_americas: Sunderland Carbon Intensity')

# # Introduction text
# st.markdown("""
# ## Welcome to the Sunderland Carbon Intensity Dashboard!

# ###### Dr Basel Barakat
# ###### University of Sunderland
# ###### School of Computer Science and Engineering
# """)

# # Filter data based on selected date range
# carbon_df['from'] = pd.to_datetime(carbon_df['from'], utc=True)
# min_date = carbon_df['from'].min()
# max_date = carbon_df['from'].max()

# # Streamlit slider for date range selection
# selected_dates = st.slider('Select the date range:', 
#                            min_value=min_date.to_pydatetime(), 
#                            max_value=max_date.to_pydatetime(), 
#                            value=(min_date.to_pydatetime(), max_date.to_pydatetime()))

# # Filter DataFrame based on selected date range
# start_date = pd.to_datetime(selected_dates[0]).tz_convert('UTC')
# end_date = pd.to_datetime(selected_dates[1]).tz_convert('UTC')
# filtered_carbon_df = carbon_df[(carbon_df['from'] >= start_date) & (carbon_df['from'] <= end_date)]


# # Load the Carbon Intensity data
# # carbon_df = get_carbon_data()

# # Ensure 'from' column is in UTC and sort by time
# carbon_df['from'] = pd.to_datetime(carbon_df['from'], utc=True)
# carbon_df = carbon_df.sort_values(by='from', ascending=False)

# # Get the latest data
# latest_data = carbon_df.iloc[0]
# latest_forecast = float( latest_data['forecast'])
# latest_index = latest_data['index'] # categorize_intensity(latest_forecast)

# # Dashboard title and introduction
# st.title(':earth_africa: Sunderland Carbon Intensity Dashboard')

# # Show the latest carbon intensity in a clear, visual format
# st.header('Latest Carbon Intensity')

# # Show the forecast value with the category
# st.metric("Carbon Intensity", f"{latest_forecast} gCO₂/kWh", f"Status: {latest_index}")

# # Map status to colors for visual appeal
# color_map = {
#     "very low": "green",
#     "low": "lightgreen",
#     "moderate": "orange",
#     "high": "red",
#     "very high": "darkred"
# }
# # st.markdown(f"### Intensity Status: :{color_map[latest_index]}[{latest_index}]:")
# # Use HTML and inline styles for colored text
# st.markdown(f'<h3>Intensity Status: <span style="color:{color_map[latest_index]};">{latest_index}</span></h3>', unsafe_allow_html=True)

# # Display a progress bar to visually indicate the intensity category
# st.progress(int((latest_forecast / 250) * 100))  # Assuming 250 is the upper limit for "Very High"

# # Visualize the trend of carbon intensity over time
# st.header('Carbon Intensity Over Time')

# # Filter data to show the latest 48 hours (for example)
# uk_timezone = pytz.timezone('Europe/London')
# current_time = datetime.now(uk_timezone)
# time_window = current_time - pd.Timedelta(hours=48)

# recent_df = carbon_df[carbon_df['from'] >= time_window]

# # Plot the line chart for recent carbon intensity
# st.line_chart(recent_df, x='from', y='forecast')




# # Display filtered data
# st.write(filtered_carbon_df)


# # Display summary statistics for filtered data
# st.header('Carbon Intensity Statistics')
# st.write(recent_df.describe())
# # Carbon Intensity Line Chart
# st.header('Carbon Intensity Over Time')
# st.line_chart(filtered_carbon_df, x='from', y='forecast')

# # Display summary statistics
# st.header('Carbon Intensity Statistics')
# st.write(filtered_carbon_df.describe())

# st.write(f"Data from {selected_dates[0].strftime('%Y-%m-%d %H:%M')} to {selected_dates[1].strftime('%Y-%m-%d %H:%M')}")


