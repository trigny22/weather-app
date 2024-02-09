import datetime
from matplotlib import pyplot as plt
import openmeteo_requests
import geopandas as gpd
import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from slack_sdk import WebClient
# from slack_sdk.errors import SlackApiError
import matplotlib.pyplot as plt
# import io
# import base64
import streamlit as st

# import requests

# Load Europe shapefile
europe = gpd.read_file('NUTS_BN_20M_2021_3035.shp')

european_capitals = {
    "France": {"Paris": (48.8566, 2.3522)},
    "Germany": {"Berlin": (52.5200, 13.4050)},
    "Spain": {"Madrid": (40.4168, -3.7038)},
    "United Kingdom": {"London": (51.5074, -0.1278)},
    "Iceland": {"Reykjavik": (64.1466, -21.9426)},
    "Sweden": {"Stockholm": (59.3293, 18.0686)},
    "Poland": {"Warsaw": (52.2297, 21.0122)},
    "Ukraine": {"Kyiv": (50.4501, 30.5234)},
    "Greece": {"Athens": (37.9838, 23.7275)}
}

# Initialize empty lists for latitudes and longitudes
latitudes = []
longitudes = []
capitals = []
countries = []

# Extracting information from the dictionary
for country, capital_info in european_capitals.items():
    capital, coordinates = list(capital_info.items())[0]
    lat, lon = coordinates

    capitals.append(capital)
    latitudes.append(lat)
    longitudes.append(lon)
    countries.append(country)
# Setup the Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=36000)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def historical_data(european_capitals):
    
        # Initialize empty lists for latitudes and longitudes
    latitudes = []
    longitudes = []
    capitals = []
    countries = []

    # Extracting information from the dictionary
    for country, capital_info in european_capitals.items():
        capital, coordinates = list(capital_info.items())[0]
        lat, lon = coordinates

        capitals.append(capital)
        latitudes.append(lat)
        longitudes.append(lon)
        countries.append(country)


    # Define the parameters for the API request
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": "1980-01-01",
        "end_date": "2019-12-31",
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"]
    }

    # Make the API request
    responses = openmeteo.weather_api(url, params=params)

    final_df = pd.DataFrame()

    # Iterate through each response
    for i, response in enumerate(responses):
        # Process hourly data for the current response
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()

        # Create a DataFrame from the hourly data
        historical = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly_temperature_2m,
            "precipitation": hourly_precipitation,
            "wind_speed_10m": hourly_wind_speed_10m,
            "capital": capitals[i],
            "country": countries[i],
            "type": "historical"
        }

        historical_df = pd.DataFrame(data=historical)
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        historical_df['date'] = historical_df['date'].dt.strftime('%m-%d %H:00:00')

        # Concatenate with the final DataFrame
        final_df = pd.concat([final_df, historical_df], ignore_index=True)
        # First, group by 'date' and 'capital' and calculate the mean for numeric columns
        grouped_numeric = final_df.groupby(['date', 'capital']).mean(numeric_only=True).reset_index()

        # Next, create a DataFrame with unique combinations of 'capital', 'country', and 'type'
        unique_capital_info = final_df[['capital', 'country', 'type']].drop_duplicates()

        # Merge the grouped numeric data with the unique capital info
        final_grouped_df = pd.merge(grouped_numeric, unique_capital_info, on='capital', how='left')

    return final_grouped_df

###

def weather_forecast(european_capitals):
    
    latitudes = []
    longitudes = []
    capitals = []
    countries = []

    # Extracting information from the dictionary
    for country, capital_info in european_capitals.items():
        capital, coordinates = list(capital_info.items())[0]
        lat, lon = coordinates

        capitals.append(capital)
        latitudes.append(lat)
        longitudes.append(lon)
        countries.append(country)
        
    # Define the parameters for the API request
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "forecast_days": 14
    }

    # Make the API request
    responses = openmeteo.weather_api(url, params=params)

    final_df = pd.DataFrame()

    # Iterate through each response
    for i, response in enumerate(responses):
        # Process hourly forecast data for the current response
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()

        # Create a DataFrame from the hourly forecast data
        forecast = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly_temperature_2m,
            "precipitation": hourly_precipitation,
            "wind_speed_10m": hourly_wind_speed_10m,
            "capital": capitals[i],
            "country": countries[i],
            "type": "forecast"
        }

        forecast_df = pd.DataFrame(data=forecast)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df['date'] = forecast_df['date'].dt.strftime('%m-%d %H:00:00')

        # Concatenate with the final DataFrame
        final_df = pd.concat([final_df, forecast_df], ignore_index=True)

    return final_df

###

all_data = []
forecast_df = weather_forecast(european_capitals)
historical_df = historical_data(european_capitals)
historical_df = historical_df[historical_df['date'].isin(forecast_df['date'])]

# Concatenate historical and forecast data
combined_df = pd.concat([historical_df, forecast_df])
combined_df['date'] = pd.to_datetime(combined_df['date'], format='%m-%d %H:%M:%S')
combined_df['date'] = combined_df['date'].apply(lambda dt: dt.replace(year=2024))

all_data.append(combined_df)

# Merge all data into a single dataframe
merged_weather_data = pd.concat(all_data)

merged_weather_data['date'] = pd.to_datetime(merged_weather_data['date'], format='%m-%d %H:%M:%S')
merged_weather_data['date'] = merged_weather_data['date'].apply(lambda dt: dt.replace(year=2024))

country_data = merged_weather_data[merged_weather_data['country'] == country]
country_data.groupby(['date', 'type']).agg({
    'temperature_2m': 'mean',
    'precipitation': 'mean',
    'wind_speed_10m': 'mean'
}).reset_index()

grouped = country_data.groupby(['date', 'type']).agg({
    'temperature_2m': 'mean',
    'precipitation': 'mean',
    'wind_speed_10m': 'mean'
}).reset_index()

# Adding 'country' and 'capital' columns with the value 'Europe'
grouped['country'] = 'Europe'
grouped['capital'] = 'Europe'

grouped = grouped[['date', 'capital', 'temperature_2m', 'precipitation', 'wind_speed_10m', 'country', 'type']]

country_data = pd.concat([country_data, grouped], ignore_index=True)

def plot_weather_charts(country):
    # Filter the merged dataframe for the specified country
    historical_country = country_data[country_data['type'] == 'historical']
    forecast_country = country_data[country_data['type'] == 'forecast']
    historical_country = historical_country.sort_values(by='date')
    forecast_country = forecast_country.sort_values(by='date')
    
    charts = []
 # Temperature forecast chart
    fig_temp, ax_temp = plt.subplots()
    ax_temp.plot(historical_country['date'], historical_country['temperature_2m'], label='1990-2020 Historical', color='orange')
    ax_temp.plot(forecast_country['date'], forecast_country['temperature_2m'], label='Forecast', color='blue')
    ax_temp.set_title(f'Temperature in {country}: Historical vs Forecast')
    ax_temp.set_ylabel('Temperature (2m)')
    ax_temp.legend()
    charts.append(fig_temp)

    # Wind speed forecast chart
    fig_wind, ax_wind = plt.subplots()
    ax_wind.plot(historical_country['date'], historical_country['wind_speed_10m'], label='1990-2020 Historical', color='orange')
    ax_wind.plot(forecast_country['date'], forecast_country['wind_speed_10m'], label='Forecast', color='blue')
    ax_wind.set_title(f'Wind Speed in {country}: Historical vs Forecast')
    ax_wind.set_ylabel('Wind Speed (10m)')
    ax_wind.legend()
    charts.append(fig_wind)

    # Precipitation forecast chart
    fig_precip, ax_precip = plt.subplots()
    ax_precip.plot(historical_country['date'], historical_country['precipitation'], label='1990-2020 Historical', color='orange')
    ax_precip.plot(forecast_country['date'], forecast_country['precipitation'], label='Forecast', color='blue')
    ax_precip.set_title(f'Precipitation in {country}: Historical vs Forecast')
    ax_precip.set_ylabel('Precipitation')
    ax_precip.legend()
    charts.append(fig_precip)

    # Improve date formatting
    for ax in [ax_temp, ax_wind, ax_precip]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    return charts

# Group by 'Country' and 'Type' and then calculate the mean
grouped_data = merged_weather_data.groupby(['country', 'type']).mean(numeric_only=True).reset_index()

# Pivot this grouped data to have 'Historical' and 'Forecast' as columns
pivot_df = grouped_data.pivot(index='country', columns='type', values=['temperature_2m', 'wind_speed_10m', 'precipitation'])

# Calculate the differences
pivot_df['temperature_diff'] = pivot_df['temperature_2m']['forecast'] - pivot_df['temperature_2m']['historical']
pivot_df['wind_diff'] = pivot_df['wind_speed_10m']['forecast'] - pivot_df['wind_speed_10m']['historical']
pivot_df['precipitation_diff'] = pivot_df['precipitation']['forecast'] - pivot_df['precipitation']['historical']
pivot_df['precipitation_diff'] *= 10
# Reset index to make 'Country' a column again and filter only the necessary columns
diff_df = pivot_df.reset_index()[['country', 'temperature_diff', 'wind_diff', 'precipitation_diff']]

diff_df.columns = ['country', 'temperature_diff', 'wind_diff', 'precipitation_diff']  # Flatten the MultiIndex columns
Final_diff = diff_df.copy()

eu_temp = Final_diff['temperature_diff'].mean()
eu_wind = Final_diff['wind_diff'].mean()
eu_rain = Final_diff['precipitation_diff'].mean()

Final_diff.rename(columns={
    "temperature_diff": "Temperature Difference C",
    "wind_diff": "Wind Difference km/h",
    "precipitation_diff": "Precipitation Difference mm"
}, inplace=True)

# Create a new DataFrame with the averages
new_row = pd.DataFrame({
    'country': ['Europe'],
    'Temperature Difference C': [eu_temp],
    'Wind Difference km/h': [eu_wind],
    'Precipitation Difference mm': [eu_rain]
})

Final_diff = pd.concat([Final_diff, new_row], ignore_index=True)
Final_diff = Final_diff.round(1)


###

europe = europe.rename(columns={'NAME_ENGL': 'country'})

merged_europe_df = europe.merge(diff_df, left_on='country', right_on='country', how='left')
cols_to_replace_nan = ['temp_col', 'wind_col', 'precipitation_col']

###


# Plotting the map
# fig, ax = plt.subplots(figsize=(22, 18))
# merged_europe_df.plot(ax=ax, color=merged_europe_df['temp_col'])
# ax.set_xlim(-23, 19)  # Longitudes for Europe roughly range from -10 to 30
# ax.set_ylim(35, 68)   # Latitudes for Europe roughly range from 34 to 72

# # Annotate each country with its 'data_value' if the value is not NaN
# for idx, row in merged_europe_df.iterrows():
#     # Check if 'data_value' is NaN
#     if not pd.isna(row['temperature_diff']):
#         # Get the centroid of the country polygon
#         centroid = row.geometry.centroid
#         # Annotate the plot with the 'data_value'
#         plt.annotate(text=f"{row['temperature_diff']:.1f}", xy=(centroid.x, centroid.y),
#                      xytext=(3, 3), textcoords="offset points",
#                      ha='center', va='center', fontsize=15, color='white', 
#                      transform=ccrs.PlateCarree())

# # Turn off the axis display
# ax.set_axis_off()

# plt.show()

###
# Getting things automated into slack

# client = WebClient(token="MWNlY2M5OGItNTg5Ny00YmI5LTgyMWQtNDczM2VhZDVjNDg4")

# url_temp = 'http://wxmaps.org/pix/temp4.png'
# url_precip = 'http://wxmaps.org/pix/prec4.png'

# plot_weather_charts("Germany")
# Final_diff
# Initialize Dash app

# Set up the Streamlit layout

chart_dict = {}
country = "Europe"
charts = plot_weather_charts(country)
chart_dict[country] = {
    'Temperature': charts[0],
    'Wind': charts[1],
    'Precipitation': charts[2],
}

st.title('Weather-App Dashboard')

yesterday = datetime.date.today() - datetime.timedelta(days=1)
yesterday_str = yesterday.strftime("%Y%m%d")

# Format the URLs with today's date
image_url1 = f'https://www.tropicaltidbits.com/analysis/models/gfs/{yesterday_str}06/gfs_T2ma_eu_1.png'
image_url2 = f'https://www.tropicaltidbits.com/analysis/models/gfs/{yesterday_str}06/gfs_mslp_pwata_eu_2.png'

# Display images in columns
col1, col2 = st.columns(2)
with col1:
    st.image(image_url1, caption='Temperature Forecasts', use_column_width=True)

with col2:
    st.image(image_url2, caption='Precipitation Forecasts', use_column_width=True)

# Assume Final_diff is defined earlier in the script
st.dataframe(Final_diff, use_container_width=True)

# Cache all charts for all countries
all_charts = chart_dict

st.plotly_chart(all_charts[country]['Temperature'], use_container_width=True)

st.plotly_chart(all_charts[country]['Wind'], use_container_width=True)

st.plotly_chart(all_charts[country]['Precipitation'], use_container_width=True)
