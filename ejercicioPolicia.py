#%%
'''
from dash import Dash, html
app = Dash(__name__)

app.layout = html.Div([html.Div(children='Holi caracoli')])

if __name__ == '__main__':
    app.run(debug=True)
    
'''
# %%
import os
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from datetime import date, datetime

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="SF Ridesharing Demo", page_icon=":oncoming_police_car:") #or :police_car:

# LOAD DATA ONCE
@st.cache_resource
def load_data():
    path = "data.csv"
    if not os.path.isfile(path):
        path = f"C:\\tmp\\ai_3002\\{path}"

    data = pd.read_csv(
        path,
        nrows=100000,  # approx. 10% of data
        names=[
            "date/time",
            "lat",
            "lon",
        ],  # specify names directly since they don't change
        skiprows=1,  # don't read header since names specified directly
        usecols=[0, 1, 2],  # doesn't load last column, constant value "B02512"
        parse_dates=[
            "date/time"
        ],  # set as datetime instead of converting after the fact
    )

    return data

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the CSV file
csv_filename = "data.csv"
csv_path = os.path.join(script_dir, csv_filename)

@st.cache_resource
def load_data():
    data = pd.read_csv(
        csv_path,
        nrows=100000,
        usecols=["Incident Date", "Longitude", "Latitude"],
        parse_dates=["Incident Date"],
    )
    print(data)
    return data

# FUNCTION FOR SAN FRANCISCO'S COORDINATES
def map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["Longitude", "Latitude"],
                    radius=100,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )

# FILTER DATA FOR A SPECIFIC DATE, CACHE
@st.cache_resource
def filter_data(df, date_selected):
    # Convert date_selected to a datetime.date object
    if isinstance(date_selected, int):
        try:
            date_selected = date.fromordinal(date_selected)
        except ValueError:
            st.warning("Invalid date selected. Please choose a valid date.")
            return df  # Return the original dataframe if date is invalid

    return df[df["Incident Date"].dt.date == date_selected]

# CALCULATE MIDPOINT FOR GIVEN SET OF DATA
@st.cache_data
def calculate_midpoint(lat, lon):
    return np.average(lat), np.average(lon)

# FILTER DATA BY DATE
@st.cache_data
def calculate_histogram(df, hora):
    # Convert date_selected to a datetime.date object
    #date_selected = datetime.from_ordinal(date_selected.toordinal())
    
    filtered = df[
        (df["Incident Datetime"].dt.date >= hora)
        & (df["Incident Datetime"].dt.date < (hora + pd.Timedelta(days=1)))
    ]

    hist = np.histogram(filtered["Incident Datetime"].dt.hour, bins=24, range=(0, 24))[0]

    return pd.DataFrame({"hour": range(24), "incidents": hist})

# STREAMLIT APP LAYOUT
data = load_data()

# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2, 3))

# SEE IF THERE'S A QUERY PARAM IN THE URL (e.g. ?pickup_hour=2)
# THIS ALLOWS YOU TO PASS A STATEFUL URL TO SOMEONE WITH A SPECIFIC HOUR SELECTED,
# E.G. https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/main?pickup_hour=2
if not st.session_state.get("url_synced", False):
    try:
        pickup_date = int(st.experimental_get_query_params()["pickup_date"][0])
        st.session_state["pickup_date"] = pickup_date
        st.session_state["url_synced"] = True
    except KeyError:
        pass

# IF THE SLIDER CHANGES, UPDATE THE QUERY PARAM
def update_query_params():
    date_selected = st.session_state["pickup_date"]
    st.experimental_set_query_params(pickup_date=date_selected)

with row1_1:
    st.title("SF Crime Location Data")
    hora = st.slider(
        "Select date of crime", 0, 24, key="pickup_date", on_change=update_query_params
    )

with row1_2:
    st.write(
        """
    ##
    Examining how SF Crime locations are identified through time.
    By sliding the slider on the left, you can view different slices of dates and explore different crime trends.
    """
    )

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row2_1, row2_2 = st.columns((2, 1))

# SETTING THE ZOOM LOCATIONS FOR SANTA CLARA
santa_clara = [37.354107, -121.955238]
zoom_level = 12
midpoint = calculate_midpoint(data["Latitude"], data["Longitude"])

with row2_1:
    st.write(f"*All San Francisco City from {hora} to {(pd.to_datetime(hora) + pd.Timedelta(days=1)).date()}*")
    map(filter_data(data, hora), midpoint[0], midpoint[1], 11)

with row2_2:
    st.write("*Santa Clara*")
    map(filter_data(data, hora), santa_clara[0], santa_clara[1], zoom_level)

# CALCULATING DATA FOR THE HISTOGRAM
histogram_data = calculate_histogram(data, hora)

# LAYING OUT THE HISTOGRAM SECTION
st.write(f"*Breakdown of crimes between {hora} and {(pd.to_datetime(hora) + pd.Timedelta(days=1)).date()}*")

st.altair_chart(
    alt.Chart(histogram_data)
    .mark_area(
        interpolate="step-after",
    )
    .encode(
        x=alt.X("hour:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("incidents:Q"),
        tooltip=["hour", "incidents"],
    )
    .configure_mark(opacity=0.2, color="red"),
    use_container_width=True,
)