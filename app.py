import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

# Function to load the trained model
def load_model():
    return joblib.load("model.joblib")

# Function to load label encoders
def load_label_encoders():
    label_encoders = {}
    for feature in ['City', 'Customer_type', 'Gender', 'Product line']:
        label_encoders[feature] = joblib.load(f"{feature}_encoder.joblib")
    return label_encoders

# ---- READ EXCEL ----
@st.cache_data
def get_data_from_excel():
    df = pd.read_excel(
        io="supermarkt_sales.xlsx",
        engine="openpyxl",
        sheet_name="Sales",
        skiprows=3,  # Adjusted skiprows to start from the header row
        usecols="A:Q",  # Selecting the correct columns containing 'City', 'Customer_type', 'Gender', 'Product line', 'Rating'
        nrows=1000,
    )
    return df

df = get_data_from_excel()

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
city = st.sidebar.selectbox(
    "Select the City:",
    options=df["City"].unique(),
    index=0  # Set default index to select the first option
)

customer_type = st.sidebar.selectbox(
    "Select the Customer Type:",
    options=df["Customer_type"].unique(),
    index=0
)

gender = st.sidebar.selectbox(
    "Select the Gender:",
    options=df["Gender"].unique(),
    index=0
)

# Load label encoders
label_encoders = load_label_encoders()

# Transform selected values using label encoders
city_pred = label_encoders['City'].transform([city])[0]
customer_type_pred = label_encoders['Customer_type'].transform([customer_type])[0]
gender_pred = label_encoders['Gender'].transform([gender])[0]

df_selection = df.query(
    "City == @city & Customer_type == @customer_type & Gender == @gender"
)

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.

# ---- MAINPAGE ----
st.title(":bar_chart: Sales Dashboard - LCO21380 Deepinder Singh")
st.markdown("##")

# TOP KPI's
total_sales = int(df_selection["Total"].sum())
average_rating = round(df_selection["Rating"].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))
average_sale_by_transaction = round(df_selection["Total"].mean(), 2)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")
with middle_column:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")
with right_column:
    st.subheader("Average Sales Per Transaction:")
    st.subheader(f"US $ {average_sale_by_transaction}")

st.markdown("""---""")

# SALES BY PRODUCT LINE [BAR CHART]
sales_by_product_line = df_selection.groupby(by=["Product line"])[["Total"]].sum().sort_values(by="Total")
fig_product_sales = px.bar(
    sales_by_product_line,
    x="Total",
    y=sales_by_product_line.index,
    orientation="h",
    title="<b>Sales by Product Line</b>",
    color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
    template="plotly_white",
)
fig_product_sales.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

# Add 'hour' column to dataframe
df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
df_selection["hour"] = df["hour"]

# SALES BY HOUR [BAR CHART]
sales_by_hour = df_selection.groupby(by=["hour"])[["Total"]].sum()
fig_hourly_sales = px.bar(
    sales_by_hour,
    x=sales_by_hour.index,
    y="Total",
    title="<b>Sales by hour</b>",
    color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
    template="plotly_white",
)
fig_hourly_sales.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
right_column.plotly_chart(fig_product_sales, use_container_width=True)

# Sales Prediction Section
st.markdown("---")
st.title("Sales Prediction")

# Load the model
model = load_model()

# Dropdown for product line (continuous feature)
product_line_pred = st.selectbox("Product Line:", options=df['Product line'].unique())

# Transform product line using label encoder
product_line_pred_encoded = label_encoders['Product line'].transform([product_line_pred])[0]

rating_pred = st.slider("Rating:", min_value=df['Rating'].min(), max_value=df['Rating'].max(), value=df['Rating'].mean())

# Prepare input for prediction
input_data = pd.DataFrame({
    'City': [city_pred],
    'Customer_type': [customer_type_pred],
    'Gender': [gender_pred],
    'Product line': [product_line_pred_encoded],
    'Rating': [rating_pred]
})

# Make prediction
prediction = model.predict(input_data)

# Display prediction
st.subheader("Predicted Total Sales:")
st.write(f"US $ {prediction[0]:,.2f}")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
