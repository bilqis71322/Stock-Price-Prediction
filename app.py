import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict the next closing price using the previous 60 trading days.")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(
        "all_stocks_5yr.csv",
        delimiter=",",
        on_bad_lines="skip"
    )

    data["date"] = pd.to_datetime(data["date"])

    return data


data = load_data()

# -----------------------------
# Company selection
# -----------------------------
companies = sorted(data["Name"].unique())

company = st.selectbox(
    "Select a company",
    companies,
    index=companies.index("AAPL") if "AAPL" in companies else 0
)

st.write("Selected company:", company)

# -----------------------------
# Filter company
# -----------------------------
company_data = data[data["Name"] == company].copy()

company_data = company_data.sort_values("date")

# Closing prices
prices = company_data["close"].values.reshape(-1, 1)

# -----------------------------
# Display historical data
# -----------------------------
st.subheader("Historical Closing Price")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    company_data["date"],
    company_data["close"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.set_title(f"{company} Closing Price")

st.pyplot(fig)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔮 Predict Next Closing Price"):

    with st.spinner("Training LSTM model and generating prediction..."):

        # -----------------------------
        # Scale data
        # -----------------------------
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_prices = scaler.fit_transform(prices)

        # -----------------------------
        # Create training sequences
        # -----------------------------
        x_train = []
        y_train = []

        for i in range(60, len(scaled_prices)):
            x_train.append(scaled_prices[i-60:i, 0])
            y_train.append(scaled_prices[i, 0])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Reshape for LSTM
        x_train = np.reshape(
            x_train,
            (x_train.shape[0], x_train.shape[1], 1)
        )

        # -----------------------------
        # Build LSTM model
        # -----------------------------
        model = keras.models.Sequential()

        model.add(
            keras.layers.LSTM(
                units=64,
                return_sequences=True,
                input_shape=(x_train.shape[1], 1)
            )
        )

        model.add(
            keras.layers.LSTM(units=64)
        )

        model.add(
            keras.layers.Dense(32)
        )

        model.add(
            keras.layers.Dropout(0.5)
        )

        model.add(
            keras.layers.Dense(1)
        )

        # -----------------------------
        # Compile model
        # -----------------------------
        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

        # -----------------------------
        # Train model
        # -----------------------------
        model.fit(
            x_train,
            y_train,
            epochs=10,
            verbose=0
        )

        # -----------------------------
        # Last 60 days
        # -----------------------------
        last_60_days = scaled_prices[-60:]

        x_test = np.array([last_60_days])

        x_test = np.reshape(
            x_test,
            (x_test.shape[0], x_test.shape[1], 1)
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(
            x_test,
            verbose=0
        )

        prediction = scaler.inverse_transform(
            prediction
        )

        predicted_price = prediction[0][0]

    # -----------------------------
    # Display result
    # -----------------------------
    st.success("Prediction completed!")

    st.metric(
        label=f"Predicted Next Closing Price ({company})",
        value=f"${predicted_price:.2f}"
    )

    # -----------------------------
    # Show last 60 days
    # -----------------------------
    st.subheader("Last 60 Trading Days")

    last_60_data = company_data.tail(60)

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        last_60_data["date"],
        last_60_data["close"],
        marker="o"
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Closing Price")
    ax2.set_title(
        f"{company} - Last 60 Trading Days"
    )

    st.pyplot(fig2)

    st.info(
        "This prediction is based only on historical closing prices "
        "and should not be considered financial advice."
    )

# -----------------------------
# Dataset information
# -----------------------------
st.sidebar.header("Dataset Information")

st.sidebar.write(
    f"Total records: {len(data):,}"
)

st.sidebar.write(
    f"Total companies: {data['Name'].nunique()}"
)

st.sidebar.write(
    f"Selected company: {company}"
)