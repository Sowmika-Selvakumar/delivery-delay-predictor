import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load & preprocess data
# -----------------------------
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("delivery_delay_data_1000.csv")
    return df

def preprocess_data(df):
    le_vendor = LabelEncoder()
    le_shipping = LabelEncoder()
    le_weather = LabelEncoder()
    le_delay = LabelEncoder()
    
    df["Vendor_enc"] = le_vendor.fit_transform(df["Vendor"])
    df["Shipping_enc"] = le_shipping.fit_transform(df["Shipping Mode"])
    df["Weather_enc"] = le_weather.fit_transform(df["Weather"])
    df["Delay_enc"] = le_delay.fit_transform(df["Delay (Yes/No)"])
    
    return df, le_vendor, le_shipping, le_weather, le_delay

# Load and preprocess
df = load_data()
df, le_vendor, le_shipping, le_weather, le_delay = preprocess_data(df)

# -----------------------------
# Train the model
# -----------------------------
X = df[["Distance (km)", "Vendor_enc", "Shipping_enc", "Weather_enc"]]
y = df["Delay_enc"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚚 Delivery Delay Predictor")

st.markdown("Upload order data or use the manual input below to predict delivery delays.")

# --- File Upload Section
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())

        data["Vendor_enc"] = le_vendor.transform(data["Vendor"])
        data["Shipping_enc"] = le_shipping.transform(data["Shipping Mode"])
        data["Weather_enc"] = le_weather.transform(data["Weather"])
        
        X_uploaded = data[["Distance (km)", "Vendor_enc", "Shipping_enc", "Weather_enc"]]
        predictions = model.predict(X_uploaded)
        data["Predicted Delay"] = le_delay.inverse_transform(predictions)
        st.success("✅ Prediction complete!")
        st.write(data[["Order ID", "Predicted Delay"]])
    except Exception as e:
        st.error(f"Error: {e}")

# --- Manual Input Section
st.subheader("📌 Manual Prediction")

distance = st.number_input("Distance (km)", min_value=1, max_value=1000, value=120)
vendor = st.selectbox("Vendor", ["Vendor A", "Vendor B", "Vendor C"])
shipping_mode = st.selectbox("Shipping Mode", ["Standard", "Express", "Same Day"])
weather = st.selectbox("Weather", ["Clear", "Rainy", "Stormy", "Foggy"])

if st.button("Predict Delay"):
    input_data = np.array([[distance,
                            le_vendor.transform([vendor])[0],
                            le_shipping.transform([shipping_mode])[0],
                            le_weather.transform([weather])[0]]])
    result = model.predict(input_data)
    prediction = le_delay.inverse_transform(result)[0]
    st.success(f"📦 Predicted Delivery Status: **{prediction}**")

# -----------------------------
# 📊 Visual Charts
# -----------------------------
st.subheader("📊 Delay Trend Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Delays by Shipping Mode**")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Shipping Mode", hue="Delay (Yes/No)", palette="Set2", ax=ax1)
    ax1.set_title("Shipping Mode vs Delay")
    st.pyplot(fig1)

with col2:
    st.markdown("**Delays by Weather**")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="Weather", hue="Delay (Yes/No)", palette="coolwarm", ax=ax2)
    ax2.set_title("Weather vs Delay")
    st.pyplot(fig2)

st.markdown("**Delays by Vendor**")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="Vendor", hue="Delay (Yes/No)", palette="pastel", ax=ax3)
ax3.set_title("Vendor vs Delay")
st.pyplot(fig3)
