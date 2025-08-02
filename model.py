import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("delivery_delay_data_1000.csv")

# Encode categorical features
le_vendor = LabelEncoder()
le_shipping = LabelEncoder()
le_weather = LabelEncoder()
le_delay = LabelEncoder()

df["Vendor_enc"] = le_vendor.fit_transform(df["Vendor"])
df["Shipping_enc"] = le_shipping.fit_transform(df["Shipping Mode"])
df["Weather_enc"] = le_weather.fit_transform(df["Weather"])
df["Delay_enc"] = le_delay.fit_transform(df["Delay (Yes/No)"])

# Features and labels
X = df[["Distance (km)", "Vendor_enc", "Shipping_enc", "Weather_enc"]]
y = df["Delay_enc"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "delay_model.pkl")
joblib.dump(le_vendor, "le_vendor.pkl")
joblib.dump(le_shipping, "le_shipping.pkl")
joblib.dump(le_weather, "le_weather.pkl")
joblib.dump(le_delay, "le_delay.pkl")

print("✅ Model and encoders saved successfully.")
