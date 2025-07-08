import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("insurance.csv")

# Feature Engineering
df["bmi"] = df["weight"] / (df["height"] ** 2)

def age_group(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"

df["age_group"] = df["age"].apply(age_group)

def lifestyle_risk(row):
    if row["smoker"] and row["bmi"] > 30:
        return "high"
    elif row["smoker"] or row["bmi"] > 27:
        return "medium"
    else:
        return "low"

df["lifestyle_risk"] = df.apply(lifestyle_risk, axis=1)

tier_1 = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2 = ["Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
          "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
          "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
          "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
          "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
          "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"]

def city_tier(city):
    if city in tier_1:
        return 1
    elif city in tier_2:
        return 2
    else:
        return 3

df["city_tier"] = df["city"].apply(city_tier)

# Final features
df_feat = df[['income_lpa', 'occupation', 'bmi', 'age_group', 'lifestyle_risk', 'city_tier', 'insurance_premium_category']]

X = df_feat.drop(columns=["insurance_premium_category"])
y = df_feat["insurance_premium_category"]

# Preprocessing
categorical = ["age_group", "lifestyle_risk", "occupation", "city_tier"]
numerical = ["bmi", "income_lpa"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
pipeline.fit(X_train, y_train)

# Accuracy (Optional)
print("Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
