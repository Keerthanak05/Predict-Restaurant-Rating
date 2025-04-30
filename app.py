# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_splits
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

st.title("🍽️ Restaurant Rating Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.write("### Data Preview", df.head())

    # Drop unnecessary columns
    drop_columns = ["Restaurant ID", "Restaurant Name", "Address", "Locality",
                    "Locality Verbose", "Longitude", "Latitude"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Define target
    target = "Aggregate rating"
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataset.")
    else:
        # Encode categorical
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if target in categorical_cols:
            categorical_cols.remove(target)
        df = pd.get_dummies(df, columns=categorical_cols)

        # Split features and labels
        X = df.drop(columns=[target])
        y = df[target]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R-squared (R²):** {r2:.4f}")

        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("🔥 Top 10 Feature Importances")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10), ax=ax)
        ax.set_title("Top 10 Features Influencing Ratings")
        st.pyplot(fig)
