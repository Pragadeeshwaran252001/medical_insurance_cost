import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Introduction", "📊 EDA", "📈 Prediction"])

# === Introduction Page ===
if page == "🏠 Introduction":
    st.title("🏥 Medical Insurance Cost Prediction")
    st.image(r"D:\GUVI_PROJECT_3\medical insurance.jpg",  use_container_width=True)
    st.markdown("""
    This project aims to estimate **medical insurance charges** based on health and demographic attributes.
    
    ### About:
    - Predict insurance cost using machine learning models.
    - Analyze how factors like **age**, **BMI**, **smoking**, and **region** influence cost.
    - Explore data trends and relationships via EDA.

    ### Technologies Used :
    - Python, Pandas, Streamlit
    - Scikit-learn, XGBoost
    - MLflow for model tracking
    """)


# === EDA Page ===
elif page == "📊 EDA":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title("📊 Exploratory Data Analysis")

    # Load dataset
    df = pd.read_csv(r"D:\GUVI_PROJECT_3\PROJECT_3\medical_insurance.csv")

    # Drop non-numeric if exists
    if 'bmi_class' in df.columns:
        df.drop(columns=['bmi_class'], inplace=True)

    # First dropdown - EDA Category
    eda_type = st.selectbox("Select EDA Category", [
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis",
        "Outlier Detection",
        "Correlation Analysis"
    ])

    # Define questions for each EDA type
    eda_questions = {
        "Univariate Analysis": [
            "Distribution of Insurance Charges",
            "Age Distribution",
            "Smoker Count",
            "Average BMI",
            "Region-wise Policyholders"
        ],
        "Bivariate Analysis": [
            "Charges vs Age",
            "Charges for Smokers vs Non-Smokers",
            "BMI vs Charges",
            "Gender vs Charges",
            "Children vs Charges"
        ],
        "Multivariate Analysis": [
            "Smoking + Age vs Charges",
            "Gender + Region for Smokers",
            "Age + BMI + Smoking Impact",
            "Obese Smokers vs Non-Obese Non-Smokers"
        ],
        "Outlier Detection": [
            "High Insurance Charges Outliers",
            "Extreme BMI Outliers"
        ],
        "Correlation Analysis": [
            "Correlation Matrix",
            "Feature vs Charges Correlation"
        ]
    }

    selected_question = st.selectbox("Select Question", eda_questions[eda_type])

    # Plotting logic
    st.subheader(f"🔍 {selected_question}")
    fig, ax = plt.subplots()

    if selected_question == "Distribution of Insurance Charges":
        sns.histplot(df['charges'], kde=True, ax=ax)
        ax.set_title("Distribution of Charges")

    elif selected_question == "Age Distribution":
        sns.histplot(df['age'], bins=20, kde=True, ax=ax)
        ax.set_title("Age Distribution")

    elif selected_question == "Smoker Count":
        sns.countplot(x='smoker', data=df, ax=ax)
        ax.set_title("Smoker vs Non-Smoker Count")

    elif selected_question == "Average BMI":
        sns.histplot(df['bmi'], bins=20, kde=True, ax=ax)
        ax.set_title("BMI Distribution")

    elif selected_question == "Region-wise Policyholders":
        sns.countplot(x='region', data=df, ax=ax)
        ax.set_title("Policyholders by Region")

    elif selected_question == "Charges vs Age":
        sns.scatterplot(x='age', y='charges', data=df, ax=ax)
        ax.set_title("Charges vs Age")

    elif selected_question == "Charges for Smokers vs Non-Smokers":
        sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
        ax.set_title("Charges by Smoking Status")

    elif selected_question == "BMI vs Charges":
        sns.scatterplot(x='bmi', y='charges', data=df, ax=ax)
        ax.set_title("BMI vs Charges")

    elif selected_question == "Gender vs Charges":
        sns.boxplot(x='sex', y='charges', data=df, ax=ax)
        ax.set_title("Charges by Gender")

    elif selected_question == "Children vs Charges":
        sns.boxplot(x='children', y='charges', data=df, ax=ax)
        ax.set_title("Charges by Number of Children")

    elif selected_question == "Smoking + Age vs Charges":
        sns.scatterplot(x='age', y='charges', hue='smoker', data=df, ax=ax)
        ax.set_title("Age and Smoking Status vs Charges")

    elif selected_question == "Gender + Region for Smokers":
        smokers_df = df[df['smoker'] == 'yes']
        sns.boxplot(x='region', y='charges', hue='sex', data=smokers_df, ax=ax)
        ax.set_title("Smoker Charges by Gender and Region")

    elif selected_question == "Age + BMI + Smoking Impact":
        sns.scatterplot(x='age', y='charges', hue='bmi', size='smoker', data=df, ax=ax)
        ax.set_title("Age, BMI, Smoking Impact on Charges")

    elif selected_question == "Obese Smokers vs Non-Obese Non-Smokers":
        df['obese'] = df['bmi'] > 30
        condition = (df['obese']) & (df['smoker'] == 'yes')
        group = ['Obese Smoker' if cond else 'Others' for cond in condition]
        sns.boxplot(x=group, y=df['charges'], ax=ax)
        ax.set_title("Charges: Obese Smokers vs Others")

    elif selected_question == "High Insurance Charges Outliers":
        sns.boxplot(y='charges', data=df, ax=ax)
        ax.set_title("Outliers in Charges")

    elif selected_question == "Extreme BMI Outliers":
        sns.boxplot(y='bmi', data=df, ax=ax)
        ax.set_title("Outliers in BMI")

    elif selected_question == "Correlation Matrix":
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")

    elif selected_question == "Feature vs Charges Correlation":
        corr = df.corr(numeric_only=True)['charges'].sort_values(ascending=False)
        st.write(corr)

    st.pyplot(fig)


# === Prediction Page ===
elif page == "📈 Prediction":
    st.title("💡 Predict Medical Insurance Charges")
    st.markdown("### 🧑‍⚕️ Enter User Details")

    # Input Fields
    sex = st.selectbox("Gender", ("Male", "Female"))
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    
    # Use number_input instead of slider/selectbox
    children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
    
    smoker = st.selectbox("Do you Smoke?", ("Yes", "No"))
    region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    # Encode inputs
    sex_encoded = 1 if sex == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0
    region_encoded = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}[region]

    # Input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_encoded],
        "bmi": [bmi],
        "children": [int(children)],
        "smoker": [smoker_encoded],
        "region": [region_encoded]
    })

    # Load best model
    best_run_id = "834b342534454766a5c768242c305952"
    model_name = "DecisionTree"
    model_uri = f"runs:/{best_run_id}/{model_name}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        st.error(f"Failed to load model from MLflow: {e}")
        st.stop()

    if st.button("Predict Insurance Cost"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Insurance Charges: ${prediction:,.2f}")
        st.info("✅ Model Used: Decision Tree (Best R² Score)")
