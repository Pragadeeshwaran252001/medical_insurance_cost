import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import preprocess_data

# Load preprocessed data
df = preprocess_data(r"D:\GUVI_PROJECT_3\PROJECT_3\medical_insurance.csv")

# ================================
# 1. UNIVARIATE ANALYSIS
# ================================
def univariate_analysis():
    # 1.1 Distribution of medical insurance charges
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True)
    plt.title("Distribution of Insurance Charges")
    plt.xlabel("Charges")
    plt.ylabel("Frequency")
    plt.show()

    # 1.2 Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title("Age Distribution of Individuals")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    # 1.3 Smoker vs non-smoker count
    plt.figure(figsize=(6, 4))
    sns.countplot(x='smoker', data=df)
    plt.title("Smoker vs Non-Smoker Count")
    plt.xlabel("Smoker (1 = Yes, 0 = No)")
    plt.ylabel("Count")
    plt.show()

    # 1.4 Average BMI
    avg_bmi = df['bmi'].mean()
    print(f"\nAverage BMI in the dataset: {avg_bmi:.2f}")

    # 1.5 Region-wise policyholder count
    plt.figure(figsize=(8, 4))
    sns.countplot(x='region', data=df)
    plt.title("Number of Policyholders by Region (Encoded)")
    plt.xlabel("Region Code (0-3)")
    plt.ylabel("Count")
    plt.show()

# ================================
# 2. BIVARIATE ANALYSIS
# ================================
def bivariate_analysis():
    # 2.1 Charges vs Age
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', data=df)
    plt.title("Insurance Charges vs Age")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.show()

    # 2.2 Charges by Smoking Status
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='smoker', y='charges', data=df)
    plt.title("Average Charges: Smokers vs Non-Smokers")
    plt.xlabel("Smoker")
    plt.ylabel("Charges")
    plt.show()

    # 2.3 Charges vs BMI
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='charges', data=df)
    plt.title("Charges vs BMI")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.show()

    # 2.4 Charges by Gender
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='sex', y='charges', data=df)
    plt.title("Charges by Gender (0 = Female, 1 = Male)")
    plt.xlabel("Sex")
    plt.ylabel("Charges")
    plt.show()

    # 2.5 Children vs Charges
    plt.figure(figsize=(10, 6))
    sns.barplot(x='children', y='charges', data=df)
    plt.title("Average Charges by Number of Children")
    plt.xlabel("Number of Children")
    plt.ylabel("Average Charges")
    plt.show()

# ================================
# 3. MULTIVARIATE ANALYSIS
# ================================
def multivariate_analysis():
    # 3.1 Age vs Charges by Smoking Status
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
    plt.title("Age vs Charges by Smoking Status")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.show()

    # 3.2 Gender & Region Impact for Smokers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='region', y='charges', hue='sex', data=df[df['smoker'] == 1])
    plt.title("Charges by Gender & Region (Only Smokers)")
    plt.xlabel("Region")
    plt.ylabel("Charges")
    plt.show()

    # 3.3 Combined Impact: Age, BMI, Smoking
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', size='age', data=df)
    plt.title("Charges by BMI, Age, and Smoking Status")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.show()

    # 3.4 Obese Smokers vs Non-Obese Non-Smokers
    obese_smokers = df[(df['bmi'] > 30) & (df['smoker'] == 1)]
    non_obese_nonsmokers = df[(df['bmi'] <= 30) & (df['smoker'] == 0)]

    print(f"\nAverage charges of obese smokers: {obese_smokers['charges'].mean():.2f}")
    print(f"Average charges of non-obese non-smokers: {non_obese_nonsmokers['charges'].mean():.2f}")

# ================================
# 4. OUTLIER DETECTION
# ================================
def detect_outliers():
    # Charges outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['charges'])
    plt.title("Outliers in Charges")
    plt.xlabel("Charges")
    plt.show()

    print("\nTop 5 most expensive individuals:")
    print(df.sort_values(by="charges", ascending=False).head())

    # BMI outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['bmi'])
    plt.title("Outliers in BMI")
    plt.xlabel("BMI")
    plt.show()

# ================================
# 5. CORRELATION ANALYSIS
# ================================
def correlation_analysis():
    correlation = df[['age', 'bmi', 'children', 'charges']].corr()
    print("\nCorrelation Matrix:\n", correlation)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between Numeric Features")
    plt.show()

# ================================
# Run all EDA sections
# ================================
if __name__ == "__main__":
    univariate_analysis()
    bivariate_analysis()
    multivariate_analysis()
    detect_outliers()
    correlation_analysis()
