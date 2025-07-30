import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    print("Initial Data Shape:", df.shape)

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])        # male=1, female=0
    df['smoker'] = le.fit_transform(df['smoker'])  # yes=1, no=0
    df['region'] = le.fit_transform(df['region'])  # regions into 0-3

    # Optional: Add BMI category
    df['bmi_class'] = pd.cut(df['bmi'],
                             bins=[0, 18.5, 25, 30, 100],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    print("Cleaned Data Sample:")
    print(df.head())

    return df

# For testing:
if __name__ == "__main__":
    file_path = r"D:\Guvi_Project3\Dataset\medical_insurance.csv"
    df = preprocess_data(file_path)



