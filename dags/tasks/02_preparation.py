
# 02 - Data Preparation

from airflow.decorators import task
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)


@task
def prepare_data(data_path):

    # Load dataset
    df = pd.read_csv(data_path)

    # Separate X and y
    X = df.drop(
        columns=["Student_ID", "Mental_Health_Risk"]
    )

    y = df["Mental_Health_Risk"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # Define Features
    numerical_features = [
        "Age",
        "Year_of_Study",
        "Sleep_Hours",
        "Study_Hours_Per_Week",
        "GPA",
        "Social_Media_Hours",
        "Screen_Time_Hours",
        "Financial_Stress",
        "Family_Support",
        "Attendance_Rate"
    ]

    categorical_features = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]

    ordinal_features = [
        "Physical_Activity",
        "Diet_Quality"
    ]

    # Check Outliers using IQR
    print("\nOutliers:")

    for col in numerical_features:

        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)

        IQR = Q3 - Q1

        outliers = X_train[
            (X_train[col] < Q1 - 1.5 * IQR) |
            (X_train[col] > Q3 + 1.5 * IQR)
        ]

        print(col, ":", len(outliers))

    # Numerical Pipeline
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Ordinal Pipeline
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # Combine Pipelines
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
        ("ord", ordinal_pipeline, ordinal_features)
    ])

    # Save prepared data
    project_root = Path(__file__).resolve().parents[1]

    state_dir = project_root / "data" / "airflow_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    output_path = state_dir / "prepared_data.joblib"

    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor
    }, output_path)

    print("\nData Preparation Completed!")

    return str(output_path)
