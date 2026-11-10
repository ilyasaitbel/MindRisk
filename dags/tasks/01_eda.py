
# Exploratory Data Analysis - MindRisk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from airflow.decorators import task


@task
def run_eda():

    # Load dataset
    df = pd.read_csv("/opt/airflow/data/student_mental_health.csv")

    target = "Mental_Health_Risk"

    # Dataset information
    print("Dimensions:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    df.info()

    # Missing values
    missing_values = df.isnull().sum()

    print("\nMissing Values:")
    print(missing_values)

    # Duplicates
    print("\nDuplicated Rows:", df.duplicated().sum())

    if "Student_ID" in df.columns:
        print(
            "Duplicated IDs:",
            df["Student_ID"].duplicated().sum()
        )

    # Categorical values
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_columns:

        if col != "Student_ID":

            print(f"\n{col}")

            print(
                df[col].value_counts(dropna=False)
            )

    # Numerical features
    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    numerical_features = [
        col for col in numerical_columns
        if col != "Student_ID"
    ]

    print("\nNumerical Statistics:")
    print(df[numerical_features].describe().T)

    # Negative values
    print("\nNegative Values:")

    for col in numerical_features:

        print(
            col,
            ":",
            (df[col] < 0).sum()
        )

    # Boxplots
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df[numerical_features],
        orient="h"
    )

    plt.title("Boxplots des variables numériques")
    plt.xlabel("Valeurs", fontsize=11)
    plt.tight_layout()
    plt.show()

    # Outliers using IQR
    outlier_results = []

    for col in numerical_features:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        outlier_results.append({

            "Variable": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "Lower_Bound": round(lower_bound, 2),
            "Upper_Bound": round(upper_bound, 2),
            "Number_of_Outliers": len(outliers)

        })

    outlier_df = pd.DataFrame(outlier_results)

    print("\nOutlier Results:")
    print(outlier_df.to_string(index=False))

    # Numerical statistics
    print("\nNumerical Description:")
    print(df[numerical_features].describe().T)

    # Categorical statistics
    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    categorical_features = [
        col for col in categorical_features
        if col not in ["Student_ID", target]
    ]

    print("\nCategorical Description:")
    print(df[categorical_features].describe().T)

    # Target distribution
    print("\nTarget Distribution:")
    print(df[target].value_counts())

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=target,
        order=df[target].value_counts().index
    )

    plt.title("Distribution de la variable cible")
    plt.xlabel("Mental Health Risk")
    plt.ylabel("Nombre d'étudiants")
    plt.tight_layout()
    plt.show()

    # Identify features
    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    categorical_features = [
        col for col in categorical_features
        if col not in [target, "Student_ID"]
    ]

    numerical_features = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numerical_features = [
        col for col in numerical_features
        if col != "Student_ID"
    ]

    print("\nCategorical Features:", categorical_features)
    print("Numerical Features:", numerical_features)

    # Categorical features vs target
    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 12)
    )

    axes = axes.flatten()

    for i, col in enumerate(categorical_features):

        sns.countplot(
            data=df,
            x=col,
            hue=target,
            ax=axes[i]
        )

        axes[i].set_title(f"{col} vs {target}")
        axes[i].tick_params(
            axis="x",
            rotation=30
        )

    for i in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Numerical features vs target
    fig, axes = plt.subplots(
        4, 3,
        figsize=(15, 16)
    )

    axes = axes.flatten()

    for i, col in enumerate(numerical_features):

        sns.boxplot(
            data=df,
            x=target,
            y=col,
            hue=target,
            ax=axes[i],
            palette="Set2",
            legend=False
        )

        axes[i].set_title(f"{col} vs {target}")

    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Correlation
    correlation = df[numerical_features].corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Corrélations entre les variables numériques")
    plt.tight_layout()
    plt.show()

    print("\nEDA completed successfully!")

    # Return dataset path for the next Airflow task
    return "/opt/airflow/data/student_mental_health.csv"
