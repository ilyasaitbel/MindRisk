
from airflow.decorators import task
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


@task
def train_models(prepared_data_path):

    # Charger les données préparées
    data = joblib.load(prepared_data_path)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    # Déclaration des modèles
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True)
    }

    # Entraînement des modèles
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(name, "trained successfully")

    # Calcul des métriques
    results = []

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, average="weighted"
            ),
            "Recall": recall_score(
                y_test, y_pred, average="weighted"
            ),
            "F1-score": f1_score(
                y_test, y_pred, average="weighted"
            )
        })

    results_df = pd.DataFrame(results)

    print("\nComparaison des modèles :")
    print(results_df.round(4).to_string(index=False))

    # Matrices de confusion
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, model) in enumerate(trained_models.items()):

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i]
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()

    # Sauvegarder le graphique
    project_path = Path(__file__).resolve().parents[1]
    output_dir = project_path / "data" / "airflow_state"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "confusion_matrices.png")
    plt.close()

    # Classification Report
    for name, model in trained_models.items():

        print("\n", name)
        print(classification_report(
            y_test,
            model.predict(X_test)
        ))

    # ROC-AUC multiclasse
    for name, model in trained_models.items():

        y_proba = model.predict_proba(X_test)

        auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted",
            labels=model.classes_
        )

        print(name, "ROC-AUC:", round(auc, 4))

    # Sauvegarder les modèles entraînés
    output_path = output_dir / "trained_models.joblib"

    joblib.dump(trained_models, output_path)

    print("\nModels saved successfully")

    return str(output_path)
