from airflow.decorators import task
from pathlib import Path
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


@task
def evaluate_model(prepared_data_path, best_models_path):

    # Charger les données préparées
    data = joblib.load(prepared_data_path)

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Charger les meilleurs modèles
    best_models = joblib.load(best_models_path)

    # Sélection du modèle SVM optimisé
    best_model = best_models["SVM"]

    # Prédictions
    y_pred = best_model.predict(X_test)

    print("Best Model: SVM")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=best_model.classes_
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=best_model.classes_
    )

    disp.plot(cmap="Blues", values_format="d")
    disp.ax_.grid(False)

    plt.title("Matrice de confusion - SVM")

    # Chemin du projet
    project_root = Path(__file__).resolve().parents[2]

    # Sauvegarder la matrice
    output_dir = project_root / "data" / "airflow_state"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "final_confusion_matrix.png")
    plt.close()

    # Comptage des prédictions correctes et incorrectes
    errors = y_test != y_pred

    print(
        "Prédictions correctes :",
        errors.value_counts().get(False, 0)
    )

    print(
        "Prédictions incorrectes :",
        errors.value_counts().get(True, 0)
    )

    print(
        "Taux d'erreur :",
        round(errors.mean() * 100, 2),
        "%"
    )

    # Tableau des prédictions erronées
    errors_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    print("\nPrédictions erronées:")
    print(
        errors_df[
            errors_df["Actual"] != errors_df["Predicted"]
        ].head(10).to_string()
    )

    # Sauvegarde du pipeline final
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "mental_health_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    print("\nPipeline sauvegardé avec succès !")

    return str(model_path)