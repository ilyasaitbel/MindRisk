
from airflow.decorators import task
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


@task
def optimize_models(prepared_data_path):

    # Charger les données préparées
    data = joblib.load(prepared_data_path)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    # Définition des modèles et des hyperparamètres
    models_params = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {
                "model__C": [0.01, 0.1, 1, 10]
            }
        },

        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5]
            }
        },

        "SVM": {
            "model": SVC(),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"]
            }
        }
    }

    # Validation croisée stratifiée
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    best_models = {}
    optimization_rows = []

    # Grid Search pour chaque modèle
    for name, config in models_params.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", config["model"])
        ])

        grid = GridSearchCV(
            pipeline,
            config["params"],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_

        optimization_rows.append({
            "Model": name,
            "Best Parameters": grid.best_params_,
            "Best F1-score": grid.best_score_
        })

        print(name, "Best Score:", grid.best_score_)

    # Résultats de l'optimisation
    optimization_results = pd.DataFrame(optimization_rows)

    print("\nOptimization Results:")
    print(optimization_results.to_string(index=False))

    # Comparaison des modèles optimisés sur le test
    results = []

    for name, model in best_models.items():

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

    comparison_df = pd.DataFrame(results)

    print("\nOptimized Models Comparison:")
    print(comparison_df.round(4).to_string(index=False))

    # Sauvegarder les meilleurs modèles
    project_path = Path(__file__).resolve().parents[1]
    output_dir = project_path / "data" / "airflow_state"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "best_models.joblib"

    joblib.dump(best_models, output_path)

    print("\nBest models saved successfully")

    return str(output_path)
