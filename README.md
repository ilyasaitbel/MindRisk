# MindRisk — analyse pédagogique du bien-être étudiant

MindRisk est un projet pédagogique de Data Science. Il explore un jeu de données étudiant, prépare les variables, compare des modèles de classification et orchestre les étapes avec Apache Airflow. Une petite application Streamlit permet de saisir un profil et d'obtenir une prédiction du modèle.

> **Limite d’usage :** ce prototype n’est pas un outil médical. Il ne doit servir ni au diagnostic, ni au dépistage clinique, ni à une décision disciplinaire, ni à la sélection d’étudiants. Ses prédictions ne sont pas des évaluations cliniques et ne doivent pas déterminer l’accès à un accompagnement.

## Objectifs

- Explorer la structure, la qualité et les distributions du jeu de données.
- Étudier les associations entre les caractéristiques étudiantes et la cible `Mental_Health_Risk`.
- Comparer des modèles de classification supervisée.
- Optimiser les hyperparamètres et évaluer un modèle sur un jeu de test.
- Automatiser les étapes avec Airflow et présenter une prédiction dans Streamlit.

## Organisation du projet

```text
.
├── app/
│   └── app.py                         # Interface Streamlit de prédiction
├── dags/
│   ├── mindrisk_pipeline_dag.py       # DAG Airflow
│   └── tasks/
│       ├── 01_eda.py                  # Analyse exploratoire
│       ├── 02_preparation.py          # Split et prétraitement
│       ├── 03_model_training.py       # Entraînement et comparaison
│       ├── 04_gridsearch_optimization.py
│       ├── 05_final_evaluation_and_save.py
│       └── __init__.py                # Expose les fonctions de tâches au DAG
├── data/
│   └── student_mental_health.csv     # Jeu de données attendu
├── models/                            # Modèles sauvegardés
├── notebooks/                         # Notebooks d’exploration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                               # Configuration locale, à garder privée
```

Les dossiers `data/airflow_state/` et `models/` contiennent des résultats générés pendant l’exécution. Les fichiers CSV et les modèles Joblib sont ignorés par Git selon `.gitignore`.

## Technologies

- Python 3.10
- Apache Airflow 2.9.2 avec `LocalExecutor`
- PostgreSQL 13 pour la base de métadonnées Airflow
- pandas, scikit-learn, matplotlib, seaborn et Joblib
- Streamlit pour l’interface de prédiction
- Docker Compose pour lancer Airflow et PostgreSQL

## Prérequis

- Docker Desktop installé et démarré.
- Python 3.10 pour exécuter l’application Streamlit localement.
- Le fichier `data/student_mental_health.csv` présent à l’emplacement attendu.
- Un fichier `.env` local contenant les variables requises par `docker-compose.yml` :
  `AIRFLOW_FERNET_KEY`, `AIRFLOW_WEBSERVER_SECRET_KEY`, `AIRFLOW_DB_USER`,
  `AIRFLOW_DB_PASSWORD` et `AIRFLOW_DB_NAME`.

Ne publiez pas `.env` et n’insérez pas de mots de passe ou de clés dans ce README.

## Lancer Airflow

Depuis la racine du projet :

```bash
docker compose up -d --build
```

Ouvrez ensuite [http://localhost:8080](http://localhost:8080) et connectez-vous avec l’utilisateur Airflow configuré pour votre instance. Si vous démarrez une base Airflow neuve et qu’aucun utilisateur n’existe, créez-en un dans le conteneur :

```bash
docker compose exec airflow-webserver airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

La commande demande le mot de passe. Gardez-le privé.

Dans Airflow, activez `mindrisk_pipeline`, puis déclenchez un run manuel. Le DAG est aussi planifié quotidiennement (`@daily`). Pour consulter l’état et les journaux :

```bash
docker compose ps
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver
```

Pour arrêter les conteneurs sans supprimer la base de métadonnées :

```bash
docker compose down
```

## Étapes du DAG

Le DAG `mindrisk_pipeline` enchaîne ces tâches :

1. **`run_eda`** — charge le CSV, affiche les dimensions, types, valeurs manquantes, doublons, statistiques, valeurs aberrantes par IQR et distributions.
2. **`prepare_data`** — sépare les données en apprentissage (80 %) et test (20 %), puis construit le prétraitement numérique, catégoriel et ordinal.
3. **`train_models`** — entraîne et compare la régression logistique, la forêt aléatoire et le SVM.
4. **`optimize_models`** — utilise `GridSearchCV` avec validation croisée stratifiée et compare les modèles optimisés.
5. **`evaluate_model`** — évalue le SVM optimisé sur le jeu de test et sauvegarde le modèle.

Les tâches transmettent leurs chemins de fichiers entre elles via les sorties Airflow (XCom). Les objets intermédiaires sont enregistrés dans `data/airflow_state/`.

## Lancer Streamlit

L’application est exécutée localement, séparément d’Airflow. `streamlit` ne figure pas actuellement dans `requirements.txt`, installez-le dans l’environnement Python local :

```bash
python -m venv .venv
```

Activation de l’environnement :

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Installation et lancement :

```bash
pip install -r requirements.txt
pip install streamlit
streamlit run app/app.py
```

L’application charge `models/mental_health_model.joblib` et présente un formulaire de prédiction.

## Résultats générés

- `data/airflow_state/prepared_data.joblib` — jeux d’apprentissage/test et préprocesseur.
- `data/airflow_state/trained_models.joblib` — modèles entraînés.
- `data/airflow_state/best_models.joblib` — modèles optimisés.
- `data/airflow_state/confusion_matrices.png` — matrices issues de la comparaison initiale.
- `data/airflow_state/final_confusion_matrix.png` — matrice du modèle évalué.
- `models/mental_health_model.pkl` — modèle final sauvegardé par la tâche d’évaluation.

## Limites actuelles et points à vérifier

- L’analyse IQR repère et rapporte des valeurs aberrantes ; le prétraitement actuel ne les supprime pas explicitement.
- Aucun rééquilibrage de classes par SMOTE ou sous-échantillonnage n’est actuellement appliqué.
- Le DAG sauvegarde le modèle final sous `models/mental_health_model.pkl`, tandis que Streamlit charge `models/mental_health_model.joblib`. Ces deux chemins/formats doivent être alignés avant d’utiliser dans l’application le modèle produit par le dernier run Airflow.
- L’application actuelle affiche une prédiction individuelle ; elle ne présente pas encore les métriques, les graphiques ou la matrice de confusion dans l’interface.
- Le fichier de données contient un identifiant étudiant. Il est exclu des variables utilisées pour l’apprentissage. Utilisez uniquement des données autorisées et anonymisées.
- Une bonne performance statistique sur ce jeu de données ne démontre ni validité clinique, ni causalité, ni aptitude à guider une intervention individuelle.

## Licence et données

Ajoutez ici la licence du code et la source/licence du jeu de données si elles sont définies pour le projet. Vérifiez les autorisations de redistribution du jeu de données avant de le publier.
