
from airflow.decorators import dag
from datetime import datetime

from tasks import (
    run_eda,
    prepare_data,
    train_models,
    optimize_models,
    evaluate_model
)


@dag(
    dag_id="mindrisk_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mindrisk"],
)
def mindrisk_pipeline():

    data = run_eda()

    prepared_data = prepare_data(data)

    trained_models = train_models(prepared_data)

    best_models = optimize_models(prepared_data)

    evaluate_model(prepared_data, best_models)


mindrisk_pipeline()
