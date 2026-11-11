
import importlib.util
from pathlib import Path

tasks_dir = Path(__file__).parent


def load_task(filename, module_name):
    file_path = tasks_dir / filename

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# Importer les tâches
eda = load_task("01_eda.py", "eda")
preparation = load_task("02_preparation.py", "preparation")
training = load_task("03_model_training.py", "training")
optimization = load_task("04_gridsearch_optimization.py", "optimization")
evaluation = load_task("05_final_evaluation_and_save.py", "evaluation")


# Exposer les fonctions
run_eda = eda.run_eda
prepare_data = preparation.prepare_data
train_models = training.train_models
optimize_models = optimization.optimize_models
evaluate_model = evaluation.evaluate_model
