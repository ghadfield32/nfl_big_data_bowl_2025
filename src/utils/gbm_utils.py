
import joblib
from pathlib import Path

def save_pipeline(model, preprocessor, path: str = "data/models/saved_models/gbm_pipeline.joblib"):
    """
    Save both model and preprocessor together to a single file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "preprocessor": preprocessor},
        out_path
    )
    print(f"✅ Pipeline saved to {out_path.resolve()}")

def load_pipeline(path: str = "data/models/saved_models/gbm_pipeline.joblib"):
    """
    Load the model+preprocessor dict back into memory.
    Returns a tuple: (model, preprocessor)
    """
    saved = joblib.load(path)
    return saved["model"], saved["preprocessor"]
