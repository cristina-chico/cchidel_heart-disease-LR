# Models

Trained model artefacts are **git-ignored** (large binary files).  
Run `python train.py` from the project root to regenerate them.

---

## Artefacts

| File | Algorithm | R² (test) | Notes |
|---|---|---|---|
| `final_model.pkl` | Random Forest Regressor | 0.72 | Saved Pipeline (StandardScaler + model). **Production model.** |
| `random_forest_model.pkl` | Random Forest Regressor | 0.72 | Legacy name from notebook. Same as above. |

---

## Loading a model

```python
import pickle

with open("models/final_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# pipeline.predict() expects the same feature columns as X_train
predictions = pipeline.predict(X_new)
```

> **Note:** The pipeline includes a `StandardScaler` — pass the raw (unscaled) features and the pipeline handles scaling automatically.
