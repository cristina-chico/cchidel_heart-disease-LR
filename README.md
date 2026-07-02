# Heart Disease Prevalence Predictor

> Análisis exploratorio y modelo de Machine Learning para predecir la prevalencia de enfermedades cardíacas a partir de datos sociodemográficos por condado en Estados Unidos.

---

## Tabla de contenidos

- [Descripción](#descripción)
- [Dataset](#dataset)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Tecnologías](#tecnologías)
- [Pipeline del proyecto](#pipeline-del-proyecto)
- [Resultados](#resultados)
- [Cómo ejecutar](#cómo-ejecutar)
- [Conclusiones](#conclusiones)
  
---

## Descripción
Este proyecto aplica técnicas de **EDA (Exploratory Data Analysis)** y **Machine Learning supervisado** para analizar la relación entre factores sociodemográficos (edad, raza, educación, pobreza, mercado laboral y recursos sanitarios) y la prevalencia de enfermedades cardíacas en condados de Estados Unidos.

---

**Variable objetivo:** `Heart disease_prevalence`

Se entrenaron y compararon tres modelos de regresión:
- Regresión Lineal
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Dataset

- **Fuente:** [4Geeks / BreatheCode — demographic_health_data.csv](https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv)
- **Dimensiones originales:** 3.140 filas × 108 columnas
- **Dimensiones tras limpieza:** 3.140 filas × 46 columnas
- **Tipos de datos:** `float64` (61), `int64` (45), `object` (2)
- **Nulos:** Ninguno
- **Duplicados:** Ninguno

  ### Grupos de variables principales

| Grupo | Descripción |
|---|---|
| Geográficas | FIPS, STATE_FIPS |
| Demografía por edad | Porcentajes agrupados (0-19, 20-39, 40-59, 60-79, 80+) |
| Raza / Etnicidad | Distribución poblacional por grupo étnico |
| Educación | Nivel educativo de adultos (% y valores absolutos) |
| Pobreza e ingresos | PCTPOVALL, MEDHHINC, Median_Household_Income |
| Mercado laboral | Tasa de desempleo, empleados, población activa |
| Recursos sanitarios | Médicos, enfermeros, hospitales, camas UCI por 100k hab. |
| Prevalencia enfermedades | Obesidad, COPD, diabetes, CKD, enfermedad cardíaca |

---


## Estructura del proyecto

```
├── ML_RegresionLineal-Heart-disease-predictor.ipynb   # Notebook principal
├── models/
│   └── random_forest_model.pkl                        # Modelo guardado
└── README.md
```

## Tencologías

- `pandas`, `numpy` — manipulación y análisis de datos
- `matplotlib`, `seaborn` — visualizaciones y heatmaps de correlación
- `scikit-learn` — Pipeline, StandardScaler, modelos ML y métricas
- `pickle` — serialización del modelo final

---

## Pipeline del proyecto

```
1. Carga de datos
        ↓
2. EDA — exploración de variables
        ↓
3. Limpieza — eliminación de nulos, duplicados y variables redundantes
        ↓
4. Feature Engineering — agrupación de rangos de edad, análisis de correlación (heatmaps)
        ↓
5. Análisis de outliers — decisión de conservarlos (datos médicos)
        ↓
6. Split Train / Test (80/20, random_state=42)
        ↓
7. Entrenamiento — Linear Regression, Random Forest, Gradient Boosting (con Pipeline + StandardScaler)
        ↓
8. Evaluación y comparación de modelos (MAE, RMSE, R²)
        ↓
9. Feature Importance (Random Forest)
        ↓
10. Guardado del modelo (.pkl)
```


## Resultados

### Comparativa de modelos

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.21 | 1.54 | 0.63 |
| Random Forest | 0.98 | 1.30 | 0.72 |
| **Gradient Boosting** | **0.95** | **1.25** | **0.74** |

El modelo guardado es **Random Forest** (`models/random_forest_model.pkl`), por su buen equilibrio entre rendimiento e interpretabilidad.

### Variables más relevantes (Feature Importance — Random Forest)

Las variables con mayor peso predictivo pertenecen a los grupos de **edad avanzada**, **prevalencia de otras enfermedades crónicas** (obesidad, diabetes, COPD) y **acceso a recursos sanitarios**.

---

## Conclusiones

- Los modelos de ensamble (**Random Forest** y **Gradient Boosting**) superan a la regresión lineal al capturar relaciones no lineales entre variables sociodemográficas y la prevalencia cardíaca.
- La **regresión lineal** sigue siendo útil para interpretar relaciones directas entre variables.
- Factores como el **envejecimiento poblacional**, la **prevalencia de otras enfermedades crónicas** y el **acceso a recursos sanitarios** son los predictores más influyentes.
- Este tipo de modelos puede apoyar la toma de decisiones en **políticas de salud pública**, identificando condados con mayor riesgo cardiovascular.

---

> Proyecto desarrollado como parte del aprendizaje en Machine Learning supervisado — Regresión.
