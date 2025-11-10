
# Bayesian Contact Rate Model (MAP Estimation)

## Overview
This project builds a Bayesian model to estimate and predict MLB hitter contact ability using data through June 2024. The model uses Maximum A Posteriori (MAP) estimation to capture each hitter’s baseline contact skill while accounting for league-wide contextual predictors such as pitch mix, swing tendencies, and velocity exposure.

Originally designed as a dynamic random-walk model, this implementation focuses on the static MAP form to efficiently estimate midseason contact ability. Each hitter’s prior is informed by their 2023 contact rate and total swings, with shrinkage applied based on prior sample size. Feature variables act as shared global effects, while per-hitter intercepts represent individual baseline contact performance.

## Repository Structure
- `main.ipynb` - Full analysis notebook (training, testing, evaluation)
- `main.html` - Rendered HTML report (primary deliverable)
- `main.py` - Core data preparation and model definition
- `helper_functions.py` - Supporting utilities and preprocessing helpers
- `requirements.txt` - Python package dependencies
- `README.md` - Project overview and setup guide

## Setup Instructions

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run or view the notebook

```bash
jupyter notebook deliverable.ipynb
```

To view the pre-rendered report:

```bash
open deliverable.html
```

## Methodology Summary

* **Model Type:** Bayesian Binomial model estimated via Maximum A Posteriori (MAP)
* **Response Variable:** Probability of making contact on a swing
* **Features (X):** Contextual and performance-based predictors (e.g., whiff rates, pitch mix exposure, velocity)
* **Hierarchy:** Player-level priors centered on 2023 contact performance
* **Training Window:** Through June 2024
* **Testing Window:** Post-June 2024 (simulated out-of-sample)

## Evaluation Metrics

The model's predictive accuracy was assessed using:

* Weighted correlation and R2 between predicted and actual contact rates
* Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Log Loss and Brier Score for probabilistic calibration

## Results Summary

MAP estimates achieved moderate predictive power:

* Weighted Corr: ~0.51
* Weighted R2: ~0.25
* RMSE: 0.087
* Log Loss: 0.31

These results indicate the model captures a meaningful portion of the variation in contact performance, though uncertainty and noise remain, particularly for low-sample players or volatile contact profiles.

## Author

Robert George
Chicago, IL
[rrgeorge00@gmail.com]

