import pandas as pd
import numpy as np
import pymc as pm
from scipy.special import logit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, brier_score_loss

def collapse_pitch_families(df, fastballs, breaking, offspeed):
    # Whiff
    whiff_cols = [c for c in df.columns if c.startswith('whiff_')]
    for fam, types in {'fastball': fastballs, 'breaking': breaking, 'offspeed': offspeed}.items():
        cols = [f'whiff_{t}' for t in types if f'whiff_{t}' in df.columns]
        if cols:
            df[f'whiff_{fam}'] = df[cols].mean(axis=1)

    # Velocity
    velo_cols = [c for c in df.columns if c.startswith('velo_')]
    for fam, types in {'fastball': fastballs, 'breaking': breaking, 'offspeed': offspeed}.items():
        cols = [f'velo_{t}' for t in types if f'velo_{t}' in df.columns]
        if cols:
            df[f'velo_{fam}'] = df[cols].mean(axis=1)

    # Mix
    mix_cols = [c for c in df.columns if c.startswith('mix_')]
    for fam, types in {'fastball': fastballs, 'breaking': breaking, 'offspeed': offspeed}.items():
        cols = [f'mix_{t}' for t in types if f'mix_{t}' in df.columns]
        if cols:
            df[f'mix_{fam}'] = df[cols].sum(axis=1)  # sum since mix_% already sum to 1

    # Drop original pitch-type columns
    drop_cols = [
        c for c in df.columns
        if any(c.startswith(prefix) for prefix in ['whiff_', 'velo_', 'mix_'])
        and not any(fam in c for fam in ['fastball', 'breaking', 'offspeed'])
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    return df


def prepare_timeseries_data(df, df_features, on, train=True,
                            eps=1e-6, sigma0=1.0, c=10.0):
    """
    Prepare aligned training or test data for the Bayesian contact model.

    Arguments:
        df : pitch-level data (must include pitch_date, contact, is_swing, batter_id)
        df_features : per-batter feature table
        on : join key(s)
        train : if True, returns data through June; if False, after June
        eps : small value to avoid log(0)
        sigma0, c : prior shrinkage parameters

    Returns:
        X, y, n, batter_idx, logit_prev, sigma_theta0_i
    """

    cutoff = pd.Timestamp('2024-07-01')
    df = df.copy()
    df['pitch_date'] = pd.to_datetime(df['pitch_date'])

    # Merge in features
    df_features = df_features.copy()
    assert 'batter_id' in df_features.columns

    # Weekly aggregates
    wk = (
        df.groupby(['batter_id', pd.Grouper(key='pitch_date', freq='W-MON')])
          .agg(y_it=('contact', 'sum'),
               n_it=('is_swing', 'sum'))
          .reset_index()
          .rename(columns={'pitch_date': 'week_start'})
    )

    # Split by cutoff
    wk_train = wk[wk['week_start'] < cutoff].copy()
    wk_test  = wk[wk['week_start'] >= cutoff].copy()

    # Identify all batters who have ANY data before cutoff (rookies included)
    pre_cutoff_batters = set(wk_train['batter_id'].unique())

    # Filter out true post-July rookies
    wk_test = wk_test[wk_test['batter_id'].isin(pre_cutoff_batters)].copy()

    # Build index from all batters who appeared before cutoff
    batters = (
        wk[wk['batter_id'].isin(pre_cutoff_batters)][['batter_id']]
        .drop_duplicates()
        .sort_values('batter_id')
        .reset_index(drop=True)
    )
    batters['batter_idx'] = np.arange(len(batters))

    # Choose subset for this run
    wk_use = wk_train if train else wk_test

    # Merge batter index and features
    wk_use = wk_use.merge(batters, on='batter_id', how='left')
    wk_use = wk_use.merge(df_features, on='batter_id', how='left')

    # Fill and filter
    wk_use['y_it'] = wk_use['y_it'].fillna(0).astype(int)
    wk_use['n_it'] = wk_use['n_it'].fillna(0).astype(int)
    wk_use = wk_use[wk_use['n_it'] > 0].copy()

    # Build numeric feature matrix
    drop_cols = {
        'batter_id', 'batter_idx', 'week_start',
        'y_it', 'n_it', 'contact_rate_2023', 'total_swings_2023'
    }
    X_df = wk_use.drop(columns=[c for c in drop_cols if c in wk_use.columns])
    X_df = X_df.select_dtypes(include=[np.number]).fillna(0)

    X = X_df.to_numpy(dtype=np.float64)
    y = wk_use['y_it'].to_numpy(dtype=np.int64)
    n = wk_use['n_it'].to_numpy(dtype=np.int64)
    batter_idx = wk_use['batter_idx'].to_numpy(dtype=np.int64)

    # Priors from 2023, aligned with all pre-cutoff batters
    prev = (
        df[['batter_id', 'contact_rate_2023', 'total_swings_2023']]
        .drop_duplicates('batter_id')
        .merge(batters, on='batter_id', how='right')
        .sort_values('batter_idx')
    )

    league_avg = prev['contact_rate_2023'].dropna().mean()
    p_prev = prev['contact_rate_2023'].fillna(league_avg).clip(eps, 1 - eps)
    n_prev = prev['total_swings_2023'].fillna(0)

    logit_prev = np.log(p_prev / (1 - p_prev))
    sigma_theta0_i = sigma0 / np.sqrt(n_prev + c)

    print(f"[prep] {'TRAIN' if train else 'TEST'}: "
          f"N={len(y)}, I={len(batters)}, K={X.shape[1]}")
    print(f"[prep] Rookies kept: {len(pre_cutoff_batters)} hitters through June.")
    print(f"[prep] Post-July rookies dropped automatically.")

    return X, y, n, batter_idx, logit_prev, sigma_theta0_i


def call_model(X, y, n, batter_idx, logit_prev, sigma_theta0_i):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    batter_idx = np.asarray(batter_idx, dtype=np.int64)
    logit_prev = np.asarray(logit_prev, dtype=np.float64)
    sigma_theta0_i = np.asarray(sigma_theta0_i, dtype=np.float64)

    N, K = X.shape
    I = logit_prev.shape[0]

    with pm.Model() as model:
        # Per-batter baseline ability
        alpha = pm.Normal(
            "alpha",
            mu=logit_prev,
            sigma=sigma_theta0_i,
            shape=I,
        )

        # Global feature effects
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=K)

        # Linear predictor
        eta = alpha[batter_idx] + pm.math.dot(X, beta)

        # Binomial likelihood
        p = pm.math.sigmoid(eta)
        pm.Binomial("y_obs", n=n, p=p, observed=y)

        print("Running MAP estimation...")
        map_estimate = pm.find_MAP(progressbar=True)
        print("MAP estimation complete.")

    return map_estimate

def weighted_metrics(actual, pred, weights, eps=1e-9):
    """
    Compute weighted model performance metrics for probabilistic predictions.
    
    Parameters:
        actual  : array-like, actual contact rates (y/n)
        pred    : array-like, predicted probabilities
        weights : array-like, swing counts or sample weights
        eps     : float, small value to stabilize logs and divisions
    """
    actual, pred, w = np.asarray(actual), np.asarray(pred), np.asarray(weights)
    w = w / (w.sum() + eps)

    # Weighted means
    mean_actual = np.sum(w * actual)
    mean_pred = np.sum(w * pred)

    # Weighted covariance and correlation
    cov = np.sum(w * (actual - mean_actual) * (pred - mean_pred))
    var_actual = np.sum(w * (actual - mean_actual)**2)
    var_pred = np.sum(w * (pred - mean_pred)**2)
    weighted_corr = cov / np.sqrt(var_actual * var_pred + eps)

    # Weighted R²
    ss_tot = np.sum(w * (actual - mean_actual)**2)
    ss_res = np.sum(w * (actual - pred)**2)
    weighted_r2 = 1 - ss_res / (ss_tot + eps)

    # Weighted MSE, RMSE, MAE
    mse = np.sum(w * (actual - pred)**2)
    rmse = np.sqrt(mse)
    mae = np.sum(w * np.abs(actual - pred))

    # Log loss and Brier score (requires binary outcomes, but still informative here)
    binary_actual = (actual > 0.5).astype(int)
    logloss = log_loss(binary_actual, np.clip(pred, eps, 1 - eps), sample_weight=weights)
    brier = brier_score_loss(binary_actual, pred, sample_weight=weights)

    # Pack results
    return {
        "Weighted Corr": weighted_corr,
        "Weighted R²": weighted_r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Log Loss": logloss,
        "Brier Score": brier
    }

