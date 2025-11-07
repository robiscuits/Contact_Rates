import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt


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
    drop_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in ['whiff_', 'velo_', 'mix_']) and not any(fam in c for fam in ['fastball', 'breaking', 'offspeed'])]
    df = df.drop(columns=drop_cols, errors='ignore')

    return df

def prepare_timeseries_data(df, df_features, on, cutoff, eps=1e-6, sigma0=1.0, c=10.0):
    df_model= df.merge(df_features, on=on, how='left')
    #Define weeks
    df_model['season_week'] = df_model['pitch_date'].dt.isocalendar().week.astype(int)

    # weekly counts per batter
    wk = (df_model.groupby(['batter_id','season_week'])
            .agg(y_it = ('contact','sum'),
                 n_it = ('is_swing','sum'))
            .reset_index())
    #Standardize week values so that gaps are reduced
    #Opportunity for future development: Incorporate missing stretches into model
    wk = wk.sort_values(['batter_id','season_week'])
    wk['t'] = wk.groupby('batter_id').cumcount()
    cutoff = pd.Timestamp('2024-07-01')

    # mark each week by its start date (Mon) to compare to cutoff
    week_start = (df_model[['season_week','pitch_date']]
                  .drop_duplicates('season_week')
                  .groupby('season_week')['pitch_date'].min())
    wk = wk.merge(week_start.rename('week_start'), on='season_week', how='left')

    wk_train = wk[wk['week_start'] < cutoff]
    wk_test  = wk[wk['week_start'] >= cutoff]
    
    # List our batters
    batters = pd.DataFrame({'batter_id': df['batter_id'].unique()}).sort_values('batter_id').reset_index(drop=True)
    batters['i'] = np.arange(len(batters))

    wk_train = wk_train.merge(batters, on='batter_id', how='right')  # keep all batters, NaN for missing weeks
    wk_train[['y_it','n_it']] = wk_train[['y_it','n_it']].fillna(0)

    # compute each batter's max t in train, then set a common T_train for full matrix form
    T_train = int(wk_train.groupby('i')['t'].max().fillna(-1).max() + 1)
    I = len(batters)

    # make dense matrices (I x T_train); weeks with no data have n=0, y=0
    y_mat = np.zeros((I, T_train), dtype=int)
    n_mat = np.zeros((I, T_train), dtype=int)

    for row in wk_train.itertuples(index=False):
        if pd.notna(row.t):
            y_mat[row.i, int(row.t)] = int(row.y_it)
            n_mat[row.i, int(row.t)] = int(row.n_it)

    # Reset X
    X =  df_features.drop(columns=['contact_rate', 'z_contact_rate','o_contact_rate'], errors='ignore')
    
    model_features = batters.merge(X, on='batter_id', how='left').fillna(0)
    X_i = model_features.to_numpy() #This is our feature row for each batter. It's not repeated t times yet.
    
    # Calculating priors
    prev = batters.merge(
        df[['batter_id','contact_rate_2023','total_swings_2023']].drop_duplicates(subset='batter_id'),
        on='batter_id', how='left'
    )

    p_prev = prev['contact_rate_2023'].fillna(prev['contact_rate_2023'].mean()).clip(eps,1-eps)
    logit_prev = np.log(p_prev/(1-p_prev))  # or use your stored 'logit_prev'

    n_prev = prev['total_swings_2023'].fillna(0).to_numpy()
    logit_prev = logit_prev.to_numpy()

    # prior SD that shrinks with prior swings
    sigma_theta0_i = sigma0 / np.sqrt(n_prev + c)
    
    return y_mat, n_mat, X_i, sigma_theta0_i, logit_prev

def call_model(K, logit_prev, sigma_theta0_i, I, T, X_i, n_mat, y_mat):
    with pm.Model() as rw_model:
        # Priors
        sigma_rw = pm.HalfNormal("sigma_rw", 0.5)
        beta = pm.Normal("beta", 0, 1, shape=K)

        # Initial contact ability θ_i0
        theta0 = pm.Normal("theta0", mu=logit_prev, sigma=sigma_theta0_i, shape=I)

        # Random walk noise
        eps = pm.Normal("eps", 0, sigma_rw, shape=(I, T))

        # Build θ recursively
        theta = pt.zeros((I, T))
        theta = pt.set_subtensor(theta[:, 0], theta0 + eps[:, 0])
        theta = pm.Deterministic("theta", pt.inc_subtensor(theta[:, 1:], theta[:, :-1] + eps[:, 1:]))

        # Link to probability
        p = pm.Deterministic("p", pm.math.sigmoid(theta + X_i @ beta))

        # Likelihood
        y_obs = pm.Binomial("y_obs", n=n_mat, p=p, observed=y_mat)