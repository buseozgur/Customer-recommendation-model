"""
Training script for Sephora Product Recommendation Model
Converted from 04_Recommendation_Model.ipynb
"""

import warnings, time, json, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# ML
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ndcg_score

# Semantic
from sentence_transformers import SentenceTransformer

# Tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# SHAP
import shap

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

DATA_DIR    = Path('data/processed')
MODELS_DIR  = Path('outputs/models')
METRICS_DIR = Path('outputs/metrics')

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

EFFECT_WEIGHTS = {
    'helped':      1.0,
    'worsened':   -1.5,
    'target_only': 0.3,
    'unknown':     0.0,
}

N_FOLDS = 5
N_TRIALS = 50  # Production: 150-200
TUNE_FOLDS = 3
SBERT_NAME = 'all-MiniLM-L6-v2'

# ═══════════════════════════════════════════════════════════════════
# HELPER CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CVResult:
    name: str
    ndcg5:  List[float] = field(default_factory=list)
    ndcg10: List[float] = field(default_factory=list)
    times:  List[float] = field(default_factory=list)

    @property
    def mn5(self):  return np.mean(self.ndcg5)
    @property
    def sd5(self):  return np.std(self.ndcg5)
    @property
    def mn10(self): return np.mean(self.ndcg10)
    @property
    def sd10(self): return np.std(self.ndcg10)
    @property
    def mt(self):   return np.mean(self.times)


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def build_aggregate_scores(rcl: pd.DataFrame, rtf: pd.DataFrame) -> pd.DataFrame:
    df = rcl.copy()

    if 'skin_type' not in df.columns or df['skin_type'].isna().all():
        smap = rtf.set_index(['author_id', 'product_id'])['skin_type'].to_dict()
        df['skin_type'] = df.apply(
            lambda r: smap.get((r.get('author_id'), r['product_id']), 'Unknown'), axis=1)
    df['skin_type'] = df['skin_type'].fillna('Unknown')

    df['effect_weight']  = df['effect_label'].map(EFFECT_WEIGHTS).fillna(0.0)
    df['rating']         = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0)
    df['rating_norm']    = (df['rating'] - 1) / 4
    df['weighted_score'] = df['rating_norm'] * df['effect_weight'] * df['concern_confidence']

    agg = df.groupby(['product_id', 'concern', 'skin_type']).agg(
        product_name       = ('product_name_final', 'first'),
        brand_name         = ('brand_name_final',   'first'),
        primary_category   = ('primary_category',   'first'),
        secondary_category = ('secondary_category',  'first'),
        mean_weighted_score= ('weighted_score',      'mean'),
        mean_rating        = ('rating',              'mean'),
        review_count       = ('product_id',          'count'),
        helped_count       = ('effect_label', lambda x: (x == 'helped').sum()),
        worsened_count     = ('effect_label', lambda x: (x == 'worsened').sum()),
        mean_confidence    = ('concern_confidence',   'mean'),
    ).reset_index()

    agg['review_count_bonus'] = np.log1p(agg['review_count']) / 10
    agg['helped_ratio']       = agg['helped_count']   / agg['review_count'].clip(lower=1)
    agg['worsened_ratio']     = agg['worsened_count'] / agg['review_count'].clip(lower=1)
    agg['net_effect_ratio']   = agg['helped_ratio'] - agg['worsened_ratio']
    agg['aggregate_score']    = agg['mean_weighted_score'] + agg['review_count_bonus']

    return agg


def build_features(agg: pd.DataFrame):
    df = agg.copy()

    # Relevance label
    conds = [
        (df['worsened_ratio'] > 0.30) | (df['helped_ratio'] < 0.10),
        (df['helped_ratio'] >= 0.10) & (df['helped_ratio'] < 0.35),
        (df['helped_ratio'] >= 0.35) & (df['helped_ratio'] < 0.60),
        (df['helped_ratio'] >= 0.60) & (df['mean_rating'] >= 3.5),
    ]
    df['relevance_label'] = np.select(conds, [0, 1, 2, 3], default=1)

    # Numeric features
    df['log_review_count']    = np.log1p(df['review_count'])
    df['sqrt_review_count']   = np.sqrt(df['review_count'])
    df['rating_x_helped']     = df['mean_rating']    * df['helped_ratio']
    df['rating_x_net']        = df['mean_rating']    * df['net_effect_ratio']
    df['confidence_x_score']  = df['mean_confidence'] * df['mean_weighted_score']
    df['helped_x_confidence'] = df['helped_ratio']   * df['mean_confidence']
    df['score_x_log_reviews'] = df['mean_weighted_score'] * df['log_review_count']
    df['helped_sq']           = df['helped_ratio']  ** 2
    df['worsened_sq']         = df['worsened_ratio'] ** 2

    # Categorical encoding
    enc = {}
    for col in ['concern', 'skin_type', 'primary_category', 'secondary_category']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str))
        enc[col] = le

    # Query ID
    df['query_id'] = df['concern_enc'].astype(str) + '_' + df['skin_type_enc'].astype(str)
    qle = LabelEncoder()
    df['query_id_enc'] = qle.fit_transform(df['query_id'])
    enc['query'] = qle

    return df, enc


# ═══════════════════════════════════════════════════════════════════
# CV FRAMEWORK
# ═══════════════════════════════════════════════════════════════════

def group_kfold(df, n=5, seed=SEED):
    uq = df['query_id_enc'].unique().copy()
    np.random.seed(seed)
    np.random.shuffle(uq)
    folds = np.array_split(uq, n)
    return [
        (df.index[df['query_id_enc'].isin(np.concatenate([folds[j] for j in range(n) if j != i]))].tolist(),
         df.index[df['query_id_enc'].isin(folds[i])].tolist())
        for i in range(n)
    ]


def calc_ndcg(df_val, preds, ks=(5, 10)):
    tmp = df_val.copy()
    tmp['_p'] = preds
    res = {k: [] for k in ks}
    for _, g in tmp.groupby('query_id_enc'):
        if len(g) < 2: continue
        tr = g['relevance_label'].values.reshape(1, -1)
        pr = g['_p'].values.reshape(1, -1)
        for k in ks:
            if len(g) >= k:
                try: res[k].append(ndcg_score(tr, pr, k=k))
                except: pass
    return {k: np.mean(v) if v else 0.0 for k, v in res.items()}


def gsizes(arr):
    _, c = np.unique(arr, return_counts=True)
    return c.tolist()


# ═══════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def run_lgbm(df, splits, features, cat_idxs, params=None, n_rounds=300):
    res = CVResult('LightGBM LambdaRank')
    p = {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5, 10],
         'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 5,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
         'label_gain': [0, 1, 3, 7], 'verbose': -1, 'n_jobs': -1}
    if params: p.update(params)
    for tr_i, va_i in splits:
        tr, va = df.loc[tr_i], df.loc[va_i]
        ds = lgb.Dataset(tr[features].values, label=tr['relevance_label'].values,
                         group=gsizes(tr['query_id_enc'].values))
        t0 = time.time()
        m = lgb.train(p, ds, num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(9999)])
        res.times.append(time.time() - t0)
        nd = calc_ndcg(va, m.predict(va[features].values))
        res.ndcg5.append(nd[5]); res.ndcg10.append(nd[10])
    return res


def run_xgb(df, splits, features, cat_idxs, params=None):
    res = CVResult('XGBoost LambdaMART')
    p = {'objective': 'rank:ndcg', 'learning_rate': 0.05, 'max_depth': 6,
         'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'n_estimators': 300, 'verbosity': 0, 'n_jobs': -1, 'random_state': SEED}
    if params: p.update(params)
    for tr_i, va_i in splits:
        tr = df.loc[tr_i].sort_values('query_id_enc')
        va = df.loc[va_i]
        m = xgb.XGBRanker(**p)
        t0 = time.time()
        m.fit(tr[features].values, tr['relevance_label'].values,
              qid=tr['query_id_enc'].values, verbose=False)
        res.times.append(time.time() - t0)
        nd = calc_ndcg(va, m.predict(va[features].values))
        res.ndcg5.append(nd[5]); res.ndcg10.append(nd[10])
    return res


def run_catboost(df, splits, features, cat_idxs, params=None):
    res = CVResult('CatBoost YetiRank')
    p = {'loss_function': 'YetiRank', 'eval_metric': 'NDCG',
         'learning_rate': 0.05, 'depth': 6, 'iterations': 300,
         'random_seed': SEED, 'verbose': False}
    if params: p.update(params)
    for tr_i, va_i in splits:
        tr = df.loc[tr_i].sort_values('query_id_enc')
        va = df.loc[va_i]
        pool = Pool(tr[features].values, label=tr['relevance_label'].values,
                    group_id=tr['query_id_enc'].values, cat_features=cat_idxs)
        m = CatBoostRanker(**p)
        t0 = time.time(); m.fit(pool); res.times.append(time.time() - t0)
        nd = calc_ndcg(va, m.predict(va[features].values))
        res.ndcg5.append(nd[5]); res.ndcg10.append(nd[10])
    return res


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  SEPHORA RECOMMENDATION MODEL TRAINING")
    print("="*70)

    # 1. Load data
    print("\n[1/8] Loading data...")
    rtf = pd.read_parquet(DATA_DIR / 'review_text_features.parquet')
    rcl = pd.read_parquet(DATA_DIR / 'review_concern_level.parquet')
    print(f"  review_text_features: {rtf.shape}")
    print(f"  review_concern_level: {rcl.shape}")

    # 2. Build aggregate scores
    print("\n[2/8] Building aggregate scores...")
    agg_scores = build_aggregate_scores(rcl, rtf)
    print(f"  Aggregate table: {agg_scores.shape}")

    # 3. Feature engineering
    print("\n[3/8] Feature engineering...")
    ml_df, label_encoders = build_features(agg_scores)

    FEATURES = [
        'mean_weighted_score', 'mean_rating', 'log_review_count', 'sqrt_review_count',
        'helped_ratio', 'worsened_ratio', 'net_effect_ratio',
        'mean_confidence', 'review_count_bonus',
        'rating_x_helped', 'rating_x_net', 'confidence_x_score',
        'helped_x_confidence', 'score_x_log_reviews',
        'helped_sq', 'worsened_sq',
        'concern_enc', 'skin_type_enc', 'primary_category_enc', 'secondary_category_enc',
    ]

    CAT_IDXS = [FEATURES.index(f) for f in
                ['concern_enc', 'skin_type_enc', 'primary_category_enc', 'secondary_category_enc']]

    print(f"  Features: {len(FEATURES)}")
    print(f"  ML df: {ml_df.shape}")

    # 4. CV splits
    print("\n[4/8] Creating CV splits...")
    cv_splits = group_kfold(ml_df, n=N_FOLDS)
    print(f"  {N_FOLDS}-Fold Group CV ready")

    # 5. Baseline comparison
    print("\n[5/8] Baseline model comparison...")
    baseline = {}
    for tag, fn in [
        ('lgbm',     lambda: run_lgbm(ml_df, cv_splits, FEATURES, CAT_IDXS)),
        ('xgb',      lambda: run_xgb(ml_df, cv_splits, FEATURES, CAT_IDXS)),
        ('catboost', lambda: run_catboost(ml_df, cv_splits, FEATURES, CAT_IDXS)),
    ]:
        print(f"  [{tag}] ", end='', flush=True)
        baseline[tag] = fn()
        r = baseline[tag]
        print(f"NDCG@10={r.mn10:.4f} ± {r.sd10:.4f}")

    # 6. Optuna tuning
    print(f"\n[6/8] Hyperparameter tuning ({N_TRIALS} trials)...")
    fast_cv = group_kfold(ml_df, n=TUNE_FOLDS)

    def make_study():
        return optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))

    # LightGBM
    def lgbm_obj(trial):
        p = {
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 3, 30),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq':      trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1':         trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
            'lambda_l2':         trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
        }
        n_rnd = trial.suggest_int('n_rounds', 100, 600)
        scores = []
        for fi, (tr_i, va_i) in enumerate(fast_cv):
            tr, va = ml_df.loc[tr_i], ml_df.loc[va_i]
            bp = {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10],
                  'label_gain': [0,1,3,7], 'verbose': -1, 'n_jobs': -1}
            bp.update(p)
            ds = lgb.Dataset(tr[FEATURES].values, label=tr['relevance_label'].values,
                             group=gsizes(tr['query_id_enc'].values))
            m = lgb.train(bp, ds, num_boost_round=n_rnd, callbacks=[lgb.log_evaluation(9999)])
            nd = calc_ndcg(va, m.predict(va[FEATURES].values))
            scores.append(nd[10])
            trial.report(np.mean(scores), fi)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    print("  LightGBM...", end=' ', flush=True)
    lgbm_study = make_study()
    lgbm_study.optimize(lgbm_obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Best={lgbm_study.best_value:.4f}")

    # XGBoost
    def xgb_obj(trial):
        p = {
            'objective': 'rank:ndcg', 'verbosity': 0, 'n_jobs': -1, 'random_state': SEED,
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
        }
        scores = []
        for fi, (tr_i, va_i) in enumerate(fast_cv):
            tr = ml_df.loc[tr_i].sort_values('query_id_enc')
            va = ml_df.loc[va_i]
            m = xgb.XGBRanker(**p)
            m.fit(tr[FEATURES].values, tr['relevance_label'].values,
                  qid=tr['query_id_enc'].values, verbose=False)
            nd = calc_ndcg(va, m.predict(va[FEATURES].values))
            scores.append(nd[10])
            trial.report(np.mean(scores), fi)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    print("  XGBoost...", end=' ', flush=True)
    xgb_study = make_study()
    xgb_study.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Best={xgb_study.best_value:.4f}")

    # CatBoost
    def cb_obj(trial):
        p = {
            'loss_function': 'YetiRank', 'eval_metric': 'NDCG',
            'random_seed': SEED, 'verbose': False,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth':         trial.suggest_int('depth', 3, 10),
            'iterations':    trial.suggest_int('iterations', 100, 500),
            'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        }
        scores = []
        for fi, (tr_i, va_i) in enumerate(fast_cv):
            tr = ml_df.loc[tr_i].sort_values('query_id_enc')
            va = ml_df.loc[va_i]
            pool = Pool(tr[FEATURES].values, label=tr['relevance_label'].values,
                        group_id=tr['query_id_enc'].values, cat_features=CAT_IDXS)
            m = CatBoostRanker(**p); m.fit(pool)
            nd = calc_ndcg(va, m.predict(va[FEATURES].values))
            scores.append(nd[10])
            trial.report(np.mean(scores), fi)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    print("  CatBoost...", end=' ', flush=True)
    cb_study = make_study()
    cb_study.optimize(cb_obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Best={cb_study.best_value:.4f}")

    # 7. Tuned model evaluation
    print("\n[7/8] Evaluating tuned models (5-fold)...")
    tuned = {}

    lgbm_p = {k: v for k, v in lgbm_study.best_params.items() if k != 'n_rounds'}
    r = run_lgbm(ml_df, cv_splits, FEATURES, CAT_IDXS, params=lgbm_p,
                 n_rounds=lgbm_study.best_params.get('n_rounds', 300))
    r.name = 'LightGBM (tuned)'; tuned['lgbm'] = r
    print(f"  LightGBM: NDCG@10={r.mn10:.4f}")

    r = run_xgb(ml_df, cv_splits, FEATURES, CAT_IDXS, params=xgb_study.best_params)
    r.name = 'XGBoost (tuned)'; tuned['xgb'] = r
    print(f"  XGBoost:  NDCG@10={r.mn10:.4f}")

    r = run_catboost(ml_df, cv_splits, FEATURES, CAT_IDXS, params=cb_study.best_params)
    r.name = 'CatBoost (tuned)'; tuned['catboost'] = r
    print(f"  CatBoost: NDCG@10={r.mn10:.4f}")

    # 8. Train final model
    print("\n[8/8] Training final model on all data...")
    BEST_KEY = max(tuned, key=lambda k: tuned[k].mn10)
    print(f"  Best model: {tuned[BEST_KEY].name} (NDCG@10={tuned[BEST_KEY].mn10:.4f})")

    X_all = ml_df[FEATURES].values
    y_all = ml_df['relevance_label'].values
    g_all = ml_df['query_id_enc'].values

    if BEST_KEY == 'lgbm':
        fp = {k: v for k, v in lgbm_study.best_params.items() if k != 'n_rounds'}
        fp.update({'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5, 10],
                  'label_gain': [0, 1, 3, 7], 'verbose': -1, 'n_jobs': -1})
        ds = lgb.Dataset(X_all, label=y_all, group=gsizes(g_all))
        final_model = lgb.train(fp, ds,
                                num_boost_round=lgbm_study.best_params.get('n_rounds', 300),
                                callbacks=[lgb.log_evaluation(9999)])
        MODEL_TYPE = 'lgbm'
    elif BEST_KEY == 'xgb':
        fp = {**xgb_study.best_params, 'objective': 'rank:ndcg',
              'verbosity': 0, 'n_jobs': -1, 'random_state': SEED}
        sdf = ml_df.sort_values('query_id_enc')
        final_model = xgb.XGBRanker(**fp)
        final_model.fit(sdf[FEATURES].values, sdf['relevance_label'].values,
                        qid=sdf['query_id_enc'].values, verbose=False)
        MODEL_TYPE = 'xgb'
    else:
        fp = {**cb_study.best_params, 'loss_function': 'YetiRank',
              'eval_metric': 'NDCG', 'random_seed': SEED, 'verbose': False}
        sdf = ml_df.sort_values('query_id_enc')
        pool = Pool(sdf[FEATURES].values, label=sdf['relevance_label'].values,
                    group_id=sdf['query_id_enc'].values, cat_features=CAT_IDXS)
        final_model = CatBoostRanker(**fp); final_model.fit(pool)
        MODEL_TYPE = 'catboost'

    print(f"  Model type: {MODEL_TYPE}")

    # 9. Save everything
    print("\n[SAVE] Saving model artifacts...")

    # Model
    if MODEL_TYPE == 'lgbm':
        final_model.save_model(str(MODELS_DIR / 'final_ranker.txt'))
    elif MODEL_TYPE == 'xgb':
        final_model.save_model(str(MODELS_DIR / 'final_ranker.json'))
    else:
        final_model.save_model(str(MODELS_DIR / 'final_ranker_catboost'))
    print(f"  ✓ Model → outputs/models/final_ranker.*")

    # Encoders
    with open(MODELS_DIR / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"  ✓ Encoders → outputs/models/label_encoders.pkl")

    # Config
    config = {
        'best_model': tuned[BEST_KEY].name,
        'model_type': MODEL_TYPE,
        'features': FEATURES,
        'cat_feature_idxs': CAT_IDXS,
        'effect_weights': EFFECT_WEIGHTS,
        'cv_scores': {
            **{f'{k}_baseline': round(r.mn10, 4) for k, r in baseline.items()},
            **{f'{k}_tuned':    round(r.mn10, 4) for k, r in tuned.items()},
        },
        'final_ndcg10': round(tuned[BEST_KEY].mn10, 4),
        'optuna_trials': N_TRIALS,
    }

    with open(MODELS_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Config → outputs/models/config.json")

    # Scoring table
    ml_df.to_parquet(DATA_DIR / 'ml_scoring_table.parquet', index=False)
    print(f"  ✓ Scoring table → data/processed/ml_scoring_table.parquet")

    print("\n" + "="*70)
    print("  TRAINING COMPLETE!")
    print("="*70)
    print(f"\n  Best Model: {tuned[BEST_KEY].name}")
    print(f"  NDCG@10:    {tuned[BEST_KEY].mn10:.4f}")
    print(f"\n  Files saved to:")
    print(f"    - outputs/models/")
    print(f"    - data/processed/ml_scoring_table.parquet")
    print()


if __name__ == '__main__':
    main()
