"""
04 - Product Recommendation Model
Converted from notebook: 04_Recommendation_Model.ipynb
"""


# ======================================================================
# # 04 — Product Recommendation Model
# **Girdi:** `data/processed/review_text_features.parquet` & `review_concern_level.parquet`  
# **Çıktı:** Eğitilmiş model + scoring tablosu + metrikler
# ---
# ### Neyi nereye kaydediyoruz?
# ```
# CUSTOMER-RECOMMENDATION-MODEL/
# │
# ├── data/processed/                       ← VERİ ÜRÜNLERİ (pipeline'ın bir sonraki adımının input'u)
# │   ├── review_text_features.parquet        (03 notebook'tan geliyor — dokunmuyoruz)
# │   ├── review_concern_level.parquet        (03 notebook'tan geliyor — dokunmuyoruz)
# │   └── ml_scoring_table.parquet          ← YENİ — ürün bazlı skorlar, API'de recommend() çağrısında kullanılacak
# │
# ├── outputs/models/                       ← EĞİTİLMİŞ MODEL DOSYALARI (inference'da yüklenir)
# │   ├── final_ranker.txt (veya .json)       ranking modeli
# │   ├── product_concern_embeddings.pkl      SBERT embedding'leri
# │   ├── label_encoders.pkl                  categorical encoder'lar
# │   ├── optuna_studies.pkl                  tuning geçmişi (opsiyonel, reproduce için)
# │   └── config.json                         tüm parametreler — API bu dosyayı okuyup modeli yükler
# │
# ├── outputs/metrics/                      ← DEĞERLENDİRME ÇIKTILARI (rapor/sunum için, production'da kullanılmaz)
# │   ├── model_comparison.csv                baseline vs tuned tablo
# │   ├── baseline_comparison.png             bar chart
# │   ├── optuna_tuning.png                   tuning geçmişi grafiği
# │   ├── baseline_vs_tuned.png               gain grafiği
# │   ├── feature_importance.png              hangi feature ne kadar etkili
# │   ├── shap_analysis.png                   SHAP beeswarm
# │   ├── ensemble_weights.png                ağırlık pie chart
# │   └── performance_heatmap.png             concern × skin_type hata analizi
# ```
# ### Pipeline Akışı
# ```
# 01_data_understanding → 02_EDA → 03_NLP_concern → [BU NOTEBOOK] → (sonraki: API & UI)
# ```
# ======================================================================


# ======================================================================
# ---
# ## 0. Kurulum & Import
# ======================================================================


# ======================================================================
# ## 1. Dizin Yapısı
# Bu notebook `notebooks/` dizininden çalışıyor → path'ler `../` ile başlıyor.
# ======================================================================


# ======================================================================
# ## 2. Veri Yükleme & Hızlı Bakış
# ======================================================================

rtf = pd.read_parquet(DATA_DIR / 'review_text_features.parquet')
rcl = pd.read_parquet(DATA_DIR / 'review_concern_level.parquet')

print(f'review_text_features : {rtf.shape[0]:,} satır, {rtf.shape[1]} sütun')
print(f'review_concern_level : {rcl.shape[0]:,} satır, {rcl.shape[1]} sütun')
print(f'\nUnique ürün  : {rcl["product_id"].nunique():,}')
print(f'Unique concern: {rcl["concern"].nunique()}')
print(f'Unique effect : {rcl["effect_label"].nunique()}')

print('── review_concern_level sütunları ──')
print(rcl.dtypes.to_string())
print()
rcl.head(3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1 - Concern dağılımı
c = rcl['concern'].value_counts()
axes[0].barh(c.index, c.values, color=COLORS[:len(c)])
axes[0].set_title('Concern Dağılımı', fontweight='bold')
axes[0].set_xlabel('Kayıt')

# 2 - Effect label
e = rcl['effect_label'].value_counts()
axes[1].pie(e.values, labels=e.index, autopct='%1.1f%%', colors=COLORS[:len(e)])
axes[1].set_title('Effect Label Dağılımı', fontweight='bold')

# 3 - Ürün başına review
pc = rcl.groupby('product_id').size()
axes[2].hist(pc.clip(upper=pc.quantile(0.95)), bins=40, color=COLORS[2], edgecolor='white')
axes[2].set_title('Ürün Başına Concern Kaydı', fontweight='bold')
axes[2].set_xlabel('Kayıt Sayısı')

plt.suptitle('Veri Genel Bakış', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(METRICS_DIR / 'eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Kaydedildi → outputs/metrics/eda_overview.png')


# ======================================================================
# ---
# ## 3. Feature Engineering
# ### Adım 3a — Aggregate Scoring
# Her `(product_id, concern, skin_type)` üçlüsü için review'lardan deterministik bir skor üretiyoruz.  
# Bu hem bir **baseline** hem de ensemble'ın bir katmanı.
# ```
# effect_weight  →  helped: +1.0 | worsened: -1.5 | target_only: +0.3 | unknown: 0
# weighted_score  = (rating_norm) × (effect_weight) × (concern_confidence)
# aggregate_score = mean(weighted_score) + log1p(review_count) / 10
# ```
# **Neden -1.5?** Negatif deneyim daha kesin dille yazılır, insanlar kötü deneyimi daha çok paylaşır → daha güvenilir sinyal.
# ======================================================================

EFFECT_WEIGHTS = {
    'helped':      1.0,
    'worsened':   -1.5,
    'target_only': 0.3,
    'unknown':     0.0,
}


def build_aggregate_scores(rcl: pd.DataFrame, rtf: pd.DataFrame) -> pd.DataFrame:
    """(product_id, concern, skin_type) bazında aggregate skor."""
    df = rcl.copy()

    # skin_type eksikse rtf'den tamamla
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


agg_scores = build_aggregate_scores(rcl, rtf)
print(f'Aggregate tablo : {agg_scores.shape[0]:,} satır')
print(f'Unique ürün     : {agg_scores["product_id"].nunique():,}')
print(f'Unique concern  : {agg_scores["concern"].nunique()}')
agg_scores.head(3)


# ======================================================================
# ### Adım 3b — ML Feature Matrisi
# Aggregate tablodan türetilen özellikler + kategorik encode.
# ======================================================================

def build_features(agg: pd.DataFrame):
    df = agg.copy()

    # ── Hedef: Relevance label (0-3, learning-to-rank) ─────────────
    conds = [
        (df['worsened_ratio'] > 0.30) | (df['helped_ratio'] < 0.10),   # 0 = kötü
        (df['helped_ratio'] >= 0.10) & (df['helped_ratio'] < 0.35),    # 1 = orta
        (df['helped_ratio'] >= 0.35) & (df['helped_ratio'] < 0.60),    # 2 = iyi
        (df['helped_ratio'] >= 0.60) & (df['mean_rating'] >= 3.5),     # 3 = çok iyi
    ]
    df['relevance_label'] = np.select(conds, [0, 1, 2, 3], default=1)

    # ── Sayısal feature'lar ────────────────────────────────────────
    df['log_review_count']    = np.log1p(df['review_count'])
    df['sqrt_review_count']   = np.sqrt(df['review_count'])
    df['rating_x_helped']     = df['mean_rating']    * df['helped_ratio']
    df['rating_x_net']        = df['mean_rating']    * df['net_effect_ratio']
    df['confidence_x_score']  = df['mean_confidence'] * df['mean_weighted_score']
    df['helped_x_confidence'] = df['helped_ratio']   * df['mean_confidence']
    df['score_x_log_reviews'] = df['mean_weighted_score'] * df['log_review_count']
    df['helped_sq']           = df['helped_ratio']  ** 2
    df['worsened_sq']         = df['worsened_ratio'] ** 2

    # ── Kategorik encode ───────────────────────────────────────────
    enc = {}
    for col in ['concern', 'skin_type', 'primary_category', 'secondary_category']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str))
        enc[col] = le

    # ── Query ID — her (concern, skin_type) bir sıralama grubu ────
    df['query_id'] = df['concern_enc'].astype(str) + '_' + df['skin_type_enc'].astype(str)
    qle = LabelEncoder()
    df['query_id_enc'] = qle.fit_transform(df['query_id'])
    enc['query'] = qle

    return df, enc


ml_df, label_encoders = build_features(agg_scores)

# Feature listesi
FEATURES = [
    # Temel istatistikler
    'mean_weighted_score', 'mean_rating', 'log_review_count', 'sqrt_review_count',
    'helped_ratio', 'worsened_ratio', 'net_effect_ratio',
    'mean_confidence', 'review_count_bonus',
    # Interaction
    'rating_x_helped', 'rating_x_net', 'confidence_x_score',
    'helped_x_confidence', 'score_x_log_reviews',
    'helped_sq', 'worsened_sq',
    # Kategorik
    'concern_enc', 'skin_type_enc', 'primary_category_enc', 'secondary_category_enc',
]

CAT_IDXS = [FEATURES.index(f) for f in
            ['concern_enc', 'skin_type_enc', 'primary_category_enc', 'secondary_category_enc']]

print(f'Feature sayısı    : {len(FEATURES)}')
print(f'Categorical index : {CAT_IDXS}')
print(f'ML df shape       : {ml_df.shape}')
print(f'\nRelevance label dağılımı:')
print(ml_df['relevance_label'].value_counts().sort_index())


# ======================================================================
# ---
# ## 4. Cross-Validation Framework
# **Group K-Fold**: `query_id` (concern × skin_type) grupları bölünmeden train/val'a ayrılır.  
# Aynı query'nin bazı ürünleri train'de, bazıları val'da olmaz → **data leakage yok**.
# ======================================================================

@dataclass
class CVResult:
    """Bir modelin CV sonuçlarını tutan küçük konteyner."""
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


def group_kfold(df, n=5, seed=SEED):
    """query_id_enc grubuna göre K-Fold böler."""
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
    """Validation seti üzerinde grup bazlı NDCG hesaplar."""
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
    """LightGBM/XGBoost group size listesi."""
    _, c = np.unique(arr, return_counts=True)
    return c.tolist()


N_FOLDS = 5
cv_splits = group_kfold(ml_df, n=N_FOLDS)

print(f'{N_FOLDS}-Fold Group CV hazır')
for i, (tr, va) in enumerate(cv_splits):
    print(f'  Fold {i+1}: train={len(tr):,}  val={len(va):,}')


# ======================================================================
# ---
# ## 5. Baseline Model Karşılaştırması
# 5 farklı modeli **default** hiperparametrelerle 5-Fold CV'den geçiriyoruz:
# | Model | Tür | Neden deniyoruz? |
# |-------|-----|------------------|
# | **LightGBM LambdaRank** | Pairwise ranking | NDCG'yi doğrudan optimize eder, hızlı |
# | **XGBoost LambdaMART** | Pairwise ranking | LightGBM alternatifi, bazen daha stabil |
# | **CatBoost YetiRank** | Pairwise ranking | Categorical feature'larda güçlü |
# | **Random Forest** | Pointwise regresyon | Yorumlanabilir baseline |
# | **Ridge Regression** | Pointwise lineer | Hızlı, alt sınır ölçer |
# ======================================================================

print('🏁 Baseline karşılaştırması başlıyor...\n')
baseline: Dict[str, CVResult] = {}

for tag, fn in [
    ('lgbm',     lambda: run_lgbm(ml_df, cv_splits)),
    ('xgb',      lambda: run_xgb(ml_df, cv_splits)),
    ('catboost', lambda: run_catboost(ml_df, cv_splits)),
    ('rf',       lambda: run_rf(ml_df, cv_splits)),
    ('ridge',    lambda: run_ridge(ml_df, cv_splits)),
]:
    print(f'  [{tag:<8}] ', end='', flush=True)
    baseline[tag] = fn()
    r = baseline[tag]
    print(f'NDCG@5={r.mn5:.4f}  NDCG@10={r.mn10:.4f} ± {r.sd10:.4f}  ({r.mt:.1f}s/fold)')

print('\n✅ Tamamlandı.')


# ======================================================================
# ---
# ## 6. Optuna ile Hyperparameter Tuning
# Top 3 modeli (LightGBM, XGBoost, CatBoost) Optuna TPE + MedianPruner ile optimize ediyoruz.
# - **TPE Sampler:** Grid search'e göre ~10x daha verimli
# - **MedianPruner:** Kötü trial'ları 1-2 fold sonra kesiyor
# - **3-Fold:** Tuning sırasında hız için 3-fold kullanılır, final değerlendirme 5-fold'da yapılır
# > `N_TRIALS = 50` → Production kalitesi için 150-200'e çek.
# ======================================================================

N_TRIALS   = 50   # ← production için 150-200 yap
TUNE_FOLDS = 3

fast_cv = group_kfold(ml_df, n=TUNE_FOLDS)

def make_study():
    return optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))

print(f'Tuning: {N_TRIALS} trial × 3 model × {TUNE_FOLDS}-fold CV')


# ======================================================================
# ---
# ## 7. Tuned Modeller — 5-Fold Final Değerlendirme
# ======================================================================

print('Tuned modeller 5-Fold CV ile değerlendiriliyor...\n')
tuned: Dict[str, CVResult] = {}

# LightGBM tuned
lgbm_p = {k: v for k, v in lgbm_study.best_params.items() if k != 'n_rounds'}
r = run_lgbm(ml_df, cv_splits, params=lgbm_p,
             n_rounds=lgbm_study.best_params.get('n_rounds', 300))
r.name = 'LightGBM (tuned)'; tuned['lgbm'] = r
print(f'  {r.name:<26} NDCG@10 = {r.mn10:.4f} ± {r.sd10:.4f}')

# XGBoost tuned
r = run_xgb(ml_df, cv_splits, params=xgb_study.best_params)
r.name = 'XGBoost (tuned)'; tuned['xgb'] = r
print(f'  {r.name:<26} NDCG@10 = {r.mn10:.4f} ± {r.sd10:.4f}')

# CatBoost tuned
r = run_catboost(ml_df, cv_splits, params=cb_study.best_params)
r.name = 'CatBoost (tuned)'; tuned['catboost'] = r
print(f'  {r.name:<26} NDCG@10 = {r.mn10:.4f} ± {r.sd10:.4f}')


# ======================================================================
# ---
# ## 8. Final Model Eğitimi (Tüm Veri)
# En iyi tuned model seçilir ve **tüm veri** üzerinde eğitilir.
# ======================================================================

BEST_KEY = max(tuned, key=lambda k: tuned[k].mn10)
print(f'🏆 En iyi tuned model: {tuned[BEST_KEY].name}')
print(f'   NDCG@10 = {tuned[BEST_KEY].mn10:.4f} ± {tuned[BEST_KEY].sd10:.4f}')
print(f'\nTüm veri üzerinde eğitiliyor...')

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

else:  # catboost
    fp = {**cb_study.best_params, 'loss_function': 'YetiRank',
          'eval_metric': 'NDCG', 'random_seed': SEED, 'verbose': False}
    sdf = ml_df.sort_values('query_id_enc')
    pool = Pool(sdf[FEATURES].values, label=sdf['relevance_label'].values,
                group_id=sdf['query_id_enc'].values, cat_features=CAT_IDXS)
    final_model = CatBoostRanker(**fp); final_model.fit(pool)
    MODEL_TYPE = 'catboost'

print(f'\n✅ Final model eğitildi: {MODEL_TYPE}')


# ======================================================================
# ---
# ## 9. Semantic Retrieval Katmanı (SBERT)
# Her `(product_id, concern)` çifti için ortalama review embedding'i hesaplanır.  
# Inference sırasında kullanıcının `"oily skin, acne"` girdisi doğal dil cümlesi olarak embed edilir ve cosine similarity ile en yakın ürünler getirilir.
# ======================================================================

SBERT_NAME = 'all-MiniLM-L6-v2'
print(f'SBERT yükleniyor: {SBERT_NAME}')
sbert = SentenceTransformer(SBERT_NAME)
print('✅ Hazır')

def build_product_embeddings(rcl, sbert, batch_size=128):
    """Her (product_id, concern) çifti → ortalama SBERT embedding."""
    texts = rcl['normalized_text'].fillna('').unique().tolist()
    print(f'  {len(texts):,} unique metin encode ediliyor...')
    embs = sbert.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True)
    t2e = dict(zip(texts, embs))
    dim = embs.shape[1]

    result = {}
    for (pid, concern), grp in rcl.groupby(['product_id', 'concern']):
        vecs = np.array([t2e.get(t, np.zeros(dim)) for t in grp['normalized_text'].fillna('')])
        avg = vecs.mean(axis=0)
        n = np.linalg.norm(avg)
        result[(pid, concern)] = avg / n if n > 0 else avg

    print(f'  ✅ {len(result):,} (product, concern) embedding hazır.')
    return result


pc_embeddings = build_product_embeddings(rcl, sbert)


# ======================================================================
# ---
# ## 10. Ensemble Ağırlık Optimizasyonu
# 3 katmanın (aggregate, ranking model, semantic) ağırlıklarını Optuna ile optimize ediyoruz.
# ======================================================================

def weight_obj(trial):
    w1 = trial.suggest_float('w_agg',   0.0, 1.0)
    w2 = trial.suggest_float('w_model', 0.0, 1.0 - w1)
    w3 = max(0.0, 1.0 - w1 - w2)
    ens = w1 * ml_df['agg_norm'] + w2 * ml_df['model_norm'] + w3 * ml_df['sem_norm']
    tmp = ml_df.assign(ens=ens)
    scores = []
    for _, g in tmp.groupby('query_id_enc'):
        if len(g) >= 5:
            try:
                scores.append(ndcg_score(
                    g['relevance_label'].values.reshape(1,-1),
                    g['ens'].values.reshape(1,-1), k=10))
            except: pass
    return np.mean(scores) if scores else 0.0


print('Ensemble ağırlık optimizasyonu (200 trial)...')
w_study = optuna.create_study(direction='maximize',
                              sampler=optuna.samplers.TPESampler(seed=SEED))
w_study.optimize(weight_obj, n_trials=200, show_progress_bar=True)

W_AGG   = w_study.best_params['w_agg']
W_MODEL = w_study.best_params['w_model']
W_SEM   = max(0.0, 1.0 - W_AGG - W_MODEL)

print(f'\n✅ Optimal ağırlıklar:')
print(f'   w_aggregate : {W_AGG:.4f}')
print(f'   w_model     : {W_MODEL:.4f}')
print(f'   w_semantic  : {W_SEM:.4f}')
print(f'   Ensemble NDCG@10: {w_study.best_value:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Trial history
wdf = w_study.trials_dataframe()
wdf = wdf[wdf['state'] == 'COMPLETE']
axes[0].plot(wdf['number'], wdf['value'], alpha=0.3, color='steelblue')
axes[0].plot(wdf['number'], wdf['value'].cummax(), color='crimson', lw=2)
axes[0].set_title('Ensemble Weight Optimization', fontweight='bold')
axes[0].set_xlabel('Trial'); axes[0].set_ylabel('NDCG@10')
axes[0].annotate(f'Best: {w_study.best_value:.4f}\nw_agg={W_AGG:.3f}  w_model={W_MODEL:.3f}  w_sem={W_SEM:.3f}',
                 xy=(0.03, 0.82), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# Pie chart
vals = [W_AGG, W_MODEL, W_SEM]
lbls = ['Aggregate', tuned[BEST_KEY].name, 'Semantic']
clrs = ['steelblue', 'crimson', 'seagreen']
nz = [(v,l,c) for v,l,c in zip(vals,lbls,clrs) if v > 0.01]
axes[1].pie([v for v,_,_ in nz], labels=[l for _,l,_ in nz],
            colors=[c for _,_,c in nz], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Optimal Ensemble Ağırlıkları', fontweight='bold')

plt.tight_layout()
plt.savefig(METRICS_DIR / 'ensemble_weights.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Kaydedildi → outputs/metrics/ensemble_weights.png')


# ======================================================================
# ---
# ## 11. Hata Analizi (Concern × Skin Type Heatmap)
# ======================================================================


# ======================================================================
# ---
# ## 12. Her Şeyi Kaydet
# ```
# data/processed/ml_scoring_table.parquet    ← API'nin okuyacağı ana tablo
# outputs/models/final_ranker.*              ← eğitilmiş model
# outputs/models/product_concern_embeddings  ← SBERT embedding'leri
# outputs/models/label_encoders.pkl          ← categorical encoder'lar
# outputs/models/optuna_studies.pkl          ← tuning geçmişi
# outputs/models/config.json                 ← tüm parametreler
# ```
# ======================================================================


# ======================================================================
# ---
# ## 13. Özet Rapor
# ======================================================================

print()
print('╔' + '═'*63 + '╗')
print('║' + '  EXPERIMENT ÖZET RAPORU'.center(63) + '║')
print('╠' + '═'*63 + '╣')
print(f'║  Tarih            : {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"):<40} ║')
print(f'║  CV               : {N_FOLDS}-Fold Group K-Fold (concern × skin_type)      ║')
print(f'║  Feature sayısı   : {len(FEATURES):<40} ║')
print(f'║  Optuna trial     : {N_TRIALS} / model{"":>33}║')
print('╠' + '═'*63 + '╣')
print('║  BASELINE NDCG@10:' + ' '*44 + '║')
for k, r in sorted(baseline.items(), key=lambda x: -x[1].mn10):
    line = f'    {r.name:<28} {r.mn10:.4f} ± {r.sd10:.4f}'
    print(f'║  {line:<61}║')
print('╠' + '═'*63 + '╣')
print('║  TUNED NDCG@10:' + ' '*47 + '║')
for k, r in sorted(tuned.items(), key=lambda x: -x[1].mn10):
    d = r.mn10 - baseline[k].mn10
    line = f'    {r.name:<28} {r.mn10:.4f} ± {r.sd10:.4f}  (Δ{d:+.4f})'
    print(f'║  {line:<61}║')
print('╠' + '═'*63 + '╣')
print(f'║  🏆 FINAL MODEL    : {tuned[BEST_KEY].name:<40}║')
print(f'║     Model NDCG@10  : {tuned[BEST_KEY].mn10:<40.4f}║')
print(f'║     Ensemble NDCG  : {w_study.best_value:<40.4f}║')
print('╠' + '═'*63 + '╣')
print(f'║  ENSEMBLE AĞIRLIKLARI:' + ' '*40 + '║')
print(f'║    w_aggregate : {W_AGG:<44.4f}║')
print(f'║    w_model     : {W_MODEL:<44.4f}║')
print(f'║    w_semantic  : {W_SEM:<44.4f}║')
print('╚' + '═'*63 + '╝')


# ======================================================================
# ---
# ## Sonraki Adım → API & UI
# Bu notebook'tan üretilen artefactlar bir sonraki aşamada şu şekilde kullanılacak:
# ```python
# # API endpoint örneği (FastAPI)
# # POST /recommend
# # body: { "skin_type": "oily", "concern": "acne", "top_n": 10 }
# @app.post('/recommend')
# def recommend(skin_type: str, concern: str, top_n: int = 10):
#     # 1. config.json oku → model type, features, weights
#     # 2. ml_scoring_table.parquet filtrele (concern + skin_type)
#     # 3. final_ranker ile rerank
#     # 4. product_concern_embeddings ile semantic score
#     # 5. ensemble_weights ile birleştir
#     # 6. top_n döndür
# ```
# ======================================================================
