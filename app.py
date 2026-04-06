import streamlit as st
import pandas as pd
import numpy as np
import os, re, datetime, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Cricket AI",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}
.stApp { background: #f8fafc !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 16px 40px !important; max-width: 560px !important; }

.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-bottom: 3px solid #0ea5e9;
    padding: 22px 20px 18px;
    margin: 0 -16px 24px;
    text-align: center;
}
.app-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px; letter-spacing: 6px;
    background: linear-gradient(90deg, #34d399, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin: 0;
}
.app-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 3px;
    color: #94a3b8; margin-top: 6px;
}
.live-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.4);
    color: #ef4444; font-size: 9px; font-weight: 700;
    letter-spacing: 2px; padding: 4px 10px;
    border-radius: 20px; font-family: 'JetBrains Mono', monospace;
    margin-top: 10px;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #ef4444; animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #64748b; margin: 20px 0 10px;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: #e2e8f0; }

.stTextInput label, .stNumberInput label, .stTextArea label {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important; font-weight: 700 !important;
    color: #374151 !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important; margin-bottom: 4px !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 20px !important; font-weight: 700 !important;
    padding: 10px 14px !important; height: 52px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.15) !important;
}
.stTextInput > div > div > input::placeholder,
.stNumberInput > div > div > input::placeholder {
    color: #94a3b8 !important; font-size: 13px !important; font-weight: 400 !important;
}
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important; color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; padding: 12px 14px !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.15) !important;
}
.stTextArea > div > div > textarea::placeholder { color: #94a3b8 !important; }

.stNumberInput > div > div > div > button {
    background: #f1f5f9 !important;
    border-color: #e2e8f0 !important; color: #475569 !important;
}

.stButton > button, .stFormSubmitButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: #ffffff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 20px !important; letter-spacing: 4px !important;
    border: none !important; border-radius: 12px !important;
    height: 54px !important; cursor: pointer !important;
    box-shadow: 0 4px 16px rgba(2,132,199,.25) !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    opacity: .92 !important; transform: translateY(-1px) !important;
}

.card {
    background: #ffffff; border: 1.5px solid #e2e8f0;
    border-radius: 14px; padding: 18px; margin: 8px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.card-green { border-color: #86efac !important; background: #f0fdf4 !important; }
.card-blue  { border-color: #93c5fd !important; background: #eff6ff !important; }
.card-yellow{ border-color: #fcd34d !important; background: #fffbeb !important; }
.card-red   { border-color: #fca5a5 !important; background: #fff1f2 !important; }

.match-bar {
    background: #0f172a; border-radius: 14px;
    padding: 16px 18px; display: flex;
    justify-content: space-between; align-items: center;
    margin-bottom: 8px; box-shadow: 0 4px 16px rgba(0,0,0,.15);
}
.match-teams { font-size: 15px; font-weight: 700; color: #f8fafc; }
.match-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #64748b; margin-top: 3px;
}

.prob-pct-bat {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 60px; color: #059669; line-height: 1;
}
.prob-pct-bowl {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 60px; color: #94a3b8; line-height: 1; text-align: right;
}
.prob-bar-track {
    height: 10px; background: #e2e8f0;
    border-radius: 99px; overflow: hidden; margin: 12px 0;
}
.prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #059669, #0ea5e9);
    border-radius: 99px; transition: width 1s cubic-bezier(.4,0,.2,1);
}
.prob-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
.prob-team { font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; }

/* ── Verdict cards ── */
.verdict {
    border-radius: 12px; padding: 14px 16px;
    display: flex; align-items: center; gap: 12px;
    margin: 12px 0; border: 1.5px solid;
}
.verdict-good { border-color: #86efac; background: #f0fdf4; }
.verdict-mid  { border-color: #fcd34d; background: #fffbeb; }
.verdict-bad  { border-color: #fca5a5; background: #fff1f2; }
.verdict-icon { font-size: 26px; }
.verdict-title { font-family: 'Bebas Neue', sans-serif; font-size: 17px; letter-spacing: 2px; }
.verdict-good .verdict-title { color: #16a34a; }
.verdict-mid  .verdict-title { color: #d97706; }
.verdict-bad  .verdict-title { color: #dc2626; }
.verdict-desc { font-size: 12px; color: #64748b; margin-top: 2px; }

.stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin: 10px 0; }
.stat-box {
    background: #ffffff; border: 1.5px solid #e2e8f0;
    border-radius: 10px; padding: 12px 8px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.stat-val { font-family: 'Bebas Neue', sans-serif; font-size: 26px; line-height: 1; margin-bottom: 2px; }
.stat-lbl { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: #94a3b8; }

/* ── Edge cards ── */
.edge-box, .edge-card {
    background: #ffffff; border: 1.5px solid #e2e8f0;
    border-radius: 12px; padding: 14px 16px; margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.edge-name { font-size: 13px; font-weight: 700; color: #0f172a; margin-bottom: 8px; }
.edge-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #f1f5f9; }
.edge-row:last-child { border-bottom: none; }
.edge-lbl, .edge-key { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b; }
.edge-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #0f172a; }
.edge-pos, .c-green { color: #16a34a !important; }
.edge-neg, .c-red   { color: #dc2626 !important; }
.edge-neu           { color: #d97706 !important; }

/* ── Value alert / banner ── */
.value-alert, .value-banner {
    background: #f0fdf4; border: 2px solid #86efac;
    border-radius: 10px; padding: 12px 16px; margin-top: 8px;
}
.value-alert-title, .value-banner-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 16px; letter-spacing: 2px; color: #16a34a;
}
.value-alert-sub, .value-banner-sub { font-size: 11px; color: #64748b; margin-top: 3px; }

/* ── Extracted info rows ── */
.extracted-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #f1f5f9;
}
.extracted-row:last-child { border-bottom: none; }
.extracted-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #64748b;
}
.extracted-val { font-size: 13px; font-weight: 600; color: #0f172a; }

/* ── Scenario grid ── */
.scenario-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 8px 0;
}
.scenario-box {
    background: #ffffff; border: 1.5px solid #e2e8f0;
    border-radius: 10px; padding: 10px 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.scenario-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b; margin-bottom: 4px;
}
.scenario-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 24px; line-height: 1;
}
.scenario-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; margin-top: 2px;
}

.hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #16a34a;
    margin: 4px 0 8px; padding: 6px 10px;
    background: #f0fdf4; border-radius: 6px;
    border: 1px solid #bbf7d0;
}

.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important;
    border: 1px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 1px !important;
    border-radius: 9px !important; font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #0284c7 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.1) !important;
}

[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important; padding: 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
}

.disclaimer {
    text-align: center; font-size: 10px; color: #94a3b8;
    padding: 16px; margin-top: 20px; border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
TOTAL_BALLS    = 120
MAX_WICKETS    = 10
EDGE_THRESHOLD = 10
VALID_OVER_MIN = 5
VALID_OVER_MAX = 15
MODEL_VERSION  = 'v2'
LOG_FILE       = 'predictions_log.csv'
NOTES_FILE     = 'match_notes.csv'
FEATS = [
    'runs_needed','balls_remaining','wickets_left',
    'crr','rrr','run_rate_ratio',
    'runs_per_wicket','balls_per_wicket','resource_score','phase',
    'runs_last_18','wkts_last_18',
    'venue_chase_wr',
    'batter_sr','bowler_eco_feat',
]

# ══════════════════════════════════════════════════════════
# MODEL LOADING  (cached — runs once per Python process)
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    cache_file = 'model_cache.pkl'
    key_file   = 'model_cache_key.txt'

    missing = [f for f in ['matches.csv','deliveries.csv'] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    matches    = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    data_key   = f"{MODEL_VERSION}_{len(deliveries)}"

    # ── Player lookups ──────────────────────────────────────
    has_players = os.path.exists('batsman.csv') and os.path.exists('bowler.csv')
    if has_players:
        batsman_df = pd.read_csv('batsman.csv')
        bowler_df  = pd.read_csv('bowler.csv')
        csr_map    = (batsman_df[batsman_df['innings_number']==2]
                      .groupby('player_name')['strike_rate'].mean())
        eco_map    = bowler_df.groupby('player_name')['economy_rate'].mean()
        team_csr   = (batsman_df[batsman_df['innings_number']==2]
                      .groupby('player_team')['strike_rate'].mean())
        team_eco   = bowler_df.groupby('player_team')['economy_rate'].mean()
        def_sr     = float(csr_map.mean())
        def_eco    = float(eco_map.mean())
    else:
        csr_map = eco_map = team_csr = team_eco = pd.Series(dtype=float)
        def_sr  = 111.7
        def_eco = 8.5

    # ── Venue stats ─────────────────────────────────────────
    chase_team = (deliveries[deliveries['inning']==2]
                  .groupby('match_id')['batting_team'].first().reset_index()
                  .rename(columns={'batting_team':'chasing_team'}))
    clean_matches = (matches[matches['super_over']=='N']
                     [matches['winner'].notna() & (matches['winner']!='')]
                     [matches['target_runs'].notna()]
                     [matches['method'].isna()])
    m_v = clean_matches.merge(chase_team, left_on='id', right_on='match_id', how='left')
    m_v['chase_win'] = (m_v['winner'] == m_v['chasing_team']).astype(float)
    venue_tbl = (m_v.groupby('venue')
                 .agg(venue_chase_wr=('chase_win','mean'), n=('id','count'))
                 .query('n >= 5').reset_index())
    venue_map     = venue_tbl.set_index('venue')['venue_chase_wr'].to_dict()
    def_venue_wr  = 0.50

    lookups = {
        'venue_map'    : venue_map,
        'team_sr'      : team_csr.to_dict(),
        'team_eco'     : team_eco.to_dict(),
        'default_sr'   : def_sr,
        'default_eco'  : def_eco,
        'default_venue': def_venue_wr,
    }

    # ── Check cache ─────────────────────────────────────────
    cache_valid = (
        os.path.exists(cache_file) and
        os.path.exists(key_file) and
        open(key_file).read().strip() == data_key
    )

    if cache_valid:
        model, feats = joblib.load(cache_file)
        return model, feats, "cached", lookups

    # ── Feature engineering ─────────────────────────────────
    keep = clean_matches[['id','winner','target_runs','venue','season']].rename(columns={'id':'match_id'})
    df   = deliveries.merge(keep, on='match_id', how='inner')
    df   = df[df['inning']==2].copy()
    df   = df.sort_values(['match_id','over','ball'])

    df['ball_num']        = df.groupby('match_id').cumcount()+1
    df = df[df['ball_num'] <= TOTAL_BALLS]
    df['balls_remaining'] = TOTAL_BALLS - df['ball_num']
    df['overs_done']      = df['ball_num']/6
    df['cum_runs']        = df.groupby('match_id')['total_runs'].cumsum()
    df['cum_wickets']     = df.groupby('match_id')['is_wicket'].cumsum()
    df['runs_needed']     = df['target_runs']-df['cum_runs']
    df['wickets_left']    = MAX_WICKETS-df['cum_wickets']
    df['crr']             = df['cum_runs']/df['overs_done'].clip(lower=0.1)
    df['rrr']             = (df['runs_needed']*6)/df['balls_remaining'].clip(lower=1)
    df['run_rate_ratio']  = df['rrr']/df['crr'].clip(lower=0.1)
    df['runs_per_wicket'] = df['runs_needed']/df['wickets_left'].clip(lower=1)
    df['balls_per_wicket']= df['balls_remaining']/df['wickets_left'].clip(lower=1)
    df['resource_score']  = (df['wickets_left']/MAX_WICKETS)*(df['balls_remaining']/TOTAL_BALLS)
    df['phase']           = pd.cut(df['overs_done'],bins=[0,6,11,16,20],labels=[0,1,2,3]).astype(float)
    df['batting_won']     = (df['batting_team']==df['winner']).astype(int)

    # Momentum
    df['runs_last_18'] = df.groupby('match_id')['total_runs'].transform(
        lambda x: x.rolling(18,min_periods=1).sum().shift(1).fillna(0))
    df['wkts_last_18'] = df.groupby('match_id')['is_wicket'].transform(
        lambda x: x.rolling(18,min_periods=1).sum().shift(1).fillna(0))

    # Venue
    df = df.merge(venue_tbl[['venue','venue_chase_wr']], on='venue', how='left')
    df['venue_chase_wr'] = df['venue_chase_wr'].fillna(def_venue_wr)

    # Player quality
    if has_players:
        df['batter_sr']       = df['batter'].map(csr_map).fillna(def_sr)
        df['bowler_eco_feat'] = df['bowler'].map(eco_map).fillna(def_eco)
    else:
        df['batter_sr']       = def_sr
        df['bowler_eco_feat'] = def_eco

    # Season weights
    season_w = {'2024':4.0,'2023':3.5,'2022':3.0,'2021':2.5,
                '2020/21':2.0,'2019':1.5,'2018':1.2}
    df['sw'] = df['season'].map(season_w).fillna(1.0)

    df = df[FEATS+['batting_won','match_id','sw']].dropna()
    df = df[(df['balls_remaining']>0)&(df['runs_needed']>=0)]

    match_ids           = df['match_id'].unique()
    train_ids, test_ids = train_test_split(match_ids,test_size=0.2,random_state=42)
    train_df = df[df['match_id'].isin(train_ids)]
    X_train  = train_df[FEATS]; y_train = train_df['batting_won']
    train_w  = train_df['sw'].values

    # ── Train ───────────────────────────────────────────────
    base  = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.04,
        min_samples_leaf=40, l2_regularization=0.1, random_state=42
    )
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    model.fit(X_train, y_train, sample_weight=train_w)

    joblib.dump((model, FEATS), cache_file)
    open(key_file,'w').write(data_key)
    return model, FEATS, "trained", lookups


@st.cache_resource
def load_lookups_only():
    """Fast player lookup for bowler search — always fresh from CSV."""
    if os.path.exists('batsman.csv') and os.path.exists('bowler.csv'):
        bat = pd.read_csv('batsman.csv')
        bwl = pd.read_csv('bowler.csv')
        bowl_eco  = bwl.groupby('player_name')['economy_rate'].mean()
        venue_avg = bat.groupby('stadium_name')['total_team_score'].mean()
        return bowl_eco, venue_avg
    return pd.Series(dtype=float), pd.Series(dtype=float)

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def predict_win(model, lookups, target, score, wickets_fallen, overs, xb=0,
                venue=None, bat_team=None, bowl_team=None,
                last_18_runs=None, last_18_wkts=None):
    balls_done     = int(overs)*6+int(xb)
    balls_rem      = TOTAL_BALLS-balls_done
    overs_done     = balls_done/6
    runs_needed    = max(0,target-score)
    wickets_left   = MAX_WICKETS-wickets_fallen
    crr            = score/max(overs_done,0.1)
    rrr            = (runs_needed*6)/max(balls_rem,1)
    rr_ratio       = rrr/max(crr,0.1)
    runs_per_wkt   = runs_needed/max(wickets_left,1)
    balls_per_wkt  = balls_rem/max(wickets_left,1)
    resource_score = (wickets_left/MAX_WICKETS)*(balls_rem/TOTAL_BALLS)
    if   overs_done < 6:  phase=0.0
    elif overs_done < 11: phase=1.0
    elif overs_done < 16: phase=2.0
    else:                 phase=3.0

    if last_18_runs is None: last_18_runs = round(crr*3.0, 1)
    if last_18_wkts is None: last_18_wkts = 0.0

    venue_wr = lookups['venue_map'].get(venue, lookups['default_venue']) if venue else lookups['default_venue']
    bat_sr   = lookups['team_sr'].get(bat_team,  lookups['default_sr'])  if bat_team  else lookups['default_sr']
    bwl_eco  = lookups['team_eco'].get(bowl_team, lookups['default_eco']) if bowl_team else lookups['default_eco']

    row = pd.DataFrame([[
        runs_needed,balls_rem,wickets_left,crr,rrr,rr_ratio,
        runs_per_wkt,balls_per_wkt,resource_score,phase,
        last_18_runs,last_18_wkts,venue_wr,bat_sr,bwl_eco
    ]],columns=FEATS)
    pb = model.predict_proba(row)[0][1]
    return pb,1-pb,runs_needed,balls_rem,wickets_left,crr,rrr


def simulate_next_over(model, lookups, target, score, wickets_fallen, overs, xb,
                       venue=None, bat_team=None, bowl_team=None, l18r=None, l18w=None):
    """Win probability under 5 next-over scenarios."""
    next_o = overs + 1
    kw = dict(venue=venue, bat_team=bat_team, bowl_team=bowl_team)
    pb_now,*_ = predict_win(model, lookups, target, score, wickets_fallen,
                             overs, xb, **kw, last_18_runs=l18r, last_18_wkts=l18w)
    results = []
    for label, ds, dw, do, dxb in [
        ("🔥 Big over +15",  15, 0, next_o, 0),
        ("✅ Good over +9",   9, 0, next_o, 0),
        ("😐 Avg over +6",    6, 0, next_o, 0),
        ("😟 Tight over +3",  3, 0, next_o, 0),
        ("💀 Wicket",         0, 1, overs,  xb),
    ]:
        p2,*_ = predict_win(model, lookups, target, score+ds, wickets_fallen+dw, do, dxb, **kw)
        results.append((label, p2*100, (p2-pb_now)*100))
    return results


def find_bowler(inp, bowl_eco):
    if not inp or not inp.strip(): return None, None
    s = inp.strip()
    if s in bowl_eco.index: return s, float(bowl_eco[s])
    for n in bowl_eco.index:
        if n.lower()==s.lower(): return n, float(bowl_eco[n])
    matches = [n for n in bowl_eco.index if n.lower().split()[-1]==s.lower()]
    if matches: return matches[0], float(bowl_eco[matches[0]])
    matches = [n for n in bowl_eco.index if s.lower() in n.lower()]
    if matches: return matches[0], float(bowl_eco[matches[0]])
    return None, None


# ── IPL team short-name → full-name aliases ────────────────
TEAM_ALIASES = {
    'RCB':'Royal Challengers Bengaluru','MI':'Mumbai Indians',
    'CSK':'Chennai Super Kings','KKR':'Kolkata Knight Riders',
    'SRH':'Sunrisers Hyderabad','PBKS':'Punjab Kings',
    'RR':'Rajasthan Royals','DC':'Delhi Capitals',
    'GT':'Gujarat Titans','LSG':'Lucknow Super Giants',
    # legacy names
    'Royal Challengers Bangalore':'Royal Challengers Bengaluru',
    'Kings XI Punjab':'Punjab Kings',
    'Delhi Daredevils':'Delhi Capitals',
    'Deccan Chargers':'Sunrisers Hyderabad',
    'Rising Pune Supergiants':'Chennai Super Kings',
    'Pune Warriors':'Kolkata Knight Riders',
}

def normalize_team(name):
    """Map short name / legacy name → current full name."""
    if not name: return name
    n = name.strip()
    return TEAM_ALIASES.get(n, n)


def clean_player_list(raw_str):
    """Parse comma-separated player list, strip role tags like (c), (w)."""
    return [re.sub(r'\([^)]*\)', '', p).strip()
            for p in raw_str.split(',') if p.strip()]


def parse_clg(clg_text):
    """
    Full CLG report parser.
    Returns dict with: team1, team2, team1_xi, team2_xi,
                       batting_team (chasing), bowling_team (defending),
                       toss_winner, toss_choice, venue, pitch_type
    """
    r = {
        'team1': '', 'team2': '',
        'team1_xi': [], 'team2_xi': [],
        'batting_team': '', 'bowling_team': '',
        'toss_winner': '', 'toss_choice': '',
        'venue': '', 'pitch_type': 'balanced',
    }

    # ── 1. Playing XI (both teams) ──────────────────────────
    xi_pattern = re.compile(
        r'([A-Za-z ]{5,50}?)\s*\(Playing XI\)\s*:\s*([^\n]+)', re.IGNORECASE)
    xi_found = xi_pattern.findall(clg_text)
    teams_xi = []
    for team_raw, players_raw in xi_found:
        team  = team_raw.strip()
        xi    = clean_player_list(players_raw)
        if team and xi:
            teams_xi.append((team, xi))

    if len(teams_xi) >= 2:
        r['team1'], r['team1_xi'] = teams_xi[0]
        r['team2'], r['team2_xi'] = teams_xi[1]
    elif len(teams_xi) == 1:
        r['team1'], r['team1_xi'] = teams_xi[0]

    # ── 2. Toss ────────────────────────────────────────────
    toss_m = re.search(
        r'([A-Za-z ]{5,50}?) have won the toss and (?:have )?opted to (bat|field|bowl)',
        clg_text, re.IGNORECASE)
    if toss_m:
        r['toss_winner'] = toss_m.group(1).strip()
        r['toss_choice'] = toss_m.group(2).strip().lower()

    # ── 3. Determine chasing / defending ──────────────────
    # "opted to field/bowl" → toss winner bowls first → bats 2nd = CHASES
    # "opted to bat"        → toss winner bats first  → bowls 2nd = DEFENDS
    if r['toss_winner'] and r['team1'] and r['team2']:
        other = r['team2'] if r['toss_winner'].lower() in r['team1'].lower() else r['team1']
        tw    = r['team1'] if r['toss_winner'].lower() in r['team1'].lower() else r['team2']
        if r['toss_choice'] in ('field', 'bowl'):
            r['batting_team']  = tw     # toss winner bowls first → chases
            r['bowling_team']  = other
        elif r['toss_choice'] == 'bat':
            r['bowling_team']  = tw     # toss winner bats first  → defends
            r['batting_team']  = other

    # ── 4. Venue / Stadium ─────────────────────────────────
    # Try common stadium keyword patterns
    for pat in [
        r'(?:at|in) (?:the )?([A-Z][A-Za-z\. ]{3,40}(?:Stadium|Ground|Gardens|Oval|Arena|Park))',
        r'([A-Z][A-Za-z\. ]{3,40}(?:Stadium|Ground|Gardens|Oval|Arena|Park))',
    ]:
        sm = re.search(pat, clg_text)
        if sm:
            r['venue'] = sm.group(1).strip()
            break

    # ── 5. Pitch type ──────────────────────────────────────
    # Search specifically in pitch report section first
    pitch_section = re.search(
        r'[Pp]itch [Rr]eport.*?(?=\n\n|\Z)', clg_text, re.DOTALL)
    check_text = (pitch_section.group(0) if pitch_section else clg_text).lower()

    batting_kw = ['batting display','batting friendly','flat','good batting','high.scor',
                  'win the toss and chase','chase','boundaries aren','runs']
    bowling_kw = ['seam','swing','bowl','pace','difficult','green top','moisture']
    spin_kw    = ['spin','turn','dry','dust','spinners']

    if any(k in check_text for k in batting_kw):
        r['pitch_type'] = 'batting'
    elif any(k in check_text for k in spin_kw):
        r['pitch_type'] = 'spin-friendly'
    elif any(k in check_text for k in bowling_kw):
        r['pitch_type'] = 'bowling'

    return r


def safe_extract(patterns, text):
    for p in patterns:
        try:
            m = re.search(p, text, re.IGNORECASE)
            if m and m.lastindex and m.lastindex>=1: return m.group(1).strip()
        except: continue
    return None


def calc_edge(ai_prob, odds, over_num):
    if not odds or odds<=1: return None,None,None,False
    imp  = (1/odds)*100
    edge = ai_prob*100-imp
    ev   = (ai_prob*(odds-1)-(1-ai_prob))*100
    val  = edge>EDGE_THRESHOLD and VALID_OVER_MIN<=over_num<=VALID_OVER_MAX and odds>1.20
    return imp,edge,ev,val


def init_log():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            'timestamp','match_id','batting_team','bowling_team',
            'target','score','wickets','over','runs_needed','balls_remaining',
            'crr','rrr','in_valid_window','current_bowler',
            'ai_prob_bat','ai_prob_bowl','market_odds_bat','market_odds_bowl',
            'market_implied_bat','market_implied_bowl','edge_bat','edge_bowl',
            'ev_per_100_bat','ev_per_100_bowl','value_bet_on',
            'actual_winner','bet_outcome','roi'
        ]).to_csv(LOG_FILE,index=False)


def log_prediction(mid,bat,bowl,target,score,wkts,over_str,over_num,
                   pb,pw,rr,br,crr,rrr,cur_bowler=None,oa=None,ob=None,winner=None):
    imp_bat,edge_bat,ev_bat,val_bat   = calc_edge(pb,oa,over_num)
    imp_bowl,edge_bowl,ev_bowl,val_bowl = calc_edge(pw,ob,over_num)
    in_window = VALID_OVER_MIN<=over_num<=VALID_OVER_MAX
    value_on  = None
    if val_bat:  value_on = bat
    if val_bowl: value_on = (value_on+'+'+bowl) if value_on else bowl
    bet_outcome = roi = None
    if winner and value_on:
        wl = winner.strip().lower()
        if val_bat and oa:
            bet_outcome = 'WIN' if bat.lower() in wl else 'LOSS'
            roi = round((oa-1)*100,1) if bet_outcome=='WIN' else -100.0
        if val_bowl and ob and not(val_bat and oa):
            bet_outcome = 'WIN' if bowl.lower() in wl else 'LOSS'
            roi = round((ob-1)*100,1) if bet_outcome=='WIN' else -100.0
    pd.DataFrame([{
        'timestamp'          : datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'match_id'           : mid, 'batting_team':bat, 'bowling_team':bowl,
        'target':target,'score':score,'wickets':wkts,'over':over_str,
        'runs_needed':rr,'balls_remaining':br,'crr':round(crr,2),'rrr':round(rrr,2),
        'in_valid_window':in_window,'current_bowler':cur_bowler,
        'ai_prob_bat':round(pb*100,1),'ai_prob_bowl':round(pw*100,1),
        'market_odds_bat':oa,'market_odds_bowl':ob,
        'market_implied_bat':round(imp_bat,1) if imp_bat else None,
        'market_implied_bowl':round(imp_bowl,1) if imp_bowl else None,
        'edge_bat':round(edge_bat,1) if edge_bat else None,
        'edge_bowl':round(edge_bowl,1) if edge_bowl else None,
        'ev_per_100_bat':round(ev_bat,1) if ev_bat else None,
        'ev_per_100_bowl':round(ev_bowl,1) if ev_bowl else None,
        'value_bet_on':value_on,'actual_winner':winner,
        'bet_outcome':bet_outcome,'roi':roi
    }]).to_csv(LOG_FILE,mode='a',header=False,index=False)

# ══════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════
init_log()

for key,val in [('active',False),('md',{}),('logs',[]),('hist',[]),
                ('result',None),('counter',1),('score_hist',[])]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Cricket AI</div>
    <div class="app-sub">IPL · T20 · Live Predictor v2</div>
    <div><span class="live-pill"><span class="live-dot"></span>LIVE</span></div>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────
with st.spinner("⏳ Loading model... first run takes a few minutes, then instant forever"):
    try:
        model, feats, status, lookups = load_model()
        bowl_eco, venue_avg = load_lookups_only()
        if status=="cached":
            st.success("⚡ Loaded instantly from cache — 15 features active")
        else:
            st.success("✅ Model trained and saved — player + venue + momentum features enabled!")
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("Upload matches.csv and deliveries.csv to get started")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab_match, tab_stats, tab_log = st.tabs(["🏏  MATCH", "📊  STATS", "📋  LOG"])

# ══════════════════════════════════════════════════════════
# TAB 1 — MATCH
# ══════════════════════════════════════════════════════════
with tab_match:

    # ── PRE-MATCH SETUP ────────────────────────────────────
    if not st.session_state.active:

        st.markdown('<div class="sec-label">Match ID</div>', unsafe_allow_html=True)
        mid = st.text_input("", value=f"M{st.session_state.counter}",
                            placeholder="e.g. IPL2026_M1", key="mid",
                            label_visibility="collapsed")

        st.markdown('<div class="sec-label">Pre-Match Commentary</div>', unsafe_allow_html=True)
        clg = st.text_area("",
            height=140,
            placeholder="Paste full Cricket Line Guru report here...\n\nThe AI will auto-extract venue, toss, pitch type and playing XI from it.",
            key="clg", label_visibility="collapsed")

        # ── Parse CLG if pasted ────────────────────────────────
        clg_parsed = {}
        toss_w = toss_c = stadium = pitch_type = ""
        auto_bat = auto_bowl = ""
        bat_xi = bowl_xi = []

        if clg and len(clg) > 50:
            clg_parsed  = parse_clg(clg)
            toss_w      = clg_parsed.get('toss_winner', '')
            toss_c      = clg_parsed.get('toss_choice', '')
            stadium     = clg_parsed.get('venue', '')
            pitch_type  = clg_parsed.get('pitch_type', 'balanced')
            auto_bat    = clg_parsed.get('batting_team', '')   # chasing team
            auto_bowl   = clg_parsed.get('bowling_team', '')   # defending team
            bat_xi      = clg_parsed.get('team1_xi' if clg_parsed.get('team1') == auto_bat else 'team2_xi', [])
            bowl_xi     = clg_parsed.get('team2_xi' if clg_parsed.get('team1') == auto_bat else 'team1_xi', [])

            # ── Extraction card ───────────────────────────────
            if toss_w or stadium or auto_bat:
                venue_wr_pct = lookups['venue_map'].get(stadium, lookups['default_venue']) * 100

                # Build venue partial-match (e.g. "Chinnaswamy" in keys)
                if not lookups['venue_map'].get(stadium):
                    for k,v in lookups['venue_map'].items():
                        if stadium.lower() in k.lower() or k.lower() in stadium.lower():
                            venue_wr_pct = v * 100
                            stadium = k        # use the canonical name from data
                            break

                chase_str = f"{auto_bat[:25]} ← chasing" if auto_bat else "—"
                def_str   = f"{auto_bowl[:25]} ← defending" if auto_bowl else "—"

                # Playing XI rows
                xi_html = ""
                t1 = clg_parsed.get('team1',''); t1_xi = clg_parsed.get('team1_xi',[])
                t2 = clg_parsed.get('team2',''); t2_xi = clg_parsed.get('team2_xi',[])
                if t1_xi:
                    xi_html += f"""
                    <div class="extracted-row" style="flex-direction:column;align-items:flex-start;gap:3px;padding:8px 0">
                        <span class="extracted-key">🏏 {t1[:30]} XI</span>
                        <span style="font-size:12px;color:#374151;line-height:1.6">{' · '.join(t1_xi)}</span>
                    </div>"""
                if t2_xi:
                    xi_html += f"""
                    <div class="extracted-row" style="flex-direction:column;align-items:flex-start;gap:3px;padding:8px 0">
                        <span class="extracted-key">🏏 {t2[:30]} XI</span>
                        <span style="font-size:12px;color:#374151;line-height:1.6">{' · '.join(t2_xi)}</span>
                    </div>"""

                st.markdown(f"""
                <div class="card card-blue" style="margin-top:8px">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;
                                color:#1d4ed8;margin-bottom:10px;font-weight:700">
                        ✅ EXTRACTED FROM CLG
                    </div>
                    <div class="extracted-row">
                        <span class="extracted-key">Venue</span>
                        <span class="extracted-val">{stadium or '—'} <span style="color:#64748b;font-size:11px">(chase win: {venue_wr_pct:.0f}%)</span></span>
                    </div>
                    <div class="extracted-row">
                        <span class="extracted-key">Toss</span>
                        <span class="extracted-val">{toss_w or '—'} won → opted to <b>{toss_c or '—'}</b></span>
                    </div>
                    <div class="extracted-row">
                        <span class="extracted-key">Pitch</span>
                        <span class="extracted-val">{pitch_type}</span>
                    </div>
                    <div class="extracted-row">
                        <span class="extracted-key">Chasing</span>
                        <span class="extracted-val c-green">{chase_str}</span>
                    </div>
                    <div class="extracted-row">
                        <span class="extracted-key">Defending</span>
                        <span class="extracted-val">{def_str}</span>
                    </div>
                    {xi_html}
                </div>
                """, unsafe_allow_html=True)

        # ── After 1st Innings inputs ──────────────────────────
        st.markdown('<div class="sec-label">After 1st Innings</div>', unsafe_allow_html=True)

        # Pre-fill hints from CLG extraction
        if auto_bat:
            st.markdown(f'<div class="hint">✅ Chasing team auto-filled from CLG — edit if needed</div>',
                        unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            bat = st.text_input("Batting Team (chasing)", value=auto_bat,
                                placeholder="e.g. RCB or Royal Challengers Bengaluru",
                                key="bat_inp")
        with c2:
            bowl = st.text_input("Bowling Team (defending)", value=auto_bowl,
                                 placeholder="e.g. SRH or Sunrisers Hyderabad",
                                 key="bowl_inp")

        target = st.number_input("Target (runs to chase)", min_value=1, max_value=400,
                                  value=180, step=1, key="target_inp")

        # ── Team quality preview ──────────────────────────────
        if bat and bowl:
            bat_n = normalize_team(bat.strip())
            bowl_n = normalize_team(bowl.strip())
            bat_csr_v = lookups['team_sr'].get(bat_n, lookups['team_sr'].get(bat.strip(), lookups['default_sr']))
            bwl_eco_v = lookups['team_eco'].get(bowl_n, lookups['team_eco'].get(bowl.strip(), lookups['default_eco']))
            v_score_v = float(venue_avg.get(stadium, 165.0)) if stadium else 165.0

            # Use extracted XI for player-level SR/ECO
            if bat_xi:
                csrs = [lookups['team_sr'].get(p, lookups['default_sr']) for p in bat_xi]
                bat_csr_v = float(np.mean(csrs))
            if bowl_xi:
                ecos = [lookups['team_eco'].get(p, lookups['default_eco']) for p in bowl_xi]
                bwl_eco_v = float(np.mean(ecos))

            xi_note = " (from Playing XI)" if bat_xi or bowl_xi else " (team average)"
            st.markdown(f"""
            <div class="card" style="margin-top:4px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:#4b5563;margin-bottom:8px">
                    TEAM QUALITY SCORES{xi_note}
                </div>
                <div class="extracted-row">
                    <span class="extracted-key">Chase SR</span>
                    <span class="extracted-val c-green">{bat_csr_v:.0f} <span style="color:#4b5563;font-size:11px">({bat[:20]})</span></span>
                </div>
                <div class="extracted-row">
                    <span class="extracted-key">Bowl Eco</span>
                    <span class="extracted-val c-green">{bwl_eco_v:.1f} <span style="color:#4b5563;font-size:11px">({bowl[:20]})</span></span>
                </div>
                <div class="extracted-row">
                    <span class="extracted-key">Venue avg</span>
                    <span class="extracted-val">{v_score_v:.0f} runs</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏏  START TRACKING MATCH", key="start_btn"):
            if not bat or not bowl:
                st.error("Enter both team names!")
            elif bat.strip().lower() == bowl.strip().lower():
                st.error("Teams cannot be the same!")
            else:
                bat_n  = normalize_team(bat.strip())
                bowl_n = normalize_team(bowl.strip())
                st.session_state.active = True
                st.session_state.md     = {
                    'mid': mid, 'bat': bat_n, 'bowl': bowl_n,
                    'bat_display': bat.strip(), 'bowl_display': bowl.strip(),
                    'target': target, 'clg': clg, 'stadium': stadium,
                    'toss_w': toss_w, 'toss_c': toss_c, 'pitch': pitch_type,
                    'bat_xi': bat_xi, 'bowl_xi': bowl_xi,
                }
                st.session_state.logs       = []
                st.session_state.hist       = []
                st.session_state.result     = None
                st.session_state.score_hist = []
                st.session_state.counter   += 1
                st.rerun()

    # ── LIVE MATCH ──────────────────────────────────────────
    else:
        md  = st.session_state.md
        bat = md['bat']; bowl = md['bowl']; target = md['target']
        venue = md.get('stadium') or None

        # Match bar
        st.markdown(f"""
        <div class="match-bar">
            <div>
                <div class="match-teams">{bat} <span style="color:#4b5563">vs</span> {bowl}</div>
                <div class="match-meta">TARGET {target} · {md['mid']} · {md.get('pitch','—').upper()}</div>
            </div>
            <span class="live-pill"><span class="live-dot"></span>LIVE</span>
        </div>
        """, unsafe_allow_html=True)

        # ── SCORE ENTRY FORM ────────────────────────────────
        with st.form(key="score_form", clear_on_submit=False):
            st.markdown('<div class="sec-label">Over Update</div>', unsafe_allow_html=True)

            c1,c2 = st.columns(2)
            with c1:
                score = st.number_input("📊 Score", min_value=0, max_value=500,
                                         value=0, step=1, key="score_k")
            with c2:
                wkts  = st.number_input("💀 Wickets", min_value=0, max_value=10,
                                         value=0, step=1, key="wkts_k")

            c3,c4 = st.columns(2)
            with c3:
                overs = st.number_input("⏱ Full Overs", min_value=0, max_value=20,
                                         value=0, step=1, key="overs_k")
            with c4:
                balls = st.number_input("🏏 Balls", min_value=0, max_value=5,
                                         value=0, step=1, key="balls_k")

            # Live preview inside form
            if overs>0 or balls>0:
                bd = overs*6+balls; br_p = TOTAL_BALLS-bd
                od = bd/6 if bd>0 else 0.1
                crr_p = score/od if od>0 else 0
                rrr_p = (max(0,target-score)*6)/max(br_p,1)
                rr_p  = max(0,target-score)
                rrr_color = '#f87171' if rrr_p>crr_p*1.3 else '#fbbf24' if rrr_p>crr_p else '#34d399'
                st.markdown(f"""
                <div class="stat-grid" style="margin:8px 0 4px">
                    <div class="stat-box">
                        <div class="stat-val" style="color:#34d399">{rr_p}</div>
                        <div class="stat-lbl">Need</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val" style="color:#60a5fa">{crr_p:.1f}</div>
                        <div class="stat-lbl">CRR</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val" style="color:{rrr_color}">{rrr_p:.1f}</div>
                        <div class="stat-lbl">RRR</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sec-label">Bowler (optional)</div>', unsafe_allow_html=True)
            bowler_inp = st.text_input("", placeholder="Type surname: Bumrah, Shami, Chahal...",
                                        key="bowler_k", label_visibility="collapsed")
            if bowler_inp:
                matched,eco = find_bowler(bowler_inp, bowl_eco)
                if matched:
                    st.markdown(f'<div class="hint">✅ {matched} · Economy {eco:.2f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hint" style="color:#f87171">⚠️ Not found — try surname only</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-label">Market Odds (optional)</div>', unsafe_allow_html=True)
            co1,co2 = st.columns(2)
            with co1:
                odds_bat  = st.number_input(f"{bat[:18]}", min_value=0.0, max_value=99.0,
                                             value=0.0, step=0.01, format="%.2f", key="ob_k")
            with co2:
                odds_bowl = st.number_input(f"{bowl[:18]}", min_value=0.0, max_value=99.0,
                                             value=0.0, step=0.01, format="%.2f", key="ow_k")

            st.markdown('<div class="sec-label">Winner (fill when match ends)</div>', unsafe_allow_html=True)
            winner_inp = st.text_input("", placeholder="Enter winner after match ends",
                                        key="win_k", label_visibility="collapsed")

            fb1,fb2 = st.columns([3,1])
            with fb1: pred_btn = st.form_submit_button("⚡  PREDICT WIN PROBABILITY")
            with fb2: end_btn  = st.form_submit_button("🏁 END")

        # ── END MATCH ────────────────────────────────────────
        if end_btn:
            st.session_state.active     = False
            st.session_state.score_hist = []
            st.rerun()

        # ── PREDICT ──────────────────────────────────────────
        if pred_btn:
            auto_winner = None
            if score>=target:  auto_winner=bat
            elif wkts>=10:     auto_winner=bowl
            elif overs>=20:    auto_winner=bat if score>=target else bowl

            # Auto-compute momentum from score history
            ball_num = overs*6 + balls
            sh = st.session_state.score_hist
            prev = next((e for e in reversed(sh) if e[0] <= ball_num-18), None)
            if prev and ball_num > 18:
                l18r = max(0, score - prev[1])
                l18w = max(0, wkts  - prev[2])
            else:
                l18r = l18w = None

            st.session_state.score_hist.append((ball_num, score, wkts))

            pb,pw,rr,br,wl,crr,rrr = predict_win(
                model, lookups, target, score, wkts, overs, balls,
                venue=venue, bat_team=bat, bowl_team=bowl,
                last_18_runs=l18r, last_18_wkts=l18w
            )

            matched_bwl,cur_eco = find_bowler(bowler_inp, bowl_eco) if bowler_inp else (None,None)
            oa = odds_bat  if odds_bat>0  else None
            ob = odds_bowl if odds_bowl>0 else None
            winner = winner_inp.strip() if winner_inp.strip() else auto_winner

            st.session_state.result = {
                'pb':pb,'pw':pw,'rr':rr,'br':br,'wl':wl,'crr':crr,'rrr':rrr,
                'score':score,'wkts':wkts,'overs':overs,'balls':balls,
                'over_str':f"{overs}.{balls}",'over_num':overs,
                'oa':oa,'ob':ob,'matched_bwl':matched_bwl,'cur_eco':cur_eco,
                'auto_winner':auto_winner,'winner':winner,
                'l18r':l18r,'l18w':l18w,
            }
            st.session_state.logs.append(st.session_state.result)
            st.session_state.hist.append({'p':pb,'l':f"O{overs}"})
            if len(st.session_state.hist)>12: st.session_state.hist.pop(0)

            log_prediction(md['mid'],bat,bowl,target,score,wkts,
                          f"{overs}.{balls}",overs,pb,pw,rr,br,crr,rrr,
                          matched_bwl,oa,ob,winner)

            if auto_winner:
                st.session_state.active     = False
                st.session_state.score_hist = []
                st.balloons()
                st.rerun()

        # ── RESULTS ──────────────────────────────────────────
        if st.session_state.result:
            r  = st.session_state.result
            pb = r['pb']; pw = r['pw']
            pct = int(pb*100)

            st.markdown('<div class="sec-label">Prediction</div>', unsafe_allow_html=True)

            # Probability bar
            st.markdown(f"""
            <div class="card">
                <div class="prob-header">
                    <div>
                        <div class="prob-team">{bat}</div>
                        <div class="prob-pct-bat">{pct}%</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4b5563;letter-spacing:1px">WIN CHANCE</div>
                    </div>
                    <div style="text-align:right">
                        <div class="prob-team">{bowl}</div>
                        <div class="prob-pct-bowl">{100-pct}%</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4b5563;letter-spacing:1px;text-align:right">WIN CHANCE</div>
                    </div>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Verdict (CSS class names now match the stylesheet)
            if pb>=0.68:
                vc,vi,vt,vd = "verdict-good","🚀","CHASE ON TRACK",f"{bat} favourites — RRR {r['rrr']:.1f} vs CRR {r['crr']:.1f}"
            elif pb>=0.48:
                vc,vi,vt,vd = "verdict-mid","⚖️","GAME IN THE BALANCE",f"Could go either way — {r['rr']} off {r['br']} balls"
            elif pb>=0.28:
                vc,vi,vt,vd = "verdict-bad","⚠️","TOUGH CHASE",f"{bowl} ahead — need {r['rrr']:.1f} RPO with {r['wl']} wickets"
            else:
                vc,vi,vt,vd = "verdict-bad","❌","CHASE LOOKS OVER",f"Very tough — need {r['rrr']:.1f} RPO from here"

            st.markdown(f"""
            <div class="verdict {vc}">
                <div class="verdict-icon">{vi}</div>
                <div><div class="verdict-title">{vt}</div><div class="verdict-desc">{vd}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Stats grid
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-box"><div class="stat-val" style="color:#34d399">{r['rr']}</div><div class="stat-lbl">Runs Needed</div></div>
                <div class="stat-box"><div class="stat-val" style="color:#60a5fa">{r['br']}</div><div class="stat-lbl">Balls Left</div></div>
                <div class="stat-box"><div class="stat-val" style="color:#fbbf24">{r['wl']}</div><div class="stat-lbl">Wkts Left</div></div>
            </div>
            <div class="stat-grid">
                <div class="stat-box"><div class="stat-val" style="color:#34d399">{r['crr']:.2f}</div><div class="stat-lbl">Current RR</div></div>
                <div class="stat-box"><div class="stat-val" style="color:#f87171">{r['rrr']:.2f}</div><div class="stat-lbl">Required RR</div></div>
                <div class="stat-box"><div class="stat-val" style="color:#9ca3af">{r['over_str']}</div><div class="stat-lbl">Over</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Momentum badge
            if r.get('l18r') is not None:
                mo_color = '#16a34a' if r['l18w']==0 else '#f87171'
                st.markdown(f"""
                <div class="card" style="margin-top:8px;padding:10px 14px">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:#64748b;margin-bottom:4px">LAST 3 OVERS (AUTO)</div>
                    <div style="display:flex;gap:20px">
                        <span style="font-family:'Bebas Neue',sans-serif;font-size:22px;color:#34d399">{r['l18r']} <span style="font-size:10px;font-family:'JetBrains Mono';color:#64748b">RUNS</span></span>
                        <span style="font-family:'Bebas Neue',sans-serif;font-size:22px;color:{mo_color}">{r['l18w']} <span style="font-size:10px;font-family:'JetBrains Mono';color:#64748b">WKTS</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Bowler info
            if r.get('matched_bwl') and r.get('cur_eco'):
                st.markdown(f"""
                <div class="card" style="margin-top:8px">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4b5563;margin-bottom:6px;letter-spacing:1px">CURRENT BOWLER</div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:14px;font-weight:600">{r['matched_bwl']}</span>
                        <span style="font-family:'Bebas Neue',sans-serif;font-size:22px;color:#34d399">{r['cur_eco']:.2f} <span style="font-size:11px;font-family:'JetBrains Mono';color:#4b5563">ECO</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Scenario Simulator (NEW) ──────────────────
            if r['overs'] < 19:
                st.markdown('<div class="sec-label">Next Over Scenarios</div>', unsafe_allow_html=True)
                scenarios = simulate_next_over(
                    model, lookups, target, r['score'], r['wkts'],
                    r['overs'], r['balls'], venue=venue,
                    bat_team=bat, bowl_team=bowl,
                    l18r=r.get('l18r'), l18w=r.get('l18w')
                )
                grid_html = '<div class="scenario-grid">'
                for label, pct2, delta in scenarios:
                    color = '#16a34a' if delta > 0 else '#dc2626'
                    arrow = '↑' if delta > 0 else '↓'
                    grid_html += f"""
                    <div class="scenario-box">
                        <div class="scenario-label">{label}</div>
                        <div class="scenario-pct" style="color:{color}">{pct2:.0f}%</div>
                        <div class="scenario-delta" style="color:{color}">{delta:+.1f}% {arrow}</div>
                    </div>"""
                grid_html += '</div>'
                st.markdown(grid_html, unsafe_allow_html=True)

            # ── Edge Detection ────────────────────────────
            if r['oa'] or r['ob']:
                st.markdown('<div class="sec-label">Edge Detection</div>', unsafe_allow_html=True)
                for ai,odds,name in [(pb,r['oa'],bat),(pw,r['ob'],bowl)]:
                    if not odds or odds <= 0: continue
                    imp,edge,ev,is_val = calc_edge(ai,odds,r['over_num'])
                    if edge is None: continue
                    ec = "edge-pos" if edge>0 else "edge-neg"
                    ow = r['over_num']
                    in_w = VALID_OVER_MIN<=ow<=VALID_OVER_MAX

                    value_html = ""
                    if is_val:
                        value_html = f"""
                        <div class="value-alert">
                            <div class="value-alert-title">✅ VALUE BET DETECTED</div>
                            <div class="value-alert-sub">Edge &gt;{EDGE_THRESHOLD}% in valid window — log &amp; verify after match</div>
                        </div>"""
                    else:
                        reason = f"Outside valid window (over {ow})" if not in_w else f"Edge below {EDGE_THRESHOLD}% threshold — skip"
                        value_html = f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;margin-top:6px">⚪ {reason}</div>'

                    st.markdown(f"""
                    <div class="edge-box">
                        <div class="edge-name">{name}</div>
                        <div class="edge-row">
                            <span class="edge-lbl">AI Probability</span>
                            <span class="edge-val">{ai*100:.1f}%</span>
                        </div>
                        <div class="edge-row">
                            <span class="edge-lbl">Market Implies</span>
                            <span class="edge-val">{imp:.1f}% <span style="color:#94a3b8">(odds {odds})</span></span>
                        </div>
                        <div class="edge-row">
                            <span class="edge-lbl">Edge</span>
                            <span class="edge-val {ec}">{edge:+.1f}%</span>
                        </div>
                        <div class="edge-row">
                            <span class="edge-lbl">EV / ₹100</span>
                            <span class="edge-val {ec}">₹{ev:+.1f}</span>
                        </div>
                        {value_html}
                    </div>
                    """, unsafe_allow_html=True)

            # Momentum chart
            if len(st.session_state.hist)>1:
                st.markdown('<div class="sec-label">Probability Trend</div>', unsafe_allow_html=True)
                hdf = pd.DataFrame(st.session_state.hist)
                hdf['Win %'] = hdf['p']*100
                st.line_chart(hdf.set_index('l')['Win %'], height=130)

# ══════════════════════════════════════════════════════════
# TAB 2 — STATS
# ══════════════════════════════════════════════════════════
with tab_stats:
    st.markdown('<div class="sec-label">Backtest Report</div>', unsafe_allow_html=True)

    if not os.path.exists(LOG_FILE) or len(pd.read_csv(LOG_FILE))==0:
        st.info("No predictions logged yet. Start tracking matches!")
    else:
        log = pd.read_csv(LOG_FILE)
        c1,c2,c3 = st.columns(3)
        c1.metric("Matches",    log['match_id'].nunique())
        c2.metric("Overs",      len(log))
        c3.metric("Value Bets", log['value_bet_on'].notna().sum())

        done = log[log['bet_outcome'].notna()]
        if len(done)>0:
            st.markdown('<div class="sec-label">Bet Results</div>', unsafe_allow_html=True)
            wins  = (done['bet_outcome']=='WIN').sum()
            loss  = (done['bet_outcome']=='LOSS').sum()
            total = done['roi'].sum()
            avg   = done['roi'].mean()
            wr    = wins/len(done)*100
            c4,c5,c6 = st.columns(3)
            c4.metric("Win Rate",  f"{wr:.1f}%")
            c5.metric("Total ROI", f"₹{total:+.0f}")
            c6.metric("Avg/Bet",   f"₹{avg:+.1f}")
            if total>0: st.success("✅ Profitable so far — need 50+ bets to confirm real edge")
            else:       st.warning("Need more data — keep logging consistently")

        st.markdown('<div class="sec-label">Model Config</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="extracted-row">
                <span class="extracted-key">Engine</span>
                <span class="extracted-val">HistGradientBoosting v2 · 15 features</span>
            </div>
            <div class="extracted-row">
                <span class="extracted-key">Edge</span>
                <span class="extracted-val c-green">{EDGE_THRESHOLD}% threshold</span>
            </div>
            <div class="extracted-row">
                <span class="extracted-key">Valid overs</span>
                <span class="extracted-val">{VALID_OVER_MIN}–{VALID_OVER_MAX} only</span>
            </div>
            <div class="extracted-row">
                <span class="extracted-key">Features</span>
                <span class="extracted-val">Core + Player + Venue + Momentum + Season wt</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — LOG
# ══════════════════════════════════════════════════════════
with tab_log:
    st.markdown('<div class="sec-label">Prediction Log</div>', unsafe_allow_html=True)

    if not os.path.exists(LOG_FILE):
        st.info("No log yet.")
    else:
        log = pd.read_csv(LOG_FILE)
        if len(log)==0:
            st.info("No entries yet.")
        else:
            c1,c2 = st.columns(2)
            vonly = c1.checkbox("Value bets only")
            wonly = c2.checkbox("Valid window only")
            filtered = log.copy()
            if vonly: filtered = filtered[filtered['value_bet_on'].notna()]
            if wonly: filtered = filtered[filtered['in_valid_window']==True]

            st.dataframe(
                filtered[['timestamp','match_id','batting_team','bowling_team',
                          'over','score','wickets','ai_prob_bat',
                          'edge_bat','value_bet_on','bet_outcome','roi']].tail(50),
                hide_index=True, use_container_width=True
            )
            csv = log.to_csv(index=False)
            st.download_button("📥 Download Full Log CSV", data=csv,
                               file_name="predictions_log.csv", mime="text/csv")

st.markdown(
    '<div class="disclaimer">⚠️ Educational only · Betting largely illegal in India · Not financial advice</div>',
    unsafe_allow_html=True
)
