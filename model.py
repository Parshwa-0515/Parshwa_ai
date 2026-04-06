import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings, os, datetime, joblib
warnings.filterwarnings('ignore')

# ── CONSTANTS ──────────────────────────────────────────────
TOTAL_BALLS   = 120
MAX_WICKETS   = 10
MODEL_VERSION = 'v2'          # bump this when features change → auto-invalidates cache

LOG_FILE = 'predictions_log.csv'

# ── CONFIG (change to tune the system) ─────────────────────
EDGE_THRESHOLD  = 10          # minimum edge % to flag as value bet
VALID_OVER_MIN  = 5           # ignore overs before this
VALID_OVER_MAX  = 15          # ignore overs after this

# ══════════════════════════════════════════════════════════
# STEP 1 — Load & Clean
# ══════════════════════════════════════════════════════════
print("Loading data...")
matches    = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
print(f"Matches: {len(matches)} | Deliveries: {len(deliveries)}")

matches = matches[matches['super_over'] == 'N']
matches = matches[matches['method'].isna()]
matches = matches[matches['winner'].notna() & (matches['winner'] != '')]
matches = matches[matches['target_runs'].notna()]

# ── Player lookup tables ────────────────────────────────────
try:
    batsman_df = pd.read_csv('batsman.csv')
    bowler_df  = pd.read_csv('bowler.csv')
    HAS_PLAYERS = True
    print(f"Players: {batsman_df['player_name'].nunique()} batters | "
          f"{bowler_df['player_name'].nunique()} bowlers loaded")
except FileNotFoundError:
    HAS_PLAYERS = False
    print("⚠️  batsman.csv / bowler.csv not found — player features will use defaults")

# ══════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering
# ══════════════════════════════════════════════════════════
keep = matches[['id','winner','target_runs','venue','season']].rename(columns={'id':'match_id'})
df   = deliveries.merge(keep, on='match_id', how='inner')
df   = df[df['inning'] == 2].copy()
df   = df.sort_values(['match_id','over','ball'])

df['cum_runs']        = df.groupby('match_id')['total_runs'].cumsum()
df['cum_wickets']     = df.groupby('match_id')['is_wicket'].cumsum()
df['ball_num']        = df.groupby('match_id').cumcount() + 1
df = df[df['ball_num'] <= TOTAL_BALLS]
df['balls_remaining'] = TOTAL_BALLS - df['ball_num']
df['overs_done']      = df['ball_num'] / 6
df['runs_needed']     = df['target_runs'] - df['cum_runs']
df['wickets_left']    = MAX_WICKETS - df['cum_wickets']
df['crr']             = df['cum_runs'] / df['overs_done'].clip(lower=0.1)
df['rrr']             = (df['runs_needed'] * 6) / df['balls_remaining'].clip(lower=1)
df['run_rate_ratio']  = df['rrr'] / df['crr'].clip(lower=0.1)
df['runs_per_wicket'] = df['runs_needed'] / df['wickets_left'].clip(lower=1)
df['balls_per_wicket']= df['balls_remaining'] / df['wickets_left'].clip(lower=1)
df['resource_score']  = (df['wickets_left'] / MAX_WICKETS) * (df['balls_remaining'] / TOTAL_BALLS)
df['phase']           = pd.cut(df['overs_done'], bins=[0,6,11,16,20],
                               labels=[0,1,2,3]).astype(float)
df['batting_won']     = (df['batting_team'] == df['winner']).astype(int)

# ── NEW ① Momentum: runs & wickets in last 3 overs (18 balls) ──
# shift(1) ensures no leakage — only history BEFORE the current ball is used
print("Computing momentum features...")
df['runs_last_18'] = df.groupby('match_id')['total_runs'].transform(
    lambda x: x.rolling(18, min_periods=1).sum().shift(1).fillna(0))
df['wkts_last_18'] = df.groupby('match_id')['is_wicket'].transform(
    lambda x: x.rolling(18, min_periods=1).sum().shift(1).fillna(0))

# ── NEW ② Venue chase win rate ──────────────────────────────
chase_team = (deliveries[deliveries['inning']==2]
              .groupby('match_id')['batting_team'].first().reset_index()
              .rename(columns={'batting_team':'chasing_team'}))
m_v = matches.merge(chase_team, left_on='id', right_on='match_id', how='left')
m_v['chase_win'] = (m_v['winner'] == m_v['chasing_team']).astype(float)
venue_tbl = (m_v.groupby('venue')
             .agg(venue_chase_wr=('chase_win','mean'), n=('id','count'))
             .query('n >= 5').reset_index())
VENUE_STATS_MAP  = venue_tbl.set_index('venue')['venue_chase_wr'].to_dict()
DEFAULT_VENUE_WR = 0.50

df = df.merge(venue_tbl[['venue','venue_chase_wr']], on='venue', how='left')
df['venue_chase_wr'] = df['venue_chase_wr'].fillna(DEFAULT_VENUE_WR)

# ── NEW ③ Player quality features ─────────────────────────
if HAS_PLAYERS:
    # Ball-level: actual batter/bowler at each delivery
    csr_map = (batsman_df[batsman_df['innings_number']==2]
               .groupby('player_name')['strike_rate'].mean())
    eco_map = bowler_df.groupby('player_name')['economy_rate'].mean()
    # Team-level averages (used at prediction time when we only know team name)
    TEAM_CHASE_SR = (batsman_df[batsman_df['innings_number']==2]
                     .groupby('player_team')['strike_rate'].mean())
    TEAM_BOWL_ECO = bowler_df.groupby('player_team')['economy_rate'].mean()
    DEFAULT_SR    = float(csr_map.mean())
    DEFAULT_ECO   = float(eco_map.mean())
    df['batter_sr']       = df['batter'].map(csr_map).fillna(DEFAULT_SR)
    df['bowler_eco_feat'] = df['bowler'].map(eco_map).fillna(DEFAULT_ECO)
else:
    DEFAULT_SR  = 111.7
    DEFAULT_ECO = 8.5
    TEAM_CHASE_SR = pd.Series(dtype=float)
    TEAM_BOWL_ECO = pd.Series(dtype=float)
    df['batter_sr']       = DEFAULT_SR
    df['bowler_eco_feat'] = DEFAULT_ECO

# ── NEW ④ Season weights — recent IPL matters more ─────────
SEASON_W = {'2024':4.0,'2023':3.5,'2022':3.0,'2021':2.5,
            '2020/21':2.0,'2019':1.5,'2018':1.2}
df['sw'] = df['season'].map(SEASON_W).fillna(1.0)

# ── Full feature list (15 features vs 10 before) ───────────
FEATS = [
    # Core features (original 10)
    'runs_needed', 'balls_remaining', 'wickets_left',
    'crr', 'rrr', 'run_rate_ratio',
    'runs_per_wicket', 'balls_per_wicket', 'resource_score', 'phase',
    # NEW: Momentum
    'runs_last_18', 'wkts_last_18',
    # NEW: Venue
    'venue_chase_wr',
    # NEW: Player quality
    'batter_sr', 'bowler_eco_feat',
]

df = df[FEATS + ['batting_won','match_id','sw']].dropna()
df = df[(df['balls_remaining'] > 0) & (df['runs_needed'] >= 0)]

# ══════════════════════════════════════════════════════════
# STEP 3 — Split by match ID (zero leakage)
# ══════════════════════════════════════════════════════════
match_ids           = df['match_id'].unique()
train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
train_df = df[df['match_id'].isin(train_ids)]
test_df  = df[df['match_id'].isin(test_ids)]
print(f"Train: {len(train_ids)} matches | Test: {len(test_ids)} matches")

X_train = train_df[FEATS]; y_train = train_df['batting_won']
X_test  = test_df[FEATS];  y_test  = test_df['batting_won']
train_w = train_df['sw'].values

# ══════════════════════════════════════════════════════════
# STEP 4 — Train + Calibrate (with caching)
# ══════════════════════════════════════════════════════════
CACHE_FILE = 'model_cache.pkl'
CACHE_KEY  = f"{MODEL_VERSION}_{len(df)}"   # changes if version or data changes
KEY_FILE   = 'model_cache_key.txt'

cache_valid = (
    os.path.exists(CACHE_FILE) and
    os.path.exists(KEY_FILE) and
    open(KEY_FILE).read().strip() == CACHE_KEY
)

if cache_valid:
    print("⚡ Loading cached model — startup instant!")
    print("   (Delete model_cache.pkl to force retrain)")
    model, FEATS = joblib.load(CACHE_FILE)
    auc = "cached"
else:
    print("🔄 Training improved model (HistGBDT + 15 features)...")
    print("   This is ~5× faster than the old GBT. Please wait...")
    # HistGradientBoostingClassifier: faster, handles NaN, better regularization
    base = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.04,
        min_samples_leaf=40, l2_regularization=0.1,
        random_state=42
    )
    # Calibration ensures "70%" actually means wins 70% of the time
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    model.fit(X_train, y_train, sample_weight=train_w)
    joblib.dump((model, FEATS), CACHE_FILE)
    open(KEY_FILE, 'w').write(CACHE_KEY)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f"✅ Done! AUC: {auc:.3f}  (baseline was ~0.80, target >0.85)")
    print("   Model saved — next startup will be instant!")

# ── Calibration check ─────────────────────────────────────
probs               = model.predict_proba(X_test)[:,1]
frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=5)
print("\nCalibration (predicted% vs actual win%):")
print(f"  {'Predicted':>10}  {'Actual':>8}  Gap     Status")
for p, a in zip(mean_pred, frac_pos):
    gap    = (p - a) * 100
    status = '✅ Good' if abs(gap) < 5 else ('⚠️  OK' if abs(gap) < 10 else '❌ Bad')
    print(f"  {p*100:>9.0f}%  {a*100:>7.0f}%  {gap:+5.1f}%  {status}")

print(f"\nConfig: Edge threshold={EDGE_THRESHOLD}% | Valid overs={VALID_OVER_MIN}-{VALID_OVER_MAX}")
print(f"Features ({len(FEATS)}): {', '.join(FEATS)}")
print("-" * 52)

# ══════════════════════════════════════════════════════════
# STEP 5 — Predict
# New optional params: venue, bat_team, bowl_team, last_18_runs, last_18_wkts
# All default gracefully — the model still works without them
# ══════════════════════════════════════════════════════════
def predict(target, score, wickets_fallen, overs, xb=0,
            venue=None, bat_team=None, bowl_team=None,
            last_18_runs=None, last_18_wkts=None):
    balls_done     = int(overs) * 6 + int(xb)
    balls_rem      = TOTAL_BALLS - balls_done
    overs_done     = balls_done / 6
    runs_needed    = max(0, target - score)
    wickets_left   = MAX_WICKETS - wickets_fallen
    crr            = score / max(overs_done, 0.1)
    rrr            = (runs_needed * 6) / max(balls_rem, 1)
    rr_ratio       = rrr / max(crr, 0.1)
    runs_per_wkt   = runs_needed / max(wickets_left, 1)
    balls_per_wkt  = balls_rem / max(wickets_left, 1)
    resource_score = (wickets_left / MAX_WICKETS) * (balls_rem / TOTAL_BALLS)
    if   overs_done < 6:  phase = 0.0
    elif overs_done < 11: phase = 1.0
    elif overs_done < 16: phase = 2.0
    else:                 phase = 3.0

    # Momentum — use provided history or estimate from current CRR
    if last_18_runs is None: last_18_runs = round(crr * 3.0, 1)
    if last_18_wkts is None: last_18_wkts = 0.0

    # Venue — historical chase win rate at this ground
    venue_wr = VENUE_STATS_MAP.get(venue, DEFAULT_VENUE_WR) if venue else DEFAULT_VENUE_WR

    # Team quality — use team-level averages at prediction time
    bat_sr  = float(TEAM_CHASE_SR.get(bat_team,  DEFAULT_SR))  if bat_team  else DEFAULT_SR
    bwl_eco = float(TEAM_BOWL_ECO.get(bowl_team, DEFAULT_ECO)) if bowl_team else DEFAULT_ECO

    row = pd.DataFrame([[
        runs_needed, balls_rem, wickets_left,
        crr, rrr, rr_ratio,
        runs_per_wkt, balls_per_wkt, resource_score, phase,
        last_18_runs, last_18_wkts, venue_wr, bat_sr, bwl_eco
    ]], columns=FEATS)

    pb = model.predict_proba(row)[0][1]
    return pb, 1-pb, runs_needed, balls_rem, wickets_left, crr, rrr


# ══════════════════════════════════════════════════════════
# STEP 6 — Scenario Simulator (NEW)
# Shows win probability under 5 different next-over outcomes
# This is the "what-if" tool for betting decisions
# ══════════════════════════════════════════════════════════
def simulate_next_over(target, score, wickets_fallen, overs, xb=0,
                       venue=None, bat_team=None, bowl_team=None,
                       l18r=None, l18w=None):
    """Return win probability under 5 likely next-over scenarios."""
    next_o = overs + 1
    kw = dict(venue=venue, bat_team=bat_team, bowl_team=bowl_team)
    pb_now, *_ = predict(target, score, wickets_fallen, overs, xb,
                         **kw, last_18_runs=l18r, last_18_wkts=l18w)
    results = []
    for label, ds, dw, do, dxb in [
        ("🔥 Big over   (+15 runs)", 15, 0, next_o, 0),
        ("✅ Good over   (+9 runs)", 9,  0, next_o, 0),
        ("😐 Avg over    (+6 runs)", 6,  0, next_o, 0),
        ("😟 Tight over  (+3 runs)", 3,  0, next_o, 0),
        ("💀 Wicket next ball",      0,  1, overs, xb),
    ]:
        p2, *_ = predict(target, score+ds, wickets_fallen+dw, do, dxb, **kw)
        delta = (p2 - pb_now) * 100
        results.append((label, p2*100, delta))
    return results


# ══════════════════════════════════════════════════════════
# STEP 7 — Edge check
# ══════════════════════════════════════════════════════════
def check_edge(ai_prob, odds, team_name, over_num):
    if not odds: return None, None, None, False
    implied    = (1 / odds) * 100
    edge       = ai_prob * 100 - implied
    ev_per_100 = (ai_prob * (odds - 1) - (1 - ai_prob)) * 100
    in_valid   = VALID_OVER_MIN <= over_num <= VALID_OVER_MAX
    is_value   = (edge > EDGE_THRESHOLD) and in_valid and (odds > 1.20)
    return implied, edge, ev_per_100, is_value


# ══════════════════════════════════════════════════════════
# STEP 8 — Log
# ══════════════════════════════════════════════════════════
def init_log():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            'timestamp','match_id','batting_team','bowling_team',
            'target','score','wickets','over',
            'runs_needed','balls_remaining','crr','rrr',
            'in_valid_window',
            'ai_prob_bat','ai_prob_bowl',
            'market_odds_bat','market_odds_bowl',
            'market_implied_bat','market_implied_bowl',
            'edge_bat','edge_bowl',
            'ev_per_100_bat','ev_per_100_bowl',
            'value_bet_on',
            'actual_winner','bet_outcome','roi'
        ]).to_csv(LOG_FILE, index=False)
        print(f"Log created: {LOG_FILE}\n")


def log_row(mid, bat, bowl, target, score, wkts, over_str, over_num,
            pb, pw, rr, br, crr, rrr, oa=None, ob=None, winner=None):
    imp_bat,  edge_bat,  ev_bat,  val_bat  = check_edge(pb, oa, bat,  over_num)
    imp_bowl, edge_bowl, ev_bowl, val_bowl = check_edge(pw, ob, bowl, over_num)
    in_window = VALID_OVER_MIN <= over_num <= VALID_OVER_MAX
    value_on  = None
    if val_bat:  value_on = bat
    if val_bowl: value_on = (value_on + '+' + bowl) if value_on else bowl
    bet_outcome = roi = None
    if winner and value_on:
        wl = winner.strip().lower()
        if val_bat and oa:
            bet_outcome = 'WIN' if bat.lower() in wl else 'LOSS'
            roi = round((oa - 1) * 100, 1) if bet_outcome == 'WIN' else -100.0
        if val_bowl and ob and not (val_bat and oa):
            bet_outcome = 'WIN' if bowl.lower() in wl else 'LOSS'
            roi = round((ob - 1) * 100, 1) if bet_outcome == 'WIN' else -100.0
    pd.DataFrame([{
        'timestamp'          : datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'match_id'           : mid,
        'batting_team'       : bat,
        'bowling_team'       : bowl,
        'target'             : target,
        'score'              : score,
        'wickets'            : wkts,
        'over'               : over_str,
        'runs_needed'        : rr,
        'balls_remaining'    : br,
        'crr'                : round(crr, 2),
        'rrr'                : round(rrr, 2),
        'in_valid_window'    : in_window,
        'ai_prob_bat'        : round(pb*100, 1),
        'ai_prob_bowl'       : round(pw*100, 1),
        'market_odds_bat'    : oa,
        'market_odds_bowl'   : ob,
        'market_implied_bat' : round(imp_bat,  1) if imp_bat  else None,
        'market_implied_bowl': round(imp_bowl, 1) if imp_bowl else None,
        'edge_bat'           : round(edge_bat,  1) if edge_bat  else None,
        'edge_bowl'          : round(edge_bowl, 1) if edge_bowl else None,
        'ev_per_100_bat'     : round(ev_bat,  1) if ev_bat  else None,
        'ev_per_100_bowl'    : round(ev_bowl, 1) if ev_bowl else None,
        'value_bet_on'       : value_on,
        'actual_winner'      : winner,
        'bet_outcome'        : bet_outcome,
        'roi'                : roi
    }]).to_csv(LOG_FILE, mode='a', header=False, index=False)


# ══════════════════════════════════════════════════════════
# STEP 9 — Backtest stats (ROI focused)
# ══════════════════════════════════════════════════════════
def show_stats():
    if not os.path.exists(LOG_FILE):
        print("No log yet."); return
    log = pd.read_csv(LOG_FILE)
    if len(log) == 0:
        print("No entries yet."); return
    print(f"\n{'═'*52}")
    print(f"  BACKTEST REPORT")
    print(f"{'═'*52}")
    print(f"  Matches tracked    : {log['match_id'].nunique()}")
    print(f"  Total overs logged : {len(log)}")
    print(f"  Valid window overs : {log['in_valid_window'].sum()}")
    all_with_odds  = log[log['edge_bat'].notna()]
    value_bets     = log[log['value_bet_on'].notna()]
    completed_bets = log[log['bet_outcome'].notna()]
    print(f"\n  Overs with odds    : {len(all_with_odds)}")
    print(f"  Value bets flagged : {len(value_bets)}")
    print(f"  Completed bets     : {len(completed_bets)}")
    if len(completed_bets) > 0:
        wins      = (completed_bets['bet_outcome'] == 'WIN').sum()
        losses    = (completed_bets['bet_outcome'] == 'LOSS').sum()
        total_roi = completed_bets['roi'].sum()
        avg_roi   = completed_bets['roi'].mean()
        win_rate  = wins / len(completed_bets) * 100
        print(f"\n  ── BET RESULTS ──────────────────────")
        print(f"  W / L               : {wins}W  {losses}L")
        print(f"  Win rate            : {win_rate:.1f}%")
        print(f"  Total ROI (₹100/bet): ₹{total_roi:+.0f}")
        print(f"  Avg ROI per bet     : ₹{avg_roi:+.1f}")
        if total_roi > 0:
            print(f"  ✅ Profitable — confirm with 50+ bets")
        else:
            print(f"  ❌ Not yet profitable — keep tracking consistently")
    if len(all_with_odds) >= 5:
        print(f"\n  ── EDGE BY OVER RANGE ───────────────")
        for rng, label in [((1,5), 'Overs 1-5  (avoid)'),
                           ((5,15),'Overs 5-15 (valid)'),
                           ((15,20),'Overs 15-20 (avoid)')]:
            sub = all_with_odds[all_with_odds['over'].apply(
                lambda x: rng[0] <= int(str(x).split('.')[0]) < rng[1])]
            if len(sub) > 0:
                print(f"  {label:<25} avg edge {sub['edge_bat'].mean():+.1f}%  ({len(sub)} obs)")
    rated = log[log['actual_winner'].notna() & log['in_valid_window']].copy()
    if len(rated) >= 10:
        bins = [0, 30, 45, 55, 70, 100]
        rated['prob_bin'] = pd.cut(rated['ai_prob_bat'], bins=bins)
        cal = rated.groupby('prob_bin', observed=True).apply(
            lambda g: pd.Series({
                'n'     : len(g),
                'actual': (g['actual_winner'] == g['batting_team']).mean() * 100
            })
        ).reset_index()
        print(f"\n  ── LIVE CALIBRATION (valid window) ──")
        print(f"  {'AI Prob':>10}  {'Actual':>8}  {'n':>4}")
        for _, r in cal.iterrows():
            print(f"  {str(r['prob_bin']):>10}  {r['actual']:>7.0f}%  {int(r['n']):>4}")
    print(f"\n  Full data: {LOG_FILE}")
    print(f"{'═'*52}\n")


# ══════════════════════════════════════════════════════════
# STEP 10 — Main loop
# ══════════════════════════════════════════════════════════
init_log()
match_counter = 1

print("\n🏏  CRICKET AI v2 — OVER BY OVER TRACKER")
print(f"    Features  : {len(FEATS)} (player · venue · momentum · season-weighted)")
print(f"    Edge      : {EDGE_THRESHOLD}%  |  Valid window: overs {VALID_OVER_MIN}–{VALID_OVER_MAX}")
print("=" * 52)
print("Commands: 'stats' → ROI report | 'quit' → exit\n")

while True:
    try:
        print("━" * 52)
        cmd = input("New match? Press Enter  |  'stats'  |  'quit' : ").strip().lower()
        if cmd == 'quit': break
        if cmd == 'stats': show_stats(); continue

        mid = input(f"Match ID (Enter = M{match_counter}): ").strip() or f"M{match_counter}"
        match_counter += 1

        # ── Pre-match commentary ──────────────────────────
        print(f"\n  ── PRE-MATCH REPORT ─────────────────────────────")
        print(f"  Paste CLG commentary. Type END on new line when done.\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END': break
            lines.append(line)
        clg = "\n".join(lines)

        toss_winner = input("  Who won toss?                         : ").strip() or "unknown"
        toss_choice = input("  Bat or bowl?                          : ").strip() or "unknown"
        pitch_type  = input("  Pitch (batting/bowling/spin/balanced) : ").strip() or "balanced"

        print(f"\n  Enter after 1st innings ends:\n")
        bat   = input("  Batting team (chasing)   : ").strip()
        bowl  = input("  Bowling team (defending) : ").strip()
        t     = int(input("  Target                   : "))
        venue = input("  Venue/stadium (Enter=skip): ").strip() or None

        # Save match notes
        notes_file = 'match_notes.csv'
        if not os.path.exists(notes_file):
            pd.DataFrame(columns=[
                'match_id','batting_team','bowling_team','target',
                'venue','toss_winner','toss_choice','pitch_type','full_commentary'
            ]).to_csv(notes_file, index=False)
        pd.DataFrame([{
            'match_id'       : mid,
            'batting_team'   : bat,
            'bowling_team'   : bowl,
            'target'         : t,
            'venue'          : venue or '',
            'toss_winner'    : toss_winner,
            'toss_choice'    : toss_choice,
            'pitch_type'     : pitch_type,
            'full_commentary': clg[:1000]
        }]).to_csv(notes_file, mode='a', header=False, index=False)

        print(f"\n  ✅ Match notes saved for {mid}")
        print(f"  Tracking  : {bat} vs {bowl} | Target: {t}")
        if venue:
            wr = VENUE_STATS_MAP.get(venue, DEFAULT_VENUE_WR)
            print(f"  Venue     : {venue}  (historical chase win rate: {wr*100:.0f}%)")
        t_sr  = float(TEAM_CHASE_SR.get(bat,  DEFAULT_SR))  if bat  else DEFAULT_SR
        t_eco = float(TEAM_BOWL_ECO.get(bowl, DEFAULT_ECO)) if bowl else DEFAULT_ECO
        print(f"  {bat[:20]} chase SR: {t_sr:.0f}  |  {bowl[:20]} eco: {t_eco:.1f}")
        print(f"\n  Format: score/wickets over balls  e.g. 94/3 10 4")
        print(f"  Type 'done' to end match.\n")

        over_logs  = []
        winner     = None
        score_hist = []   # (ball_num, cum_score, cum_wkts) for momentum

        while True:
            raw = input(f"  Score (or 'done'): ").strip()
            if raw.lower() == 'done': break

            try:
                parts = raw.split()
                if len(parts) == 3 and '/' in parts[0]:
                    sw = parts[0].split('/')
                    s, w, o, xb = int(sw[0]), int(sw[1]), int(parts[1]), int(parts[2])
                else:
                    s  = int(input("  Score       : "))
                    w  = int(input("  Wickets     : "))
                    o  = int(input("  Full overs  : "))
                    xb = int(input("  Balls this  : "))

                ball_num = o * 6 + xb

                # Auto-compute momentum from score history (no user input needed)
                prev = next((e for e in reversed(score_hist) if e[0] <= ball_num - 18), None)
                if prev and ball_num > 18:
                    l18r = max(0, s - prev[1])
                    l18w = max(0, w - prev[2])
                else:
                    l18r = l18w = None  # predict() will estimate from CRR

                score_hist.append((ball_num, s, w))

                pb, pw, rr, br, wl, crr, rrr = predict(
                    t, s, w, o, xb,
                    venue=venue, bat_team=bat, bowl_team=bowl,
                    last_18_runs=l18r, last_18_wkts=l18w
                )
                over_str = f"{o}.{xb}"
                over_num = o

                # ── Auto-detect match end ──────────────────
                if s >= t:
                    winner = bat
                    over_logs.append((mid, bat, bowl, t, s, w, over_str, over_num,
                                      pb, pw, rr, br, crr, rrr, None, None))
                    print(f"\n  🏆 {bat} WON! Target reached ({s}/{w})")
                    break
                if w >= 10:
                    winner = bowl
                    over_logs.append((mid, bat, bowl, t, s, w, over_str, over_num,
                                      pb, pw, rr, br, crr, rrr, None, None))
                    print(f"\n  🏆 {bowl} WON! {bat} all out for {s}")
                    break
                if o >= 20 or (o == 19 and xb >= 6):
                    winner = bat if s >= t else bowl
                    margin = (f"by {s-t+1} runs" if winner == bat
                              else f"{bat} fell short by {t-s} runs")
                    over_logs.append((mid, bat, bowl, t, s, w, over_str, over_num,
                                      pb, pw, rr, br, crr, rrr, None, None))
                    print(f"\n  🏆 {winner} WON! ({margin})")
                    break

                # ── Warning if outside valid window ────────
                if over_num < VALID_OVER_MIN:
                    print(f"  ⚠️  Over {over_num} — early, predictions less reliable")
                elif over_num > VALID_OVER_MAX:
                    print(f"  ⚠️  Over {over_num} — late, market very efficient")

                print(f"\n  Over {over_str}  {s}/{w}  Need {rr} off {br}  "
                      f"CRR {crr:.2f}  RRR {rrr:.2f}")
                print(f"  {bat:<30} {pb*100:.1f}%")
                print(f"  {bowl:<30} {pw*100:.1f}%")

                if   pb >= 0.68: print(f"  🚀 {bat} on track")
                elif pb >= 0.48: print(f"  ⚖️  Balance — could go either way")
                elif pb >= 0.28: print(f"  ⚠️  {bowl} ahead")
                else:            print(f"  ❌ Very tough for {bat}")

                if l18r is not None:
                    print(f"  📊 Last 3 overs: {l18r} runs  {l18w} wickets")

                # ── Scenario Simulator ─────────────────────
                scenarios = simulate_next_over(
                    t, s, w, o, xb,
                    venue=venue, bat_team=bat, bowl_team=bowl,
                    l18r=l18r, l18w=l18w
                )
                print(f"\n  NEXT OVER SCENARIOS:")
                for label, pct, delta in scenarios:
                    arrow = "📈" if delta > 0 else "📉"
                    print(f"    {label:<28} → {bat[:18]}: {pct:.0f}%  ({delta:+.1f}%) {arrow}")

                # ── Odds ───────────────────────────────────
                print(f"\n  Odds (Enter to skip):")
                oa_s = input(f"  {bat[:22]} : ").strip()
                ob_s = input(f"  {bowl[:22]} : ").strip()
                oa   = float(oa_s) if oa_s else None
                ob   = float(ob_s) if ob_s else None

                # ── Edge display ───────────────────────────
                for side_prob, odds, name in [(pb, oa, bat), (pw, ob, bowl)]:
                    if not odds: continue
                    imp, edge, ev, is_val = check_edge(side_prob, odds, name, over_num)
                    marker = ('✅ VALUE BET' if is_val else
                              ('⚪ skip' if abs(edge) < EDGE_THRESHOLD else '❌ avoid'))
                    print(f"  {name[:25]}: AI {side_prob*100:.1f}% vs Mkt {imp:.1f}% "
                          f"→ Edge {edge:+.1f}%  {marker}")
                    if is_val:
                        print(f"    EV/₹100: ₹{ev:+.1f}  ← log this, verify after match")

                over_logs.append((mid, bat, bowl, t, s, w, over_str, over_num,
                                  pb, pw, rr, br, crr, rrr, oa, ob))
                print()

            except Exception as e:
                print(f"  Error: {e}\n")

        # ── Save match ─────────────────────────────────────
        if over_logs:
            for entry in over_logs:
                log_row(*entry, winner=winner)
            if winner:
                print(f"  ✅ {len(over_logs)} overs saved | Winner: {winner}\n")
            else:
                print(f"  ✅ {len(over_logs)} overs saved | No winner recorded\n")

    except Exception as e:
        print(f"Error: {e}\n")
