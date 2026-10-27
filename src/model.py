"""
Explainable early-warning model + honest backtest.

Why logistic regression and not a black box: a credit team has to explain to a sales RM (and
sometimes to the customer) why a limit was cut. Each feature's contribution to the score is
just coefficient x standardised value, which turns directly into reason codes. We still train
a gradient-boosted model as a benchmark, to show what accuracy we give up for explainability.

Evaluation is out-of-time: train on snapshots up to TRAIN_END, test on later snapshots only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import (PD_TIERS, SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO, TEST_END, TRAIN_END)

# (column, transform, human label)
FEATURES = [
    ("utilisation",               lambda s: s.clip(0, 3),            "Credit limit utilisation"),
    ("share_overdue_60p",         lambda s: s.clip(0, 1),            "Share of dues overdue 60+ days"),
    ("dso",                       lambda s: (s / 100).clip(0, 4),    "Days sales outstanding"),
    ("dso_trend",                 lambda s: s / 100,                 "DSO change vs 3 months ago"),
    ("days_late_90d",             lambda s: (s / 30).clip(-1, 6),    "Avg days paid late (last 90d)"),
    ("late_trend",                lambda s: s / 30,                  "Change in days paid late"),
    ("collection_efficiency_90d", lambda s: s.clip(0, 1.5),          "Collection efficiency (90d)"),
    ("partial_share_90d",         lambda s: s.clip(0, 1),            "Share paid in partial tranches"),
    ("bounces_180d",              lambda s: s.clip(0, 4),            "Cheque/NACH bounces (180d)"),
    ("dispute_rate_90d",          lambda s: s.clip(0, 0.5),          "Challan dispute rate (90d)"),
    ("open_dispute_share",        lambda s: s.clip(0, 1),            "Open disputes as share of dues"),
    ("billing_change",            lambda s: s.clip(-1, 1.5),         "Order volume change (90d vs prior)"),
] 
# Relationship age is deliberately excluded: in this synthetic data most accounts were onboarded when
# the data starts, so "months on book" just tracks the calendar and would leak time into the model.

# Direction a credit analyst expects. Features whose fitted sign disagrees (usually collinearity)
# are dropped, because a reason code that says "more overdue lowers your risk" destroys trust.
EXPECTED_SIGN = {"utilisation": 1, "share_overdue_60p": 1, "dso": 1, "dso_trend": 1, "days_late_90d": 1,
                 "late_trend": 1, "collection_efficiency_90d": -1, "partial_share_90d": 1, "bounces_180d": 1,
                 "dispute_rate_90d": 1, "open_dispute_share": 1, "billing_change": -1}
FEATURE_NAMES = [f[0] for f in FEATURES]
LABELS = {f[0]: f[2] for f in FEATURES}


def design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({c: fn(df[c].astype(float)) for c, fn, _ in FEATURES}, index=df.index)
    return X.fillna(0.0)


class Scorecard:
    def __init__(self, C=0.3, features=None):
        self.C = C
        self.features = features        # None = select with sign constraints
        self.dropped = []

    def _fit(self, Z, y, cols):
        lr = LogisticRegression(C=self.C, max_iter=2000).fit(Z[cols], y)
        return lr, pd.Series(lr.coef_[0], index=cols)

    def fit(self, df):
        X = design(df)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        Z = (X - self.mu) / self.sd
        y = df["bad_next_90d"].astype(int)
        cols = list(self.features or FEATURE_NAMES)
        while True:
            lr, coef = self._fit(Z, y, cols)
            wrong = {c: abs(v) for c, v in coef.items() if np.sign(v) != EXPECTED_SIGN[c]}
            if not wrong:
                break
            worst = max(wrong, key=wrong.get)
            self.dropped.append(worst); cols.remove(worst)
        self.cols, self.lr, self.coef = cols, lr, coef
        return self

    def z(self, df):
        return ((design(df) - self.mu) / self.sd)[self.cols]

    def pd_(self, df):
        return self.lr.predict_proba(self.z(df))[:, 1]

    def contributions(self, df) -> pd.DataFrame:
        """Log-odds contribution of each feature relative to an average account."""
        return self.z(df) * self.coef


def to_score(pd_):
    pd_ = np.clip(pd_, 1e-4, 1 - 1e-4)
    odds = (1 - pd_) / pd_
    return np.clip(SCORE_BASE + SCORE_PDO / np.log(2) * np.log(odds / SCORE_BASE_ODDS), 300, 900).round()


def pd_tier(p):
    return np.where(p < PD_TIERS["Low"], "Low", np.where(p < PD_TIERS["Medium"], "Medium", "High"))


def eligible(s):
    """Accounts the early-warning system is for: active and not already seriously delinquent."""
    return s["active"] & ~s["is_bad_now"]


def backtest(snaps: pd.DataFrame) -> dict:
    lab = snaps[eligible(snaps) & snaps["label_complete"]].copy()
    train = lab[lab["date"] <= TRAIN_END]
    test = lab[(lab["date"] > TRAIN_END) & (lab["date"] <= TEST_END)]
    y = test["bad_next_90d"].astype(int)

    sc = Scorecard().fit(train)
    p_lr = sc.pd_(test)
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                         random_state=0).fit(design(train), train["bad_next_90d"].astype(int))
    p_gbm = gbm.predict_proba(design(test))[:, 1]

    # Baseline 1: the aging report — flag anyone with dues overdue 60+ days.
    base_aging = ((test["od_61_90"] + test["od_90p"]) > 0).astype(int)
    # Baseline 2: limit breach — flag anyone using more than 100% of their limit.
    base_util = (test["utilisation"] > 1.0).astype(int)
    flag_lr = (p_lr >= PD_TIERS["Medium"]).astype(int)

    def pr(flag):
        tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
        fn = int(((flag == 0) & (y == 1)).sum())
        return dict(alerts_per_month=round(flag.sum() / test["date"].nunique(), 1),
                    alert_rate=round(flag.mean(), 3),
                    precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3))

    # Same alert volume as the aging rule, so the comparison is fair
    k = int(base_aging.sum())
    thr_k = np.sort(p_lr)[::-1][k - 1] if k > 0 else 1
    flag_lr_matched = (p_lr >= thr_k).astype(int)

    comparison = [
        dict(method="Aging report rule (any dues 60+ days overdue)", **pr(base_aging), auc=None),
        dict(method="Limit breach rule (utilisation > 100%)", **pr(base_util), auc=None),
        dict(method="Scorecard, High tier (PD >= 15%)", **pr(flag_lr), auc=round(roc_auc_score(y, p_lr), 3)),
        dict(method="Scorecard, Medium + High tiers (PD >= 5%)", **pr((p_lr >= PD_TIERS["Low"]).astype(int)), auc=None),
        dict(method="Scorecard, same alert volume as aging rule", **pr(flag_lr_matched), auc=None),
        dict(method="Gradient boosting benchmark (black box)", **pr((p_gbm >= PD_TIERS["Medium"]).astype(int)),
             auc=round(roc_auc_score(y, p_gbm), 3)),
    ]

    # Calibration: predicted vs observed bad rate by PD band
    cal = pd.DataFrame({"p": p_lr, "y": y.values})
    cal["band"] = pd.cut(cal["p"], [0, 0.02, 0.05, 0.10, 0.15, 0.30, 1.0])
    calib = cal.groupby("band", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    calib = [dict(band=str(b), predicted=round(r.predicted, 3), observed=round(r.observed, 3), n=int(r.n))
             for b, r in calib.iterrows()]

    # Lead time: for accounts that went bad during the test period, how early did each method warn?
    all_elig = snaps[snaps["active"]].copy()
    all_elig["pd"] = sc.pd_(all_elig)
    first_bad = (snaps[snaps["is_bad_now"] & (snaps["date"] > TRAIN_END)]
                 .groupby("dev_id")["date"].min())
    prior_bad = snaps[snaps["is_bad_now"] & (snaps["date"] <= TRAIN_END)]["dev_id"].unique()
    first_bad = first_bad[~first_bad.index.isin(prior_bad)]
    leads = []
    for dev, ev in first_bad.items():
        h = all_elig[(all_elig["dev_id"] == dev) & (all_elig["date"] < ev) &
                     (all_elig["date"] >= ev - pd.Timedelta(days=183))]
        def lead(mask):
            d = h[mask]["date"]
            return int((ev - d.min()).days) if len(d) else 0
        leads.append(dict(dev_id=dev, event=str(ev.date()),
                          scorecard=lead(h["pd"] >= PD_TIERS["Medium"]),
                          aging=lead((h["od_61_90"] + h["od_90p"]) > 0)))
    leads_df = pd.DataFrame(leads)

    gbm_auc, lr_auc = roc_auc_score(y, p_gbm), roc_auc_score(y, p_lr)
    return dict(
        model=sc,
        metrics=dict(
            train_rows=int(len(train)), test_rows=int(len(test)),
            train_period=f"{train['date'].min().date()} to {train['date'].max().date()}",
            test_period=f"{test['date'].min().date()} to {test['date'].max().date()}",
            base_rate=round(float(y.mean()), 3),
            auc=round(lr_auc, 3), pr_auc=round(average_precision_score(y, p_lr), 3),
            gbm_auc=round(gbm_auc, 3),
            events=int(len(leads_df)),
            caught_scorecard=int((leads_df["scorecard"] > 0).sum()) if len(leads_df) else 0,
            caught_aging=int((leads_df["aging"] > 0).sum()) if len(leads_df) else 0,
            median_lead_scorecard=int(leads_df.loc[leads_df["scorecard"] > 0, "scorecard"].median()) if len(leads_df) else 0,
            median_lead_aging=int(leads_df.loc[leads_df["aging"] > 0, "aging"].median()) if len(leads_df) else 0,
        ),
        comparison=comparison, calibration=calib, leads=leads_df.to_dict("records"),
    )


def fit_final(snaps, features):
    """Refit on all labelled history for live scoring. Starts from the backtested feature set and
    applies the same sign check, since a refit on more data can flip a weak feature."""
    lab = snaps[eligible(snaps) & snaps["label_complete"]]
    return Scorecard(features=features).fit(lab)
