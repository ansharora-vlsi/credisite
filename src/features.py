"""
Point-in-time features. For every month-end snapshot we only use data that existed on that
date, so the backtest is honest (no future information leaks into a past score).

Each row = (developer, snapshot date). Columns = what a credit analyst would look at, plus the
90-day forward label used for training/evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AS_OF, BAD_LIMIT_SHARE, BAD_MIN_AMOUNT, LABEL_HORIZON_DAYS


def load(data_dir="data") -> dict[str, pd.DataFrame]:
    t = {
        "developers": pd.read_csv(f"{data_dir}/developers.csv", parse_dates=["onboarded"]),
        "invoices": pd.read_csv(f"{data_dir}/invoices.csv", parse_dates=["invoice_date", "due_date"]),
        "payments": pd.read_csv(f"{data_dir}/payments.csv", parse_dates=["date"]),
        "disputes": pd.read_csv(f"{data_dir}/disputes.csv", parse_dates=["raised", "resolved"]),
        "bounces": pd.read_csv(f"{data_dir}/bounces.csv", parse_dates=["date"]),
        "challans": pd.read_csv(f"{data_dir}/challans.csv", parse_dates=["date"]),
        "projects": pd.read_csv(f"{data_dir}/projects.csv", parse_dates=["start_date"]),
        "truth": pd.read_csv(f"{data_dir}/truth.csv", parse_dates=["event_start"]),
    }
    return t


def snapshot_dates() -> list[pd.Timestamp]:
    ends = list(pd.date_range("2025-01-31", pd.Timestamp(AS_OF) - pd.offsets.MonthEnd(1), freq="ME"))
    return ends + [pd.Timestamp(AS_OF)]


def invoice_state(t, a: pd.Timestamp) -> pd.DataFrame:
    """Outstanding and days-past-due of every invoice as of date a."""
    inv = t["invoices"][t["invoices"]["invoice_date"] <= a].copy()
    paid = t["payments"][t["payments"]["date"] <= a].groupby("invoice_id")["amount"].sum()
    d = t["disputes"]
    cn = d[d["resolved"].notna() & (d["resolved"] <= a)].groupby("invoice_id")["credit_note"].sum()
    inv["outstanding"] = (inv["amount"] - inv["invoice_id"].map(paid).fillna(0)
                          - inv["invoice_id"].map(cn).fillna(0)).clip(lower=0)
    inv.loc[inv["outstanding"] < 100, "outstanding"] = 0.0   # rounding dust
    inv["dpd"] = (a - inv["due_date"]).dt.days
    return inv


def _snapshot(t, a: pd.Timestamp) -> pd.DataFrame:
    dev = t["developers"].set_index("dev_id")
    inv = invoice_state(t, a)
    open_ = inv[inv["outstanding"] > 0]
    g = open_.groupby("dev_id")
    f = pd.DataFrame(index=dev.index)
    f["outstanding"] = g["outstanding"].sum()
    for lo, hi, col in [(0, 30, "od_1_30"), (30, 60, "od_31_60"), (60, 90, "od_61_90"), (90, 10**6, "od_90p")]:
        m = (open_["dpd"] > lo) & (open_["dpd"] <= hi)
        f[col] = open_[m].groupby("dev_id")["outstanding"].sum()
    f["max_dpd"] = g["dpd"].max().clip(lower=0)
    f = f.fillna(0.0)
    f["overdue"] = f[["od_1_30", "od_31_60", "od_61_90", "od_90p"]].sum(axis=1)

    i_all = t["invoices"]
    w90 = (i_all["invoice_date"] > a - pd.Timedelta(days=90)) & (i_all["invoice_date"] <= a)
    w180 = (i_all["invoice_date"] > a - pd.Timedelta(days=180)) & (i_all["invoice_date"] <= a - pd.Timedelta(days=90))
    f["billing_90d"] = i_all[w90].groupby("dev_id")["amount"].sum()
    f["billing_prev_90d"] = i_all[w180].groupby("dev_id")["amount"].sum()
    due90 = (i_all["due_date"] > a - pd.Timedelta(days=90)) & (i_all["due_date"] <= a)
    f["due_90d"] = i_all[due90].groupby("dev_id")["amount"].sum()
    f["disputed_90d"] = i_all[w90].groupby("dev_id")["disputed_amount"].sum()

    p = t["payments"].merge(i_all[["invoice_id", "due_date"]], on="invoice_id")
    p["late"] = (p["date"] - p["due_date"]).dt.days
    for lo, hi, sfx in [(0, 90, "90d"), (90, 180, "prev_90d")]:
        m = (p["date"] > a - pd.Timedelta(days=hi)) & (p["date"] <= a - pd.Timedelta(days=lo))
        pp = p[m]
        f[f"collected_{sfx}"] = pp.groupby("dev_id")["amount"].sum()
        wl = (pp["late"] * pp["amount"]).groupby(pp["dev_id"]).sum() / pp.groupby("dev_id")["amount"].sum()
        f[f"days_late_{sfx}"] = wl
        if sfx == "90d":
            part = pp[pp["kind"].isin(["partial", "balance"])].groupby("dev_id")["amount"].sum()
            f["partial_share_90d"] = part / pp.groupby("dev_id")["amount"].sum()

    b = t["bounces"]
    f["bounces_180d"] = b[(b["date"] > a - pd.Timedelta(days=180)) & (b["date"] <= a)].groupby("dev_id").size()
    d = t["disputes"]
    od = d[(d["raised"] <= a) & (d["resolved"].isna() | (d["resolved"] > a))]
    f["open_dispute_value"] = od.groupby("dev_id")["amount"].sum()

    fill0 = ["billing_90d", "billing_prev_90d", "due_90d", "disputed_90d", "collected_90d",
             "collected_prev_90d", "partial_share_90d", "bounces_180d", "open_dispute_value"]
    f[fill0] = f[fill0].fillna(0.0)

    f["credit_limit"] = dev["credit_limit"]
    f["utilisation"] = f["outstanding"] / f["credit_limit"]
    f["share_overdue_60p"] = np.where(f["outstanding"] > 0, (f["od_61_90"] + f["od_90p"]) / f["outstanding"].clip(lower=1), 0)
    daily = (f["billing_90d"] / 90).clip(lower=1)
    f["dso"] = (f["outstanding"] / daily).clip(upper=400)
    f.loc[f["outstanding"] == 0, "dso"] = 0
    f["collection_efficiency_90d"] = np.where(f["due_90d"] > 0, f["collected_90d"] / f["due_90d"].clip(lower=1), 1.0)
    f["collection_efficiency_90d"] = f["collection_efficiency_90d"].clip(0, 2)
    f["billing_change"] = np.where(f["billing_prev_90d"] > 0, f["billing_90d"] / f["billing_prev_90d"].clip(lower=1) - 1, 0).clip(-1, 3)
    f["dispute_rate_90d"] = np.where(f["billing_90d"] > 0, f["disputed_90d"] / f["billing_90d"].clip(lower=1), 0)
    f["open_dispute_share"] = np.where(f["outstanding"] > 0, f["open_dispute_value"] / f["outstanding"].clip(lower=1), 0).clip(0, 1)
    f["months_on_book"] = ((a - dev["onboarded"]).dt.days / 30.4).clip(lower=0)
    f["bad_threshold"] = np.maximum(BAD_MIN_AMOUNT, BAD_LIMIT_SHARE * f["credit_limit"])
    f["is_bad_now"] = f["od_90p"] >= f["bad_threshold"]
    f["active"] = (dev["onboarded"] <= a - pd.Timedelta(days=60)) & ((f["outstanding"] > 0) | (f["billing_90d"] > 0))
    f["date"] = a
    return f.reset_index()


def build_snapshots(t) -> pd.DataFrame:
    snaps = pd.concat([_snapshot(t, a) for a in snapshot_dates()], ignore_index=True)
    snaps = snaps.sort_values(["dev_id", "date"]).reset_index(drop=True)
    # Trends vs ~3 months earlier (same developer)
    g = snaps.groupby("dev_id")
    snaps["dso_3m_ago"] = g["dso"].shift(3)
    snaps["dso_trend"] = (snaps["dso"] - snaps["dso_3m_ago"]).fillna(0).clip(-150, 250)
    snaps["days_late_90d"] = snaps["days_late_90d"].fillna(snaps["days_late_prev_90d"])
    snaps["late_trend"] = (snaps["days_late_90d"] - snaps["days_late_prev_90d"]).fillna(0).clip(-60, 120)
    med_late = snaps.groupby("date")["days_late_90d"].transform("median")
    snaps["days_late_90d"] = snaps["days_late_90d"].fillna(med_late)

    # Forward label: bad at any month-end within the next 90 days
    me = snaps[snaps["date"].dt.is_month_end][["dev_id", "date", "is_bad_now"]]
    fut = me.rename(columns={"date": "fdate", "is_bad_now": "fbad"})
    m = snaps[["dev_id", "date"]].reset_index().merge(fut, on="dev_id")
    m = m[(m["fdate"] > m["date"]) & (m["fdate"] <= m["date"] + pd.Timedelta(days=LABEL_HORIZON_DAYS))]
    agg = m.groupby("index").agg(bad_next_90d=("fbad", "max"), n_future=("fbad", "size"))
    snaps = snaps.join(agg)
    snaps["label_complete"] = snaps["date"] + pd.Timedelta(days=LABEL_HORIZON_DAYS) <= pd.Timestamp(AS_OF)
    snaps.loc[~snaps["label_complete"], "bad_next_90d"] = np.nan
    return snaps
