"""
End-to-end pipeline: generate data -> features -> backtest -> live scoring -> credit policy -> export.

Run:  python -m src.pipeline
Writes data/*.csv (raw tables), data/live_scores.csv and web/data.json (what the dashboard reads).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import generate_data
from .config import (ACTION_MATRIX, AS_OF, COST_OF_FUNDS, EXPOSURE_TIERS, LGD, PD_TIERS)
from .features import build_snapshots, invoice_state, load
from .model import LABELS, backtest, fit_final, pd_tier, to_score

warnings.filterwarnings("ignore")


def exposure_tier(x):
    return np.where(x < EXPOSURE_TIERS["Low"], "Low", np.where(x < EXPOSURE_TIERS["Medium"], "Medium", "High"))


def reason_text(f, v, med):
    """Turn a feature value into the sentence an analyst would write."""
    if f == "utilisation":
        return f"Using {v:.0%} of credit limit"
    if f == "dso":
        return (f"DSO above 400 days (portfolio median {med:.0f})" if v >= 400
                else f"DSO of {v:.0f} days (portfolio median {med:.0f})")
    if f == "dso_trend":
        return f"DSO {'up' if v > 0 else 'down'} {abs(v):.0f} days in 3 months"
    if f == "days_late_90d":
        return f"Paying {v:.0f} days past due on average (portfolio median {med:.0f})"
    if f == "collection_efficiency_90d":
        return f"Collected {v:.0%} of dues that fell due in the last 90 days"
    if f == "partial_share_90d":
        return f"{v:.0%} of recent receipts came as part-payments"
    if f == "bounces_180d":
        return f"{int(v)} cheque/NACH bounce{'s' if v != 1 else ''} in the last 180 days"
    if f == "dispute_rate_90d":
        return f"Disputes raised on {v:.1%} of billed value (last 90 days)"
    if f == "billing_change":
        return f"Orders {'down' if v < 0 else 'up'} {abs(v):.0%} vs the previous quarter"
    if f == "share_overdue_60p":
        return f"{v:.0%} of dues overdue 60+ days"
    return f"{LABELS.get(f, f)}: {v:.2f}"


def recommended_limit(row):
    monthly = (row["billing_90d"] + row["billing_prev_90d"]) / 6
    if row["tier"] == "Delinquent":
        return 0.0
    mult = {"Low": 1.15, "Medium": 0.90, "High": 0.50}[row["tier"]]
    lim = monthly * (row["credit_terms_days"] / 30 + 0.5) * mult
    return float(max(5e5, round(lim / 5e5) * 5e5)) if monthly > 0 else float(row["credit_limit"])


def r0(x):
    return None if pd.isna(x) else float(round(x, 0))


def main():
    print("1/5 generating synthetic data ...")
    generate_data.main("data")
    t = load("data")
    print("2/5 building point-in-time snapshots ...")
    snaps = build_snapshots(t)
    print("3/5 backtesting ...")
    bt = backtest(snaps)
    feats = bt["model"].cols
    final = fit_final(snaps, feats)
    print("   ", json.dumps(bt["metrics"]))

    print("4/5 scoring every snapshot with the live model ...")
    snaps["pd"] = final.pd_(snaps)
    snaps["score"] = to_score(snaps["pd"])
    snaps["tier"] = np.where(snaps["is_bad_now"], "Delinquent", pd_tier(snaps["pd"]))
    contrib = final.contributions(snaps)

    as_of = pd.Timestamp(AS_OF)
    dev = t["developers"].set_index("dev_id")
    live = snaps[snaps["date"] == as_of].set_index("dev_id").join(
        dev[["name", "segment", "city", "region", "rm", "credit_terms_days", "trade_credit_insured", "onboarded"]])
    live_idx = snaps.index[snaps["date"] == as_of]
    live_c = contrib.loc[live_idx]
    live_c.index = snaps.loc[live_idx, "dev_id"].values
    live = live[live["active"] | (live["outstanding"] > 0)].copy()
    live_c = live_c.loc[live.index]

    live["exposure_tier"] = exposure_tier(live["outstanding"])
    live["action"] = [("Collections / legal track" if tr == "Delinquent" else ACTION_MATRIX[(tr, ex)])
                      for tr, ex in zip(live["tier"], live["exposure_tier"])]
    live["rec_limit"] = live.apply(recommended_limit, axis=1)
    # No new credit where the action is advance-only or a supply pause
    no_credit = live["action"].isin(["Advance payment only", "Pause new supply", "Pause supply + escalate"])
    live.loc[no_credit, "rec_limit"] = 0.0
    live["expected_loss"] = np.where(live["tier"] == "Delinquent", 0.6, live["pd"]) * live["outstanding"] * LGD
    live["overdue_31p"] = live["od_31_60"] + live["od_61_90"] + live["od_90p"]
    live["priority"] = live["overdue_31p"] * (0.3 + np.where(live["tier"] == "Delinquent", 1.0, live["pd"]))
    live["monthly_billing"] = live["billing_90d"] / 3
    live.reset_index().drop(columns=["index"], errors="ignore").to_csv("data/live_scores.csv", index=False)

    med = live.median(numeric_only=True)
    inv_now = invoice_state(t, as_of)
    pays = t["payments"]; disp = t["disputes"]; bnc = t["bounces"]; proj = t["projects"]
    hist = snaps.sort_values("date")

    devs_out = []
    for dev_id, r in live.iterrows():
        c = live_c.loc[dev_id]
        reasons = []
        for f, v in c.sort_values(ascending=False).items():
            if v > 0.15 and len(reasons) < 4:
                reasons.append(dict(feature=f, impact=round(float(v), 2),
                                    text=reason_text(f, r[f], med.get(f, np.nan))))
        mitigants = []
        for f, v in c.sort_values().items():
            if v < -0.15 and len(mitigants) < 3:
                mitigants.append(dict(feature=f, impact=round(float(v), 2),
                                      text=reason_text(f, r[f], med.get(f, np.nan))))
        h = hist[hist["dev_id"] == dev_id]
        iv = inv_now[inv_now["dev_id"] == dev_id].sort_values("invoice_date", ascending=False).head(10)
        dp = disp[(disp["dev_id"] == dev_id) & disp["resolved"].isna()].sort_values("raised", ascending=False).head(6)
        bb = bnc[(bnc["dev_id"] == dev_id) & (bnc["date"] > as_of - pd.Timedelta(days=180))]
        pr = proj[proj["dev_id"] == dev_id]
        monthly = (t["invoices"][t["invoices"]["dev_id"] == dev_id]
                   .assign(m=lambda d: d["invoice_date"].dt.to_period("M").astype(str))
                   .groupby("m")["amount"].sum())
        coll = (pays[pays["dev_id"] == dev_id].assign(m=lambda d: d["date"].dt.to_period("M").astype(str))
                .groupby("m")["amount"].sum())
        months = [str(p) for p in pd.period_range("2024-10", as_of.to_period("M"), freq="M")]
        devs_out.append(dict(
            id=dev_id, name=r["name"], segment=r["segment"], city=r["city"], rm=r["rm"],
            terms=int(r["credit_terms_days"]), limit=r0(r["credit_limit"]), insured=bool(r["trade_credit_insured"]),
            onboarded=str(pd.Timestamp(r["onboarded"]).date()),
            pd=round(float(r["pd"]), 4), score=int(r["score"]), tier=r["tier"], exposure_tier=r["exposure_tier"],
            action=r["action"], rec_limit=r0(r["rec_limit"]), expected_loss=r0(r["expected_loss"]),
            priority=r0(r["priority"]),
            outstanding=r0(r["outstanding"]), overdue=r0(r["overdue"]),
            od=[r0(r["od_1_30"]), r0(r["od_31_60"]), r0(r["od_61_90"]), r0(r["od_90p"])],
            not_due=r0(r["outstanding"] - r["overdue"]), max_dpd=int(r["max_dpd"]),
            monthly_billing=r0(r["monthly_billing"]), billing_90d=r0(r["billing_90d"]),
            dso=round(float(r["dso"]), 0), dso_trend=round(float(r["dso_trend"]), 0),
            utilisation=round(float(r["utilisation"]), 3),
            coll_eff=round(float(r["collection_efficiency_90d"]), 3),
            days_late=round(float(r["days_late_90d"]), 0), billing_change=round(float(r["billing_change"]), 3),
            partial_share=round(float(r["partial_share_90d"]), 3), bounces=int(r["bounces_180d"]),
            dispute_rate=round(float(r["dispute_rate_90d"]), 4), open_disputes=r0(r["open_dispute_value"]),
            reasons=reasons, mitigants=mitigants,
            hist=dict(d=[str(x.date()) for x in h["date"]], score=[int(x) for x in h["score"]],
                      pd=[round(float(x), 4) for x in h["pd"]], out=[r0(x) for x in h["outstanding"]],
                      od60=[r0(a + b) for a, b in zip(h["od_61_90"], h["od_90p"])],
                      dso=[round(float(x), 0) for x in h["dso"]]),
            months=months, billed=[r0(monthly.get(m, 0)) for m in months],
            collected=[r0(coll.get(m, 0)) for m in months],
            invoices=[dict(id=x.invoice_id, date=str(x.invoice_date.date()), due=str(x.due_date.date()),
                           amount=r0(x.amount), outstanding=r0(x.outstanding), dpd=int(x.dpd),
                           status="Paid" if x.outstanding == 0 else ("Part-paid" if x.outstanding < x.amount else "Open"))
                      for x in iv.itertuples()],
            disputes=[dict(challan=x.challan_id, raised=str(x.raised.date()), reason=x.reason, amount=r0(x.amount))
                      for x in dp.itertuples()],
            bounce_list=[dict(date=str(x.date.date()), amount=r0(x.amount)) for x in bb.itertuples()],
            projects=len(pr),
        ))

    # Portfolio history
    act = snaps[snaps["active"] | (snaps["outstanding"] > 0)]
    port = []
    for d, g in act.groupby("date"):
        out = g["outstanding"].sum(); b90 = g["billing_90d"].sum()
        port.append(dict(d=str(d.date()), out=r0(out), od=[r0(g[c].sum()) for c in ["od_1_30", "od_31_60", "od_61_90", "od_90p"]],
                         dso=round(out / max(b90 / 90, 1), 1), billing_90d=r0(b90),
                         collected_90d=r0(g["collected_90d"].sum()),
                         high=int((g["tier"] == "High").sum()), delinquent=int(g["is_bad_now"].sum()),
                         el=r0((np.where(g["is_bad_now"], 0.6, g["pd"]) * g["outstanding"] * LGD).sum())))

    model_card = dict(
        metrics=bt["metrics"], comparison=bt["comparison"], calibration=bt["calibration"], leads=bt["leads"],
        features=[dict(feature=f, label=LABELS[f], coef=round(float(final.coef[f]), 3)) for f in final.cols],
        dropped=[dict(feature=f, label=LABELS[f]) for f in bt["model"].dropped + final.dropped],
        params=dict(intercept=float(final.lr.intercept_[0]),
                    mu={f: float(final.mu[f]) for f in final.cols},
                    sd={f: float(final.sd[f]) for f in final.cols},
                    coef={f: float(final.coef[f]) for f in final.cols}),
    )
    payload = dict(
        as_of=AS_OF, generated_note="Synthetic data. Fictional developers. Not ArisInfra data.",
        config=dict(pd_tiers=PD_TIERS, exposure_tiers=EXPOSURE_TIERS, lgd=LGD, cost_of_funds=COST_OF_FUNDS,
                    action_matrix={f"{k[0]}|{k[1]}": v for k, v in ACTION_MATRIX.items()}),
        developers=devs_out, portfolio=port, model=model_card,
        counts=dict(challans=int(len(t["challans"])), invoices=int(len(t["invoices"])),
                    payments=int(len(t["payments"])), disputes=int(len(t["disputes"])),
                    bounces=int(len(t["bounces"])), projects=int(len(t["projects"]))),
    )
    with open("web/data.json", "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"5/5 wrote web/data.json ({len(devs_out)} accounts)")


if __name__ == "__main__":
    main()
