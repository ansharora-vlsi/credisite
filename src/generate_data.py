"""
Synthetic construction-materials receivables generator.

Simulates a B2B platform supplying developers and EPC contractors across Indian cities,
week by week. Every table looks like something a real ops/finance system would hold:

  developers   master data: segment, city, RM, credit terms, credit limit
  projects     each developer's sites and their construction stage over time
  challans     truck-level deliveries (challan qty vs weighbridge qty, rate, value)
  invoices     weekly invoice per project, with due date from credit terms
  payments     receipts against invoices (full, partial tranches, dispute releases)
  disputes     challan disputes and how they were resolved
  bounces      cheque / NACH bounces
  truth        hidden ground truth (stress events) used ONLY to sanity-check, never by the model

Behaviour built in:
  * Construction-stage material mix (excavation -> foundation -> superstructure -> finishing)
  * Seasonality: monsoon slowdown Jun-Sep, festive dip Oct-Nov, March year-end push
  * Everyone pays late; how late depends on segment. Stress makes it worse.
  * Stress events (project slowdown / funding gap) ramp up over weeks. Some developers
    recover, some default, a few are sudden shocks with little warning.
  * Stressed developers raise more challan disputes and bounce more cheques.
  * Credit hold: supply is paused when 90+ day overdues cross 15% of the limit.
  * EPC contractors pay in lumps when their own running-account bills are cleared.
  * Credit limits are set once at onboarding and never revisited (the problem this tool solves).

Run:  python -m src.generate_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AS_OF, N_DEVELOPERS, SEED, START

RNG = np.random.default_rng(SEED)

WEEKS = pd.date_range(START, AS_OF, freq="W-MON")
W = len(WEEKS)

SEGMENTS = {
    # monthly billing at full activity (Rs), credit terms (days), typical days paid late (mean, sd),
    # baseline stress, chance of a stress event in the 2-year window, partial-payment base rate
    "Large developer":    dict(share=0.15, bill=(80e5, 220e5), terms=[45, 60],     late=(16, 9),  stress=0.04, event_p=0.10, partial=0.04),
    "Mid-size developer": dict(share=0.42, bill=(15e5, 60e5),  terms=[45, 60],     late=(36, 14), stress=0.10, event_p=0.24, partial=0.08),
    "Small developer":    dict(share=0.28, bill=(3e5, 15e5),   terms=[30, 45],     late=(46, 18), stress=0.16, event_p=0.32, partial=0.12),
    "EPC contractor":     dict(share=0.15, bill=(50e5, 150e5), terms=[60, 90],     late=(40, 16), stress=0.09, event_p=0.20, partial=0.06),
}

CITIES = {  # city: (weight, price multiplier, region)
    "Mumbai": (0.18, 1.08, "MMR"), "Thane": (0.13, 1.05, "MMR"), "Navi Mumbai": (0.10, 1.04, "MMR"),
    "Pune": (0.20, 1.00, "Pune"), "Bengaluru": (0.17, 1.02, "South"), "Hyderabad": (0.12, 0.97, "South"),
    "Chennai": (0.10, 0.99, "South"),
}
RMS = {"MMR": ["Rohit Menon", "Priya Kulkarni"], "Pune": ["Vikram Desai", "Neha Bansal"],
       "South": ["Arjun Shetty", "Sneha Iyer"]}

MATERIALS = {  # base rate (Rs per unit, ex-site), unit, truckload range, weighed at weighbridge?
    "RMC M25":          dict(rate=5600,  unit="m3",  load=(6, 7),     weighed=False),
    "TMT steel Fe550D": dict(rate=56000, unit="t",   load=(15, 25),   weighed=True),
    "Cement OPC 53":    dict(rate=360,   unit="bag", load=(400, 600), weighed=False),
    "Aggregates 20mm":  dict(rate=1250,  unit="t",   load=(20, 30),   weighed=True),
    "M-sand":           dict(rate=1100,  unit="t",   load=(20, 30),   weighed=True),
    "AAC blocks":       dict(rate=4300,  unit="m3",  load=(25, 35),   weighed=False),
}
MAT_NAMES = list(MATERIALS)

STAGES = ["Excavation", "Foundation", "Superstructure", "Finishing"]
STAGE_SPLIT = [0.08, 0.15, 0.50, 0.27]           # share of project duration
STAGE_ACTIVITY = [0.40, 0.90, 1.00, 0.50]        # material intensity
STAGE_MIX = {                                     # share of material value by stage
    "Excavation":     [0.20, 0.10, 0.00, 0.50, 0.20, 0.00],
    "Foundation":     [0.45, 0.40, 0.05, 0.10, 0.00, 0.00],
    "Superstructure": [0.40, 0.35, 0.08, 0.00, 0.05, 0.12],
    "Finishing":      [0.00, 0.00, 0.35, 0.15, 0.25, 0.25],
}
EPC_MIX = [0.35, 0.35, 0.10, 0.20, 0.00, 0.00]

NAME_A = ["Aakar", "Nirman", "Sankalp", "Vastu", "Prithvi", "Akshar", "Riddhi", "Mangal", "Navkar",
          "Trimurti", "Utsav", "Swaraj", "Orchid", "Crescent", "Meridian", "Evergreen", "Silverline",
          "Bluestone", "Greenfield", "Terracotta", "Granite", "Keystone", "Sahyadri", "Kaveri",
          "Narmada", "Deccan", "Konkan", "Tulsi", "Amrut", "Shubharambh", "Vishwas", "Sarthak",
          "Aarambh", "Saptagiri", "Nilgiri", "Cauvery", "Indus", "Godavari", "Malabar", "Coromandel"]
NAME_B = {"Large developer": ["Realty", "Estates", "Lifespaces"],
          "Mid-size developer": ["Developers", "Realty", "Housing", "Builders"],
          "Small developer": ["Builders", "Constructions", "Homes", "Developers"],
          "EPC contractor": ["Infraprojects", "Infra Engineers", "Constructions (EPC)"]}


def _seasonality(d: pd.Timestamp) -> float:
    m = d.month
    if m in (6, 7, 8, 9):
        return 0.70           # monsoon
    if m in (10, 11):
        return 0.88           # festive season dip
    if m == 3:
        return 1.15           # financial year-end push
    return 1.0


SEASON = np.array([_seasonality(d) for d in WEEKS])

# Commodity price index: steel moves the most, aggregates barely.
PRICE_VOL = {"RMC M25": 0.006, "TMT steel Fe550D": 0.015, "Cement OPC 53": 0.008,
             "Aggregates 20mm": 0.004, "M-sand": 0.004, "AAC blocks": 0.005}
PRICE_IDX = {m: np.exp(np.cumsum(RNG.normal(0, v, W))) for m, v in PRICE_VOL.items()}


def _stress_path(seg: dict) -> tuple[np.ndarray, dict]:
    """Weekly latent financial stress in [0,1]."""
    base = float(np.clip(RNG.normal(seg["stress"], 0.04), 0.0, 0.35))
    noise = np.zeros(W)
    for t in range(1, W):
        noise[t] = 0.85 * noise[t - 1] + RNG.normal(0, 0.025)
    s = base + noise
    truth = dict(event="none", event_start=None, peak=None)

    r = RNG.random()
    if r < seg["event_p"]:
        start = int(RNG.integers(8, W - 6))
        ramp = int(RNG.integers(8, 23))
        peak = float(RNG.uniform(0.65, 1.0))
        recovers = RNG.random() < 0.45
        plateau = int(RNG.integers(4, 13))
        decay = int(RNG.integers(10, 21))
        for t in range(start, W):
            k = t - start
            if k < ramp:
                lvl = base + (peak - base) * (k / ramp) ** 1.3
            elif not recovers or k < ramp + plateau:
                lvl = peak
            else:
                lvl = max(base, peak - (peak - base) * (k - ramp - plateau) / decay)
            s[t] = lvl + noise[t]
        truth = dict(event="recovered" if recovers else "defaulted",
                     event_start=WEEKS[start], peak=round(peak, 2))
    elif r < seg["event_p"] + 0.035:
        # Sudden shock: e.g. project stopped by authority, promoter dispute. Little warning.
        start = int(RNG.integers(8, W - 4))
        peak = float(RNG.uniform(0.85, 1.0))
        s[start:] = peak + noise[start:]
        truth = dict(event="sudden shock", event_start=WEEKS[start], peak=round(peak, 2))
    return np.clip(s, 0, 1), truth


def _projects(dev_id: str, city: str, seg_name: str) -> list[dict]:
    out, start_m = [], RNG.uniform(-30, -2)
    i = 1
    while start_m < len(WEEKS) / 4.33:
        dur = RNG.uniform(24, 42) if seg_name != "EPC contractor" else RNG.uniform(18, 36)
        out.append(dict(project_id=f"{dev_id}-P{i}", dev_id=dev_id, city=city,
                        start=pd.Timestamp(START) + pd.Timedelta(days=30.4 * start_m),
                        months=dur, size=RNG.lognormal(0, 0.35)))
        start_m += RNG.uniform(7, 18) * (0.7 if seg_name == "Large developer" else 1.0)
        i += 1
    return out


def _stage_at(p: dict, d: pd.Timestamp):
    frac = (d - p["start"]).days / (30.4 * p["months"])
    if frac < 0 or frac >= 1:
        return None
    edges = np.cumsum(STAGE_SPLIT)
    return int(np.searchsorted(edges, frac, side="right"))


def generate():
    devs, projects, challans, invoices, payments, disputes, bounces, truth_rows = ([] for _ in range(8))
    seg_names = list(SEGMENTS)
    seg_p = [SEGMENTS[s]["share"] for s in seg_names]
    city_names = list(CITIES)
    city_p = np.array([CITIES[c][0] for c in city_names]); city_p /= city_p.sum()
    used_names: set[str] = set()
    inv_counter = 0
    ch_counter = 0

    for n in range(N_DEVELOPERS):
        dev_id = f"DV{1001 + n}"
        seg_name = RNG.choice(seg_names, p=seg_p)
        seg = SEGMENTS[seg_name]
        city = RNG.choice(city_names, p=city_p)
        region = CITIES[city][2]
        rm = RNG.choice(RMS[region])
        while True:
            name = f"{RNG.choice(NAME_A)} {RNG.choice(NAME_B[seg_name])}"
            if name not in used_names:
                used_names.add(name); break
        terms = int(RNG.choice(seg["terms"]))
        bill = float(np.exp(RNG.uniform(np.log(seg["bill"][0]), np.log(seg["bill"][1]))))
        # limit set once at onboarding, sized to cover the credit cycle, with sales-driven optimism
        limit = round(bill * (terms / 30 + 0.5) * RNG.uniform(0.8, 1.4) / 5e5) * 5e5
        onboard_w = 0 if RNG.random() < 0.72 else int(RNG.integers(4, W - 30))
        late_mu, late_sd = seg["late"]
        late_mu *= RNG.uniform(0.7, 1.3)          # each developer has its own habits
        insured = bool(RNG.random() < (0.55 if seg_name in ("Large developer", "EPC contractor") else 0.25))

        stress, truth = _stress_path(seg)
        truth_rows.append(dict(dev_id=dev_id, **truth))
        projs = _projects(dev_id, city, seg_name)
        for p in projs:
            projects.append(dict(project_id=p["project_id"], dev_id=dev_id, city=city,
                                 start_date=p["start"].date(), planned_months=round(p["months"], 1)))
        epc_cycle = int(RNG.integers(6, 11))       # weeks between RA-bill realisations
        epc_offset = int(RNG.integers(0, epc_cycle))

        # running state for credit-hold logic
        inv_due = []; inv_amt = []; inv_paid_sched = []   # paid_sched: list of (date, amount)

        def overdue_90_at(d):
            tot = 0.0
            for due, amt, sched in zip(inv_due, inv_amt, inv_paid_sched):
                if (d - due).days > 90:
                    paid = sum(a for pd_, a in sched if pd_ is not None and pd_ <= d)
                    tot += max(0.0, amt - paid)
            return tot

        devs.append(dict(dev_id=dev_id, name=name, segment=seg_name, city=city, region=region, rm=rm,
                         credit_terms_days=terms, credit_limit=limit, onboarded=WEEKS[onboard_w].date(),
                         trade_credit_insured=insured, expected_monthly_billing=round(bill, -3)))

        hold_until = -1
        for t in range(onboard_w, W):
            d = WEEKS[t]
            s = stress[t]
            # Credit hold check every 4 weeks (ops reviews aging report monthly)
            if t % 4 == 0 and t > onboard_w + 8:
                if overdue_90_at(d) > 0.15 * limit:
                    hold_until = t + 4
            on_hold = t < hold_until
            if on_hold:
                continue

            active = []
            for p in projs:
                st = _stage_at(p, d)
                if st is not None:
                    active.append((p, st))
            if not active:
                continue
            weekly_total = bill / 4.33
            norm = sum(p["size"] * STAGE_ACTIVITY[st] for p, st in active)
            norm = max(norm, 0.6)                 # developers don't scale billing linearly with sites
            for p, st in active:
                share = p["size"] * STAGE_ACTIVITY[st] / norm
                value = weekly_total * share * SEASON[t] * (1 - 0.75 * s) * RNG.lognormal(0, 0.30)
                if value < 2e4:
                    continue
                mix = EPC_MIX if seg_name == "EPC contractor" else STAGE_MIX[STAGES[st]]
                inv_lines = []
                for m_idx, m_share in enumerate(mix):
                    if m_share == 0:
                        continue
                    mname = MAT_NAMES[m_idx]; mat = MATERIALS[mname]
                    rate = mat["rate"] * CITIES[city][1] * PRICE_IDX[mname][t] * RNG.uniform(0.98, 1.02)
                    target = value * m_share * RNG.uniform(0.8, 1.2)
                    avg_load = np.mean(mat["load"])
                    n_trucks = int(RNG.poisson(target / (rate * avg_load)))
                    for _ in range(n_trucks):
                        ch_counter += 1
                        qty = round(float(RNG.uniform(*mat["load"])), 2 if mat["unit"] != "bag" else 0)
                        cday = d + pd.Timedelta(days=int(RNG.integers(0, 6)))
                        wb = round(qty * (1 + RNG.normal(0, 0.006)), 2) if mat["weighed"] else None
                        dispute_p = 0.012 + 0.07 * s
                        is_dispute = RNG.random() < dispute_p
                        reason = None
                        if is_dispute:
                            if mat["weighed"]:
                                wb = round(qty * (1 - RNG.uniform(0.025, 0.08)), 2)
                                reason = "Weighbridge shortfall"
                            elif mname == "RMC M25":
                                reason = RNG.choice(["Cube test failure", "Late arrival / load rejected"])
                            elif mname == "Cement OPC 53":
                                reason = "Short count"
                            else:
                                reason = RNG.choice(["Breakage / quality", "Rate mismatch vs PO"])
                        amt = round(qty * rate, 0)
                        cid = f"CH{ch_counter:07d}"
                        challans.append(dict(challan_id=cid, dev_id=dev_id, project_id=p["project_id"],
                                             date=cday.date(), material=mname, unit=mat["unit"],
                                             qty_challan=qty, qty_weighbridge=wb, rate=round(rate, 0),
                                             value=amt, disputed=is_dispute, dispute_reason=reason))
                        inv_lines.append((cid, amt, is_dispute, reason))
                if not inv_lines:
                    continue
                inv_counter += 1
                inv_id = f"INV{inv_counter:06d}"
                inv_date = d + pd.Timedelta(days=6)
                due = inv_date + pd.Timedelta(days=terms)
                amount = float(sum(a for _, a, _, _ in inv_lines))
                disputed_amt = float(sum(a for _, a, dsp, _ in inv_lines if dsp))
                invoices.append(dict(invoice_id=inv_id, dev_id=dev_id, project_id=p["project_id"],
                                     invoice_date=inv_date.date(), due_date=due.date(), amount=amount,
                                     disputed_amount=disputed_amt, n_challans=len(inv_lines)))

                sched = _payment_schedule(inv_id, dev_id, due, amount, disputed_amt, stress, s,
                                          late_mu, late_sd, seg, seg_name, epc_cycle, epc_offset,
                                          payments, disputes, bounces, inv_lines)
                inv_due.append(due); inv_amt.append(amount); inv_paid_sched.append(sched)

    as_of = pd.Timestamp(AS_OF)
    t = {
        "developers": pd.DataFrame(devs),
        "projects": pd.DataFrame(projects),
        "challans": pd.DataFrame(challans),
        "invoices": pd.DataFrame(invoices),
        "payments": pd.DataFrame(payments),
        "disputes": pd.DataFrame(disputes),
        "bounces": pd.DataFrame(bounces),
        "truth": pd.DataFrame(truth_rows),
    }
    # Drop anything that "happens" after today: the system only knows the past.
    for name, col in [("payments", "date"), ("bounces", "date"), ("challans", "date")]:
        t[name]["date"] = pd.to_datetime(t[name]["date"])
        t[name] = t[name][t[name]["date"] <= as_of].reset_index(drop=True)
    t["invoices"]["invoice_date"] = pd.to_datetime(t["invoices"]["invoice_date"])
    t["invoices"]["due_date"] = pd.to_datetime(t["invoices"]["due_date"])
    t["invoices"] = t["invoices"][t["invoices"]["invoice_date"] <= as_of].reset_index(drop=True)
    d = t["disputes"]
    d["raised"] = pd.to_datetime(d["raised"]); d["resolved"] = pd.to_datetime(d["resolved"])
    d = d[d["raised"] <= as_of].copy()
    d.loc[d["resolved"] > as_of, ["resolved", "resolution", "credit_note"]] = [pd.NaT, "Open", 0.0]
    t["disputes"] = d.reset_index(drop=True)
    return t


def _stress_at(stress, d):
    idx = min(max(int((d - WEEKS[0]).days // 7), 0), W - 1)
    return stress[idx], idx


def _push_past_default(stress, d, limit_date):
    """A developer in default (stress > 0.8) doesn't pay; payment waits until stress eases."""
    while d <= limit_date:
        s, _ = _stress_at(stress, d)
        if s <= 0.80:
            return d
        d = d + pd.Timedelta(days=14)
    return None   # never paid within the simulated horizon


def _payment_schedule(inv_id, dev_id, due, amount, disputed_amt, stress, s_now, late_mu, late_sd,
                      seg, seg_name, epc_cycle, epc_offset, payments, disputes, bounces, inv_lines):
    horizon = pd.Timestamp(AS_OF) + pd.Timedelta(days=200)   # simulate beyond today, then truncate
    s_due, _ = _stress_at(stress, due)
    late = max(-5.0, RNG.normal(late_mu + 95 * s_due ** 1.6, late_sd * (1 + s_due)))
    intended = due + pd.Timedelta(days=int(late))
    if seg_name == "EPC contractor":
        # payments land when the contractor's own RA bill is cleared by the client
        wk = int((intended - WEEKS[0]).days // 7)
        wk = wk + ((epc_offset - wk) % epc_cycle)
        intended = WEEKS[0] + pd.Timedelta(weeks=wk, days=int(RNG.integers(0, 4)))
    sched = []
    payable = amount - disputed_amt

    def pay(date, amt, kind):
        if date is None or amt <= 0:
            sched.append((None, amt)); return
        s_pay, _ = _stress_at(stress, date)
        if RNG.random() < 0.004 + 0.09 * s_pay ** 2:      # cheque / NACH bounce, re-presented later
            bounces.append(dict(dev_id=dev_id, invoice_id=inv_id, date=date.date(), amount=round(amt, 0)))
            date = _push_past_default(stress, date + pd.Timedelta(days=int(RNG.integers(10, 30))), horizon)
            if date is None:
                sched.append((None, amt)); return
        payments.append(dict(invoice_id=inv_id, dev_id=dev_id, date=date.date(), amount=round(amt, 0), kind=kind))
        sched.append((date, amt))

    first = _push_past_default(stress, intended, horizon)
    if first is not None and RNG.random() < seg["partial"] + 0.45 * s_due:
        frac = RNG.uniform(0.4, 0.8)
        pay(first, payable * frac, "partial")
        second = _push_past_default(stress, first + pd.Timedelta(days=int(RNG.integers(15, 60))), horizon)
        pay(second, payable * (1 - frac), "balance")
    else:
        pay(first, payable, "full")

    # Disputed challans: withheld until resolved; some end in a credit note.
    for cid, amt, dsp, reason in inv_lines:
        if not dsp:
            continue
        raised = due - pd.Timedelta(days=int(RNG.integers(5, 25)))
        resolved = raised + pd.Timedelta(days=int(RNG.integers(10, 50)))
        cn_share = RNG.uniform(0.03, 0.10) if RNG.random() < 0.35 else 0.0
        disputes.append(dict(challan_id=cid, invoice_id=inv_id, dev_id=dev_id, raised=raised.date(),
                             reason=reason, amount=amt, resolved=resolved.date(),
                             resolution="Credit note" if cn_share else "Upheld (paid)",
                             credit_note=round(amt * cn_share, 0)))
        release = _push_past_default(stress, max(resolved, first or resolved), horizon)
        pay(release, amt * (1 - cn_share), "dispute release")
        if cn_share:
            sched.append((resolved, amt * cn_share))   # credit note reduces receivable
    return sched


def main(out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    tables = generate()
    for name, df in tables.items():
        df.to_csv(f"{out_dir}/{name}.csv", index=False)
        print(f"{name:12s} {len(df):>8,d} rows")


if __name__ == "__main__":
    main()
