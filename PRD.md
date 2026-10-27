# PRD: CrediSite, early warning for construction credit

| | |
|---|---|
| **Author** | Ansh Arora |
| **Status** | Prototype built and backtested on synthetic data. Proposed: 8-week shadow pilot |
| **Last updated** | 27 Oct 2026 |
| **Live demo** | Dashboard (demo logins below) · code and data generator in this repo |

> All data in this prototype is synthetic. Company names are invented. Nothing here uses ArisInfra or any customer's data.

---

## 1. Problem

A B2B construction-materials platform supplies cement, TMT steel, RMC and aggregates to developers and EPC contractors **on credit**. Credit wins the business, but it makes the platform a lender to a cyclical industry.

Three things go wrong today:

1. **The alert that exists is useless.** Credit teams watch the aging report and react when dues pass 60 days. In construction almost everyone pays late, so in the simulated book this rule flags **61% of accounts every month** and only **9% of those alerts** are accounts that actually go bad. Analysts stop reading it.
2. **Limits are set once and forgotten.** A limit is sized at onboarding and rarely revisited, even when a developer's orders fall 80% and part-payments start. By the time dues hit 90 days, the damage is done.
3. **Collections chase the biggest balance, not the riskiest one.** A ₹60 lakh balance at a developer who always pays late but always pays gets the same call as a ₹40 lakh balance at one whose project has stalled.

**Why it matters:** receivables are funded with working capital. Listed players in this space have reported trade receivables at 46–58% of revenue and average credit cycles near 140 days ([DSIJ](https://insights.dsij.in/dsijarticledetail/arisinfra-solutions-rs-500-cr-ipo-opens-june-18-a-tech-driven-bet-on-indias-construction-supply-chain-should-you-invest)). Every day of DSO costs money, and every default costs far more.

## 2. Users and what each needs

| User | Job to be done | What they need from this tool | Tension |
|---|---|---|---|
| **Credit analyst** | Decide limits and terms; catch problems early | A short, ranked list of accounts that are actually slipping, *with reasons* they can defend | Wants fewer, better alerts |
| **Sales RM** | Grow volume with their developers | To know early if a customer is about to be cut off, and what would fix it | Wants to keep supplying |
| **Collections executive** | Get money in | Who to call today, what to say, which disputes to close first | Limited calls per day |
| **CFO** | Working capital, bad-debt provisions | Portfolio risk, DSO trend, what a policy change costs in sales | Risk vs growth |

The Sales vs Credit tension is the core product problem. A tool that only says "cut credit" will be ignored by Sales and overridden. The tool has to show **both sides of every credit decision**.

## 3. Goals and non-goals

**Goals**
- Flag accounts that will go 90+ days overdue **at least a month before** it happens.
- **Cut alert volume** relative to the aging report **without missing more bad accounts**.
- Turn each alert into a **specific action and limit**, with reasons an RM can repeat to the customer.
- Let leadership **price a credit policy**: loss avoided vs margin given up.

**Non-goals (v1)**
- Automatically blocking orders. Humans decide; the tool recommends.
- Underwriting brand-new customers with no payment history (needs bureau/GST data; separate project).
- Provisioning-grade PD. v1 ranks risk well, but its top band over-predicts (see §8).

## 4. Solution

| Capability | What it does |
|---|---|
| **Risk score (300–900)** | Monthly score per account from 8 payment-behaviour signals: collection efficiency, DSO and its trend, days paid late, limit utilisation, cheque/NACH bounces, challan dispute rate, order-volume change. Predicts the chance of crossing 90+ days overdue in the next 90 days. |
| **Reason codes** | Every score lists what raised and lowered it, in plain words ("Collected 40% of dues that fell due in the last 90 days"). Comes directly from each signal's weight, so it is exact, not a post-hoc story. |
| **Risk matrix → action** | Likelihood (Low / Medium / High) × exposure (₹50 L and ₹2 Cr breakpoints) maps to one of six actions, from *Continue* to *Pause supply + escalate*. |
| **Recommended limit** | Sized to the account's credit cycle and recent billing, scaled by tier. No new credit where the action is advance-only or a pause. |
| **Collections queue** | Ranked by overdue money × risk, with a next step per account (RM call, statement of account, close disputes first, insurance claim, legal notice). |
| **Policy simulator** | Change cutoffs, apply a payment-delay shock (e.g. monsoon or a sales freeze), and see sales held back, margin lost, expected loss avoided, DSO and funding cost. Recomputes scores in the browser with the same model. |
| **AI analyst** | Drafts a credit memo or a collections email from the account's data, and turns plain-English questions ("Pune developers with more than ₹20 lakh overdue 60+ days") into filters. Always reviewed by a person before sending. |
| **Decision log** | Analyst accepts or overrides the recommendation; an override requires a reason. This is the feedback loop for the next model version. |
| **Role-based views** | RMs see only their accounts and can't change limits. CFO sees portfolio, simulator and model. |

## 5. Key user stories and acceptance criteria

1. **As a credit analyst, I see which accounts need attention this week.**
   - Queue sorts worst-first by default; each row shows tier, PD, exposure, action and the top reason.
   - Clicking a risk-matrix cell filters the queue to those accounts.
   - Accounts already 90+ overdue are separated, because they belong to collections, not early warning.
2. **As a credit analyst, I can justify a limit change to Sales.**
   - Account page shows up to 4 risk reasons and 3 mitigants with the underlying numbers.
   - The recommendation states the action, the new limit and the current limit.
   - An override cannot be saved without a reason.
3. **As an RM, I'm not surprised by a credit hold.**
   - RM sees their own accounts' tiers and the reasons, but cannot edit limits.
   - (v2) RM gets a notification when an account moves up a tier, with the reasons.
4. **As a collections executive, I know who to call first.**
   - Collections list shows only accounts with dues 31+ days late, ranked by overdue × risk.
   - Each row has a next step; open challan disputes are called out to close first.
5. **As the CFO, I can see what tightening credit would cost.**
   - Simulator shows sales held back per month, margin given up, expected loss avoided over 90 days, and the break-even PD (= margin ÷ LGD).

## 6. Success metrics

**North star:** dues 90+ days overdue as a share of receivables.

| Type | Metric | Backtest / baseline | Pilot target |
|---|---|---|---|
| Model | Recall: bad accounts flagged Medium or High before going bad | 96% (aging rule: 87%) | ≥ 90% |
| Model | Precision of High-tier alerts | 24% (aging rule: 9%) | ≥ 20% |
| Model | Median warning before crossing 90+ days | 45 days | ≥ 30 days |
| Workflow | Alerts per month needing review | 51 (aging rule: 83) | ≤ 60% of today's |
| Workflow | High-tier accounts with a logged decision within 5 working days | n/a | ≥ 80% |
| Business | 90+ overdue share of receivables, pilot vs control region | measure | −15% in 2 quarters |
| Business | Portfolio DSO | measure | −5 days |
| **Guardrail** | Sales lost from credit actions on accounts that did *not* go bad | measure | track; review if > 2% of GMV |
| **Guardrail** | Analyst override rate | n/a | review model if > 30% |
| **Guardrail** | RM-reported false alarms | n/a | track weekly |

**The headline from the backtest:** Medium + High tiers together raise **fewer alerts than the aging report (51 vs 83 a month), catch more of the accounts that go bad (96% vs 87%), and are right almost twice as often (17% vs 9%)**.

## 7. Launch plan

1. **Shadow mode (weeks 1–8).** Score every account weekly; show it to the credit team; take no automatic action. Compare against what actually happens. Confirms calibration on real data.
2. **Region pilot (next quarter).** Use the recommendations in one region (e.g. Pune), keep another (e.g. MMR) on current process. Compare 90+ share, DSO and sales lost (difference-in-differences). Randomising individual customers is avoided: two developers in the same group could get different treatment.
3. **Scale.** Roll out to all regions; add RM notifications and weekly review.

**Dependencies:** invoice, receipt and credit-note data from the ERP; challan and weighbridge data from the document pipeline (disputes are a signal); bank feed for bounces.

## 8. Risks, edge cases and open questions

| Area | Issue | How v1 handles it |
|---|---|---|
| **New accounts** | Fewer than 60 days of history makes the score unreliable | Excluded from scoring; standard onboarding rules apply |
| **EPC contractors** | Pay in lumps when their own government bills clear; they look late but aren't | Collection efficiency over 90 days smooths lumps; v2: segment-specific weights |
| **Disputes inflate overdue** | A weighbridge shortfall holds up a whole invoice | Open disputes shown separately; collections told to close them first |
| **Group companies** | One promoter, several entities and limits | Not handled. v2: aggregate exposure by promoter group |
| **Seasonality** | Monsoon slows sites and payments every year | Trends compare 90-day windows; simulator lets you stress it |
| **Sudden shocks** | Project stopped by an authority, promoter dispute: little payment warning | Can't be seen in payment data. Needs RM input and outside data (RERA updates, news) |
| **Over-prediction** | Top PD band predicts 59%, observed 30% (some stressed accounts recover) | Fine for ranking and tiers. Recalibrate before using PD for provisions |
| **Gaming** | RMs nudge customers to pay just enough before month-end | Use trends and 90-day windows, not month-end balances alone |
| **Trust** | A weight with the "wrong" sign kills credibility | Features whose sign flips against credit logic are dropped automatically |

**Open questions**
- What gross margin and LGD should the simulator default to per material? (Today: 9% and 50%.)
- Should limit cuts be automatic below a PD threshold, or always need an analyst?
- Is trade-credit insurance claim timing a constraint on when to send legal notices?

## Appendix: backtest (synthetic data)

Out-of-time test: trained on month-end snapshots Jan–Dec 2025 (1,500 rows), tested on Jan–Jun 2026 (819 rows). Label: account crosses max(₹5 lakh, 10% of limit) overdue 90+ days at any month-end in the next 90 days. Base rate 6.5%.

| Method | Alerts / month | Precision | Recall |
|---|---|---|---|
| Aging report rule (any dues 60+ days overdue) | 83 | 9.2% | 87% |
| Limit breach rule (utilisation > 100%) | 73 | 6.4% | 53% |
| **Scorecard, High tier (PD ≥ 15%)** | **21** | **23.6%** | **55%** |
| **Scorecard, Medium + High (PD ≥ 5%)** | **51** | **16.7%** | **96%** |
| Gradient boosting (black-box benchmark) | 15 | 33.3% | 57% |

AUC 0.86 (scorecard) vs 0.88 (gradient boosting). We give up about 1.5 points of AUC for reason codes a credit team can defend. All 18 accounts that went bad in the test period were flagged High before they crossed 90 days overdue; median warning 45 days (range 28–183).
