"""Central configuration. Every number here is an assumption you can defend in an interview."""

SEED = 42
START = "2024-10-07"          # first simulated week (Monday)
AS_OF = "2026-10-25"          # "today" for the live dashboard
N_DEVELOPERS = 160

# Label: a developer "goes bad" if, at any month-end in the next 90 days,
# dues overdue by 90+ days exceed max(Rs 5 lakh, 10% of their credit limit).
LABEL_HORIZON_DAYS = 90
BAD_MIN_AMOUNT = 5e5
BAD_LIMIT_SHARE = 0.10

# Time split for honest evaluation (no peeking into the future)
TRAIN_END = "2025-12-31"      # snapshots up to here train the model
TEST_END = "2026-06-30"       # last snapshot whose 90-day label is fully observed

# Risk policy
PD_TIERS = {"Low": 0.05, "Medium": 0.15}          # PD < 5% Low, 5-15% Medium, else High
EXPOSURE_TIERS = {"Low": 50e5, "Medium": 200e5}   # < Rs 50L Low, 50L-2Cr Medium, else High
LGD = 0.50                    # share of exposure lost on default (net of recovery / insurance)
COST_OF_FUNDS = 0.12          # annual cost of working-capital borrowing used to fund receivables

ACTION_MATRIX = {
    ("Low", "Low"): "Continue",
    ("Low", "Medium"): "Continue",
    ("Low", "High"): "Continue + monitor",
    ("Medium", "Low"): "Continue + monitor",
    ("Medium", "Medium"): "Reduce limit / shorten terms",
    ("Medium", "High"): "Reduce limit + RM visit",
    ("High", "Low"): "Advance payment only",
    ("High", "Medium"): "Pause new supply",
    ("High", "High"): "Pause supply + escalate",
}

# Score scaling (familiar 300-900 range). 600 points at 50:1 good:bad odds, 40 points to double the odds.
SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO = 600, 50, 40
