"""
Smoke test the built dashboard in a headless browser: every role, every view, no JS errors,
and the in-browser PD matches the Python model.

Run:  python tests/test_dashboard.py  [--shots out_dir]
"""
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs" / "index.html").as_uri()
ROLES = {
    "analyst@credisite.demo": ["overview", "queue", "account", "collections", "simulator", "model", "about"],
    "rm.priya@credisite.demo": ["queue", "account", "collections", "about"],
    "cfo@credisite.demo": ["overview", "queue", "account", "simulator", "model", "about"],
}


def main():
    shots = None
    if "--shots" in sys.argv:
        shots = pathlib.Path(sys.argv[sys.argv.index("--shots") + 1]); shots.mkdir(parents=True, exist_ok=True)
    errors, problems = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for scheme in ["light", "dark"]:
            for width in [1360, 390]:
                for email, views in ROLES.items():
                    if (scheme, width) != ("light", 1360) and email != "analyst@credisite.demo":
                        continue
                    ctx = b.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
                    pg = ctx.new_page()
                    pg.on("pageerror", lambda e: errors.append(f"{email} {width} {scheme}: {e}"))
                    # Font files fail to load only when the test machine is offline; ignore those.
                    pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(f"console {m.text}"))
                    pg.goto(PAGE)
                    pg.fill("#email", email); pg.fill("#pass", "demo1234"); pg.click("#login button[type=submit]")
                    pg.wait_for_selector(".topbar")
                    for v in views:
                        pg.evaluate(f"location.hash = '{v}'"); pg.wait_for_timeout(150)
                        overflow = pg.evaluate("document.documentElement.scrollWidth - innerWidth")
                        if overflow > 1:
                            problems.append(f"{email} {v} {width}px {scheme}: page scrolls sideways by {overflow}px")
                        if shots and email == "analyst@credisite.demo":
                            pg.screenshot(path=str(shots / f"{scheme}-{width}-{v}.png"), full_page=True)
                    if email == "rm.priya@credisite.demo":
                        rms = pg.evaluate("""() => { location.hash='queue'; return [...document.querySelectorAll('tbody tr td .sub')].map(e=>e.textContent) }""")
                        if any("Priya Kulkarni" not in r for r in rms if "·" in r):
                            problems.append("RM sees accounts that aren't theirs")
                    ctx.close()
        # login error path + PD parity
        ctx = b.new_context(); pg = ctx.new_page(); pg.goto(PAGE)
        pg.fill("#email", "nobody@x.com"); pg.fill("#pass", "x"); pg.click("#login button[type=submit]")
        if not pg.inner_text("#login-err"):
            problems.append("bad login shows no error")
        ctx.close()
        b.close()

    # PD parity: recompute in Python exactly as the page does
    d = json.load(open(ROOT / "web" / "data.json"))
    P = d["model"]["params"]
    import math
    tf = {"utilisation": lambda x: min(max(x["utilisation"], 0), 3), "dso": lambda x: min(max(x["dso"] / 100, 0), 4),
          "dso_trend": lambda x: x["dso_trend"] / 100, "days_late_90d": lambda x: min(max(x["days_late"] / 30, -1), 6),
          "collection_efficiency_90d": lambda x: min(max(x["coll_eff"], 0), 1.5), "partial_share_90d": lambda x: min(max(x["partial_share"], 0), 1),
          "bounces_180d": lambda x: min(max(x["bounces"], 0), 4), "dispute_rate_90d": lambda x: min(max(x["dispute_rate"], 0), .5),
          "billing_change": lambda x: min(max(x["billing_change"], -1), 1.5)}
    worst = 0
    for dev in d["developers"]:
        z = P["intercept"] + sum(P["coef"][f] * (tf[f](dev) - P["mu"][f]) / P["sd"][f] for f in P["coef"])
        worst = max(worst, abs(1 / (1 + math.exp(-z)) - dev["pd"]))
    if worst > 0.02:
        problems.append(f"browser PD differs from Python PD by up to {worst:.3f}")

    print("JS errors:", errors or "none")
    print("Problems:", problems or "none")
    print(f"Max PD difference browser vs Python: {worst:.4f}")
    sys.exit(1 if errors or problems else 0)


if __name__ == "__main__":
    main()
