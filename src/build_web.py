"""
Bundle the dashboard into single self-contained HTML files.

  dist/credisite.html  page body only (for hosts that add their own <html> skeleton)
  docs/index.html      complete page for GitHub Pages (Settings > Pages > main branch, /docs)

Run after the pipeline:  python -m src.build_web
"""
import json
import os

HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}'
        'img{max-width:100%}[hidden]{display:none!important}</style></head><body>')


def main():
    with open("web/data.json") as fh:
        data = json.load(fh)
    seed = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open("web/app.html") as fh:
        page = fh.read().replace("__SEED__", seed)
    os.makedirs("dist", exist_ok=True); os.makedirs("docs", exist_ok=True)
    with open("dist/credisite.html", "w") as fh:
        fh.write(page)
    with open("docs/index.html", "w") as fh:
        fh.write(HEAD + page + "</body></html>")
    print(f"dist/credisite.html  {len(page)/1024:.0f} KB")
    print(f"docs/index.html      {(len(page)+len(HEAD))/1024:.0f} KB")


if __name__ == "__main__":
    main()
