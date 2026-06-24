#!/usr/bin/env python3
"""Build ARTHJAX_SHOWCASE.pdf without pdflatex (HTML + Chrome headless)."""
import base64
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MD = DOCS / "ARTHJAX_SHOWCASE.md"
HTML = DOCS / "ARTHJAX_SHOWCASE.html"
PDF = DOCS / "ARTHJAX_SHOWCASE.pdf"
ASSETS = DOCS / "assets"

FIGURES = [
    ("macro_evolution_v2.png", "Figure 1 — Macro dashboard"),
    ("linkedin_hero.png", "Figure 2 — LinkedIn hero chart"),
    ("boom_bust_v2.png", "Figure 3 — Boom-bust & contagion"),
    ("world_model_loss_v2.png", "Figure 4 — World model training loss"),
    ("real_vs_learned_v2.png", "Figure 5 — Real vs learned economy"),
]

PRINT_CSS = """
@page { margin: 18mm 14mm; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1a1a1a;
  max-width: 900px;
  margin: 0 auto;
  padding: 12px;
}
h1 { font-size: 22pt; border-bottom: 2px solid #2563eb; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #1e40af; margin-top: 1.4em; }
h3 { font-size: 12pt; color: #334155; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }
th { background: #f1f5f9; }
code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 6px; font-size: 9pt; }
blockquote { border-left: 4px solid #2563eb; margin: 12px 0; padding: 8px 14px; background: #f8fafc; }
.figure-block { margin: 20px 0 28px; page-break-inside: avoid; }
.figure-block img {
  width: 100%;
  max-height: 420px;
  object-fit: contain;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #0f1117;
}
.figure-caption { font-size: 9pt; color: #64748b; margin-top: 6px; text-align: center; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }
"""


def pandoc_html_body(md_text: str) -> str:
    proc = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html", "--wrap=none"],
        input=md_text,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def inject_figure_images(html: str) -> str:
    for fname, alt in FIGURES:
        path = ROOT / fname
        asset_path = ASSETS / fname
        if not path.exists() and asset_path.exists():
            path = asset_path
        if fname not in html and path.exists():
            pattern = re.compile(
                rf"(<h3[^>]*>.*?{re.escape(fname)}.*?</h3>)",
                re.DOTALL | re.IGNORECASE,
            )
            block = (
                f'<div class="figure-block">'
                f'<img src="{path.as_uri()}" alt="{alt}"/>'
                f'<div class="figure-caption">{alt}</div></div>'
            )
            html, n = pattern.subn(r"\1" + block, html, count=1)
            if n == 0:
                html += block
    return html


def find_chrome() -> str:
    for cmd in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            return cmd
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return ""


def main() -> int:
    if not MD.exists():
        print(f"Missing {MD}", file=sys.stderr)
        return 1

    md_text = MD.read_text(encoding="utf-8")
    body = pandoc_html_body(md_text)
    body = inject_figure_images(body)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ArthJAX Showcase</title>
<style>{PRINT_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""
    HTML.write_text(full_html, encoding="utf-8")
    print(f"✓ HTML: {HTML}")

    chrome = find_chrome()
    if not chrome:
        print("Chrome/Chromium not found. Open HTML in browser → Print → Save as PDF.")
        return 0

    url = HTML.as_uri()
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}",
            url,
        ],
        check=True,
    )
    print(f"✓ PDF:  {PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
