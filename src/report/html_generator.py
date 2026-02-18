# src/report/html_generator.py
from pathlib import Path
import base64
import shutil

try:
    # preferred: install `markdown` package (pip install markdown)
    import markdown as md

    HAVE_MARKDOWN = True
except Exception:
    HAVE_MARKDOWN = False

# TODO: refactor w.r.t new benchmark framwork
def _embed_image_tag(img_path: Path) -> str:
    """Return HTML <img> tag with image data URI (embedded)"""
    img_bytes = img_path.read_bytes()
    ext = img_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f'<img src="data:{mime};base64,{b64}" alt="{img_path.name}" style="max-width:100%;height:auto;" />'


def convert_markdown_to_html(md_text: str) -> str:
    if HAVE_MARKDOWN:
        # use python-markdown for better fidelity
        return md.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    # fallback: a very small converter for common elements
    lines = md_text.splitlines()
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("```"):
            # naive codeblock handling
            if line.strip() == "```":
                out.append("<pre><code>")
            else:
                out.append("<pre><code>")
        elif line.strip() == "```" and out and out[-1].startswith("<pre>"):
            out.append("</code></pre>")
        elif line.startswith("![](") and ")" in line:
            # image markdown: ![](path)
            start = line.find("![](") + 4
            end = line.find(")", start)
            path = line[start:end]
            out.append(
                f'<img src="{path}" alt="" style="max-width:100%;height:auto;" />'
            )
        else:
            out.append(f"<p>{line}</p>")
    return "\n".join(out)


def generate_html_report_from_md(
    exp_dir: str | Path, md_filename: str = "report.md"
) -> Path:
    exp_dir = Path(exp_dir)
    md_path = exp_dir / md_filename
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown report not found: {md_path}")

    md_text = md_path.read_text(encoding="utf-8")

    # Replace Markdown image references (relative) with embedded data URIs
    # Find lines like ![](plots/foo.png)
    lines = md_text.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("![](") and line.strip().endswith(")"):
            # get path between parentheses
            p = line.strip()[4:-1]
            img_path = (exp_dir / p).resolve()
            if img_path.exists():
                new_lines.append(_embed_image_tag(img_path))
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    md_text_embedded = "\n".join(new_lines)

    html_body = convert_markdown_to_html(md_text_embedded)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Benchmark Report - {exp_dir.name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; padding: 24px; max-width: 1100px; margin: auto; background: #fff; color: #111; }}
pre {{ background: #f6f8fa; padding: 12px; overflow:auto; border-radius:6px; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace; }}
img {{ box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius:4px; margin:10px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    out_path = exp_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
