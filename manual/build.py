"""
Build the scAnimator manual PDF.

  1. runs capture.py unless --no-capture (every picture, from the real viewer)
  2. prints scanimator.html twice with headless Edge: the first pass finds the
     page each chapter starts on, the second prints the contents and the
     cross-references with those numbers

Run:  python manual/build.py [--no-capture]
Needs: pip install pymupdf (and capture.py's: playwright pillow numpy)
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = os.path.join(HERE, "scAnimator_Manual.pdf")


def edge_print(html_path, pdf_path):
    """msedge.exe returns before the PDF is written, so the old file is
    removed first and the new one waited for until its size settles."""
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    url = "file:///" + html_path.replace("\\", "/")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=5000", f"--print-to-pdf={pdf_path}", url],
                   check=True, capture_output=True)
    last = -1
    for _ in range(240):
        time.sleep(0.5)
        size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else -1
        if size > 0 and size == last:
            return
        last = size
    raise SystemExit(f"Edge did not write {pdf_path}")


def print_pdf():
    import pymupdf
    with open(os.path.join(HERE, "scanimator.html"), encoding="utf-8") as f:
        src = f.read()
    ids = re.findall(r'<section class="chapter[^"]*" id="([\w-]+)"', src)
    tmp_html = os.path.join(HERE, "_print.html")
    tmp_pdf = os.path.join(HERE, "_pass1.pdf")

    marked = src
    for i in ids:
        marked = re.sub(rf'(<section class="chapter[^"]*" id="{i}"[^>]*>)', rf'\1<span class="mark">@@{i}@@</span>', marked, count=1)
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(marked)
    edge_print(tmp_html, tmp_pdf)
    pages = {}
    with pymupdf.open(tmp_pdf) as doc:
        for n, pg in enumerate(doc):
            for m in re.findall(r"@@([\w-]+)@@", pg.get_text()):
                pages.setdefault(m, n + 1)
    missing = [i for i in ids if i not in pages]
    if missing:
        raise SystemExit(f"chapter markers not found: {missing}")

    final = re.sub(r'<span class="pg" data-ref="([\w-]+)">[^<]*</span>',
                   lambda m: f'<span class="pg">{pages[m.group(1)]}</span>', src)
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(final)
    edge_print(tmp_html, OUT)
    os.remove(tmp_html)
    os.remove(tmp_pdf)
    with pymupdf.open(OUT) as doc:
        print(f"wrote {OUT} ({doc.page_count} pages, {os.path.getsize(OUT) / 1e6:.1f} MB)")


def main():
    if "--no-capture" not in sys.argv:
        subprocess.run([sys.executable, os.path.join(HERE, "capture.py")], check=True)
    print_pdf()


if __name__ == "__main__":
    main()
