"""Render a storyboard (scripted, captioned scAnimator video) to MP4 clips.

The viewer does the drawing (viewer.html, "Storyboard" section: the JSON format
is documented there); this script serves the scAnimator folder, opens the
viewer in Edge through Playwright, loads the storyboard and pulls frames out of
it one at a time, encoding them with ffmpeg (imageio-ffmpeg's bundled binary).

    python scripts/render_storyboard.py <storyboard.json> [--out DIR]
        clips: per step a transition (NNa_<name>_in.mp4, from step 2 on) and a
        hold of whole camera turns that loops seamlessly (NNb_<name>_hold.mp4),
        plus _full.mp4 (everything in order, each hold played once)
    ... --stills     one frame per hold and mid-transition + contact.png (quick check)
    ... --draft      half size, 15 fps (faster clips for checking timing)
    ... --only a,b   just these steps (by name)
    ... --preview    opens a window playing it live (arrow keys jump steps); close it to quit

Outputs go to <storyboard dir>/<storyboard name>/ unless --out is given. They
show real data: keep them out of this repository.
Needs: pip install playwright imageio-ffmpeg (Edge is used as installed).
"""
import argparse
import base64
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Static server with HTTP Range support (gene expression is fetched by range) ──


class _Slice:
    def __init__(self, f, n):
        self.f, self.left = f, n

    def read(self, n=-1):
        if self.left <= 0:
            return b''
        n = self.left if n < 0 else min(n, self.left)
        data = self.f.read(n)
        self.left -= len(data)
        return data

    def close(self):
        self.f.close()


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get('Range')
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        m = re.match(r'bytes=(\d*)-(\d*)', rng)
        start = int(m[1]) if m and m[1] else 0
        end = min(int(m[2]) if m and m[2] else size - 1, size - 1)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        f.seek(start)
        return _Slice(f, end - start + 1)

    def log_message(self, *args):
        pass


def serve():
    handler = functools.partial(RangeHandler, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ── Browser ──


def open_viewer(pw, dataset, port, size, headed, preview):
    args = ['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11']
    if preview:
        args.append('--window-size=1400,1000')
    browser = pw.chromium.launch(channel='msedge', headless=not (headed or preview), args=args)
    if preview:
        ctx = browser.new_context(no_viewport=True)
    else:
        ctx = browser.new_context(viewport={'width': size, 'height': size}, device_scale_factor=1)
    page = ctx.new_page()
    page.on('pageerror', lambda e: print(f'[page error] {e}', file=sys.stderr))
    page.goto(f'http://127.0.0.1:{port}/viewer.html?dataset={dataset}')
    page.wait_for_function('window.storyboard && window.storyboard.ready', timeout=300_000)
    return browser, page


def frame_bytes(page, t, kind):
    url = page.evaluate('([t, type]) => window.storyboard.frame(t, type, 0.95)',
                        [t, 'image/png' if kind == 'png' else 'image/jpeg'])
    return base64.b64decode(url.split(',', 1)[1])


# ── Encoding ──


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(page, path, start, frames, fps, kind):
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error',
           '-f', 'image2pipe', '-framerate', str(fps), '-c:v', 'png' if kind == 'png' else 'mjpeg', '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
           '-r', str(fps), '-movflags', '+faststart', str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k in range(frames):
        proc.stdin.write(frame_bytes(page, start + k / fps, kind))
        if (k + 1) % 60 == 0 or k + 1 == frames:
            print(f'\r  {path.name}: {k + 1}/{frames} frames ({time.time() - t0:.0f} s)', end='', flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f'ffmpeg failed on {path}')
    print()


def concat(paths, out):
    listing = out.with_suffix('.txt')
    listing.write_text(''.join(f"file '{p.name}'\n" for p in paths), encoding='utf-8')
    subprocess.run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(out)], check=True)
    listing.unlink()


# ── Stills ──


def contact_sheet(stills, out, cols=3, thumb=480):
    from PIL import Image, ImageDraw, ImageFont
    rows = (len(stills) + cols - 1) // cols
    label_h = 34
    sheet = Image.new('RGB', (cols * thumb, rows * (thumb + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('segoeui.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    for i, (path, label) in enumerate(stills):
        img = Image.open(path).convert('RGB').resize((thumb, thumb), Image.LANCZOS)
        x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
        sheet.paste(img, (x, y))
        draw.text((x + 8, y + thumb + 6), label, fill=(220, 220, 220), font=font)
    sheet.save(out)


# ── Main ──


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('storyboard')
    ap.add_argument('--out')
    ap.add_argument('--dataset', help='overrides the storyboard\'s "dataset"')
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--only', help='comma-separated step names')
    ap.add_argument('--headed', action='store_true', help='render in a visible window')
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()

    sb_path = Path(a.storyboard).resolve()
    board = json.loads(sb_path.read_text(encoding='utf-8'))
    if a.draft:
        board['size'] = board.get('size', 1080) // 2
        board['fps'] = 15
    dataset = a.dataset or board.get('dataset')
    if not dataset:
        raise SystemExit('no dataset: set "dataset" in the storyboard or pass --dataset')
    out = Path(a.out).resolve() if a.out else sb_path.parent / sb_path.stem
    if a.draft:
        out = out / 'draft'

    from playwright.sync_api import sync_playwright, Error as PwError
    srv = serve()
    port = srv.server_address[1]
    with sync_playwright() as pw:
        browser, page = open_viewer(pw, dataset, port, board.get('size', 1080), a.headed, a.preview)
        try:
            plan = page.evaluate('j => window.storyboard.load(j)', board)
        except PwError as e:
            raise SystemExit(f'storyboard rejected: {e.message.splitlines()[0]}')
        print(f'[info] gpu: {page.evaluate("window.storyboard.gpu()")}')
        print(f'[info] {len(plan["steps"])} steps, {plan["total"]:.1f} s, {plan["fps"]} fps, {plan["size"]} px')
        cam = plan['camera']
        print(f'[info] camera at {cam["distance"]:.2f} from {[round(v, 2) for v in cam["target"]]} '
              f'(cloud radius {cam["fit_radius"]:.2f})')
        for st in plan['steps']:
            mapped = f' ({st["gene_mapping"]} by {st["gene"]})' if st.get('gene') else ''
            print(f'[info]   {st["name"]}: {st["cells"]:,} cells by {st["color_by"]}{mapped}, '
                  f'n_neighbors {st["n_neighbors"]}, "{st["caption"]}"')
        for w in plan['warnings']:
            print(f'[warn] {w}')

        if a.preview:
            page.evaluate('window.storyboard.play()')
            print('[info] playing; close the window to quit')
            page.wait_for_event('close', timeout=0)
            return

        only = set(a.only.split(',')) if a.only else None
        segs = [s for s in plan['segments'] if not only or s['name'] in only]
        out.mkdir(parents=True, exist_ok=True)
        fps = plan['fps']

        if a.stills:
            sdir = out / 'stills'
            sdir.mkdir(exist_ok=True)
            stills = []
            for s in segs:
                at = s['start'] + (s['dur'] / 2 if s['kind'] == 'in' else 0)
                tag = 'a_mid' if s['kind'] == 'in' else 'b'
                path = sdir / f'{s["step"] + 1:02d}{tag}_{s["name"]}.png'
                path.write_bytes(frame_bytes(page, at, 'png'))
                stills.append((path, f'{s["step"] + 1}. {s["name"]} ({"mid-transition" if s["kind"] == "in" else "hold"}, {at:.1f} s)'))
            contact_sheet(stills, out / 'contact.png')
            print(f'[info] {len(stills)} stills, {out / "contact.png"}')
            return

        kind = 'jpeg' if a.draft else 'png'
        clips = []
        t0 = time.time()
        for s in segs:
            tag = 'a' if s['kind'] == 'in' else 'b'
            path = out / f'{s["step"] + 1:02d}{tag}_{s["name"]}_{s["kind"]}.mp4'
            encode(page, path, s['start'], s['frames'], fps, kind)
            clips.append(path)
        if not only:
            concat(clips, out / '_full.mp4')
            clips.append(out / '_full.mp4')
        print(f'[info] {len(clips)} files in {out} ({time.time() - t0:.0f} s)')
        browser.close()
    srv.shutdown()


if __name__ == '__main__':
    main()
