"""
Pictures for the README and the manual, taken from the real viewer.

Opens viewer.html in Edge through Playwright (on the GPU, like
scripts/render_storyboard.py), drives it the way a person would (the panel's
own selects, sliders and buttons) and saves:

  media/banner.png        README banner (wide, no UI)
  media/warp.gif          the n_neighbors warp, animated
  media/interface.png     the whole viewer, panel and legend showing
  manual/img/*.png        every picture in the manual

Cloud-only pictures are framed by the viewer's storyboard engine (its camera
fits the cloud), so they are the same on every run.

Datasets: pbmc3k comes from this repository. Pictures marked LAB use a larger
dataset that is NOT in the repository (only pictures of it are); they are
taken from --extra-datasets (default C:\\Users\\Nautilus\\scAnimator\\datasets)
and fall back to pbmc3k when it is missing.

Run:  python manual/capture.py            (every picture)
      python manual/capture.py banner gif (only pictures whose name contains a word)
Needs: pip install playwright pillow numpy
"""
import argparse
import base64
import functools
import http.server
import io
import json
import re
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import render_storyboard as rs  # noqa: E402  (its Range-capable static server)

IMG = ROOT / 'manual' / 'img'
MEDIA = ROOT / 'media'
LAB_DEFAULT = Path(r'C:\Users\Nautilus\scAnimator\datasets')
LAB_ID = 'allcombined'

ap = argparse.ArgumentParser()
ap.add_argument('only', nargs='*')
ap.add_argument('--extra-datasets', default=str(LAB_DEFAULT))
ap.add_argument('--headed', action='store_true')
A = ap.parse_args()
EXTRA = Path(A.extra_datasets)
LAB = LAB_ID if (EXTRA / LAB_ID / 'metadata.json').is_file() else 'pbmc3k'


# ---------------------------------------------------------------- server
class Handler(rs.RangeHandler):
    """The repository, plus datasets that live outside it."""
    def translate_path(self, path):
        m = re.match(r'/datasets/([^/?]+)/([^?]+)', path)
        if m and not (ROOT / 'datasets' / m[1]).is_dir():
            return str(EXTRA / m[1] / m[2])
        return super().translate_path(path)


def serve():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------- driving the viewer
class Viewer:
    def __init__(self, browser, port, dataset, w=1600, h=1000, scale=1):
        self.page = browser.new_page(viewport={'width': w, 'height': h}, device_scale_factor=scale)
        self.page.on('pageerror', lambda e: print(f'[page error] {e.stack}', flush=True))
        self.page.goto(f'http://127.0.0.1:{port}/viewer.html?dataset={dataset}')
        self.page.wait_for_function('window.storyboard && window.storyboard.ready', timeout=300_000)
        self.page.wait_for_function("document.getElementById('panel').style.display === ''")
        self.page.wait_for_timeout(400)

    def set(self, id_, value):
        """Set a control and fire the event its handler listens for."""
        self.page.evaluate("""([id, v]) => { const e = document.getElementById(id);
            if (e.type === 'checkbox') e.checked = !!v; else e.value = v;
            const ev = (e.tagName === 'SELECT' || e.type === 'checkbox') ? 'change' : 'input';
            e.dispatchEvent(new Event(ev, {bubbles: true})); }""", [id_, value if isinstance(value, bool) else str(value)])

    def click(self, sel):
        self.page.click(sel)

    def open_group(self, title):
        """Expand a panel group by its title (and collapse the others)."""
        self.page.evaluate("""t => { for (const g of document.querySelectorAll('#panel > .panel-group')) {
            const name = g.querySelector('.panel-group-title').textContent.trim();
            g.classList.toggle('collapsed', name !== t); } }""", title)

    def open_sub(self, title):
        self.page.evaluate("""t => { for (const g of document.querySelectorAll('.panel-group .panel-group')) {
            if (g.querySelector('.panel-group-title').textContent.trim() === t) g.classList.remove('collapsed'); } }""", title)

    def pause_at(self, nn=None, slider=None):
        """Pause and hold one n_neighbors frame (morph off)."""
        p = self.page
        if p.inner_text('#btn-play') == 'Pause':
            p.click('#btn-play')
        if slider is not None:
            self.set('slider', slider)
        if nn is not None:
            labels = p.evaluate('async () => (await (await fetch("datasets/" + '
                                'new URLSearchParams(location.search).get("dataset") + "/metadata.json")).json()).nn_labels')
            # slider runs over the whole ping-pong timeline; the first half sweeps up
            n = len(labels)
            best = min(range(n // 2 + 1), key=lambda i: abs(float(labels[i]) - nn))
            self.set('slider', round(1000 * best / (n - 1)))
        p.wait_for_timeout(300)

    def wheel(self, n, x=None, y=None):
        vp = self.page.viewport_size
        self.page.mouse.move(x or vp['width'] * 0.6, y or vp['height'] * 0.5)
        for _ in range(abs(n)):
            self.page.mouse.wheel(0, -100 if n > 0 else 100)
            self.page.wait_for_timeout(40)

    def drag(self, dx, dy, x=None, y=None):
        vp = self.page.viewport_size
        x = x or vp['width'] * 0.6
        y = y or vp['height'] * 0.5
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        self.page.mouse.move(x + dx, y + dy, steps=12)
        self.page.mouse.up()

    def hide_ui(self):
        self.click('#btn-hide-ui')
        self.page.evaluate("document.getElementById('util-buttons').style.visibility = 'hidden'")

    def settle(self, ms=1200):
        self.page.wait_for_timeout(ms)

    def shot(self, path, clip=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), clip=clip)
        print(f'  {path.relative_to(ROOT)}')

    def close(self):
        self.page.close()


def storyboard_frames(viewer, board, times):
    """Rendered storyboard frames (square, board['size'] px) as PIL images."""
    viewer.page.evaluate('j => window.storyboard.load(j)', board)
    out = []
    for t in times:
        url = viewer.page.evaluate('t => window.storyboard.frame(t, "image/png")', t)
        out.append(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB'))
    return out


def want(name):
    return not A.only or any(w in name for w in A.only)


# ---------------------------------------------------------------- the pictures
LOOK = {'point_size': 18, 'brightness': 0.7, 'bloom': 0.35}      # the hero look (LAB)
CAM = {'elevation': 15, 'start_angle': 120}


def fit(v, nn=30, zoom=1.0, **cam):
    """Frame the cloud like a storyboard would, then hand the viewer back."""
    board = {'steps': [{'name': 'fit', 'n_neighbors': nn, 'legend': False}],
             'camera': {**CAM, 'zoom': zoom, **cam}}
    v.page.evaluate('async j => { await window.storyboard.load(j); window.storyboard.frame(0);'
                    ' window.storyboard.unload(); }', board)
    v.pause_at(nn=nn)


def look(v, size=None, brightness=None, bloom=None):
    for id_, val in (('point-size', size), ('brightness', brightness), ('bloom-strength', bloom)):
        if val is not None:
            v.set(id_, val)


def gene(v, name, mapping='colormap', cmap='viridis'):
    v.open_group('Gene Expression')
    v.set('gene-mapping', mapping)
    if mapping == 'colormap':
        v.set('gene-colormap', cmap)
    v.page.fill('#gene-input', name)
    v.page.press('#gene-input', 'Enter')
    v.page.wait_for_function("!/loading/i.test(document.getElementById('gene-status').textContent)", timeout=60_000)
    v.page.wait_for_timeout(500)


def uncheck(v, column, values):
    v.open_group('Filter')
    v.set('filter-col', column)
    v.page.evaluate("""vals => { for (const row of document.querySelectorAll('#filter-checkboxes label')) {
        const cb = row.querySelector('input');
        if (vals.includes(row.textContent.trim()) && cb.checked) cb.click(); } }""", values)


def clip_of(v, sel, pad=8):
    box = v.page.locator(sel).bounding_box()
    return {'x': max(0, box['x'] - pad), 'y': max(0, box['y'] - pad),
            'width': box['width'] + 2 * pad, 'height': box['height'] + 2 * pad}


def stage_tiles(b, port):
    """The warp in five stills (banner + manual). LAB"""
    v = Viewer(b, port, LAB, 1000, 1000)
    for nn in (5, 10, 20, 40, 150):
        board = {'size': 800, 'steps': [{'name': 's', 'n_neighbors': nn, 'legend': False}], 'render': LOOK,
                 'camera': {**CAM, 'zoom': 1.05}}
        storyboard_frames(v, board, [0])[0].save(IMG / f'stage_{nn}.png')
        print(f'  manual/img/stage_{nn}.png')
    v.close()


def banner(b, port):
    """media/banner.png from manual/banner.html (title + the five stages)."""
    page = b.new_page(viewport={'width': 2400, 'height': 1000}, device_scale_factor=1)
    page.goto((ROOT / 'manual' / 'banner.html').as_uri())
    page.wait_for_timeout(500)
    box = page.locator('.card').bounding_box()
    page.screenshot(path=str(MEDIA / 'banner.png'), clip=box, omit_background=True)
    print('  media/banner.png')
    page.close()


def gif(b, port):
    """media/warp.gif: n_neighbors 5 -> 30 -> 5 during one camera turn, so it loops seamlessly. LAB"""
    v = Viewer(b, port, LAB, 1000, 1000)
    T, fps, size = 16, 12, 440
    board = {'size': size, 'fps': fps, 'turn_seconds': T, 'render': {**LOOK, 'bloom': 0.1}, 'camera': {**CAM, 'zoom': 1.0},
             'steps': [{'name': 'lo', 'n_neighbors': 5, 'turns': 2 / T, 'legend': False},
                       {'name': 'hi', 'n_neighbors': 30, 'turns': 2 / T, 'transition': 6},
                       {'name': 'back', 'n_neighbors': 5, 'turns': 0, 'transition': 6}]}
    frames = storyboard_frames(v, board, [k / fps for k in range(T * fps)])
    v.close()
    # one shared palette from a spread of frames keeps the colours steady
    picks = frames[::max(1, len(frames) // 16)][:16]
    strip = Image.new('RGB', (size, size * len(picks)))
    for i, f in enumerate(picks):
        strip.paste(f, (0, i * size))
    pal = strip.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = MEDIA / 'warp.gif'
    q[0].save(out, save_all=True, append_images=q[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f'  media/warp.gif ({out.stat().st_size / 1e6:.1f} MB, {len(q)} frames)')


def interface(b, port):
    """media/interface.png and manual/img/ui_overview.png: the whole viewer. LAB"""
    v = Viewer(b, port, LAB, 1600, 1000)
    fit(v, nn=8, zoom=0.85)
    look(v, 14, 0.8, 0.35)
    v.open_group('Render Settings')
    v.settle(1500)
    v.shot(MEDIA / 'interface.png')
    v.open_group('')
    v.settle(300)
    v.shot(IMG / 'ui_overview.png')
    v.close()


def pbmc(b, port, nn=5, zoom=1.3, size=40, brightness=0.8, bloom=0.5, w=1600, h=1000):
    v = Viewer(b, port, 'pbmc3k', w, h)
    fit(v, nn=nn, zoom=zoom, start_angle=0)
    look(v, size, brightness, bloom)
    return v


def features(b, port):
    """The manual's feature pictures, on the public PBMC 3k demo data."""
    v = pbmc(b, port)
    v.set('color-by', 'cell_type')
    v.open_group('Color By')
    v.settle()
    v.shot(IMG / 'color_by.png')
    v.shot(IMG / 'panel_color_by.png', clip=clip_of(v, '#panel'))
    for name in ('CD4 T', 'CD14+ Mono'):          # legend toggles
        v.page.locator('#legend .legend-item', has_text=name).first.click()
    v.open_group('')
    v.settle()
    v.shot(IMG / 'legend_toggle.png')
    v.close()

    v = pbmc(b, port)
    v.set('color-by', 'cell_type')
    uncheck(v, 'leiden', ['0', '1'])
    v.settle()
    v.shot(IMG / 'filter.png')
    v.shot(IMG / 'panel_filter.png', clip=clip_of(v, '#panel'))
    v.close()

    v = pbmc(b, port)
    v.set('color-by', 'cell_type')
    v.open_group('Normalize')
    v.set('normalize-col', 'cell_type')
    v.click('#btn-normalize')
    v.settle()
    v.shot(IMG / 'normalize.png')
    v.shot(IMG / 'panel_normalize.png', clip=clip_of(v, '#panel'))
    v.close()

    for g, mapping, cmap, tag in (('LYZ', 'colormap', 'magma', 'gene_colormap'),
                                  ('MS4A1', 'size', None, 'gene_size'),
                                  ('NKG7', 'intensity', None, 'gene_intensity')):
        v = pbmc(b, port)
        v.set('color-by', 'cell_type')
        gene(v, g, mapping, cmap or 'viridis')
        v.settle()
        v.shot(IMG / f'{tag}.png')
        if tag == 'gene_colormap':
            v.shot(IMG / 'panel_gene.png', clip=clip_of(v, '#panel'))
        v.close()

    v = pbmc(b, port)
    v.open_group('Gene Expression')
    v.page.fill('#gene-input', 'CD')
    v.page.wait_for_timeout(600)
    v.shot(IMG / 'panel_gene_autocomplete.png', clip=clip_of(v, '#panel'))
    v.close()

    v = pbmc(b, port)
    v.set('color-by', '__gradient_cluster__')
    v.open_group('Color By')
    v.click('#btn-play')            # gradients move with the timeline
    v.settle(2500)
    v.shot(IMG / 'gradient_cluster.png')
    v.close()


def render_effects(b, port):
    """The same cloud with each render setting on its own. LAB"""
    cases = [('fx_plain', {}), ('fx_bloom', {'bloom-strength': 1.4}), ('fx_persistence', {'decay-time': 85}),
             ('fx_dof', {'dof-amount': 35}), ('fx_tracers', {'tracer-length': 90}),
             ('fx_fractal', {'fractal-enable': True})]
    for name, settings in cases:
        v = Viewer(b, port, LAB, 900, 900)
        fit(v, nn=8, zoom=1.0)
        look(v, 16, 0.7, 0.0)
        for k, val in settings.items():
            v.set(k, val)
        if name in ('fx_persistence', 'fx_tracers'):
            v.click('#btn-rotate')      # these show on motion
            v.click('#btn-play')
            v.settle(3000)
        v.hide_ui()
        v.settle(1500)
        v.shot(IMG / f'{name}.png')
        v.close()


def experimental(b, port):
    v = pbmc(b, port)
    v.set('color-by', 'cell_type')
    v.open_group('Experimental')
    v.open_sub('Reactivity')
    v.open_sub('Fractal Reflection')
    v.set('fractal-enable', True)
    v.settle()
    v.shot(IMG / 'experimental.png')
    v.shot(IMG / 'panel_experimental.png', clip=clip_of(v, '#panel'))
    v.close()


def controls(b, port):
    """The bottom bar, the corner buttons and the recording indicator."""
    v = pbmc(b, port)
    v.set('color-by', 'cell_type')
    v.shot(IMG / 'controls.png', clip=clip_of(v, '#controls', 10))
    v.shot(IMG / 'util_buttons.png', clip=clip_of(v, '#util-buttons', 10))
    v.set('rec-duration', 4)
    v.click('#btn-record')
    v.settle(800)
    v.shot(IMG / 'recording.png')
    v.close()


def storyboard(b, port):
    """Frames of the example storyboard (examples/pbmc3k_tour.json)."""
    v = Viewer(b, port, 'pbmc3k', 1000, 1000)
    board = json.loads((ROOT / 'examples' / 'pbmc3k_tour.json').read_text(encoding='utf-8'))
    plan = v.page.evaluate('j => window.storyboard.load(j)', board)
    for s in [s for s in plan['segments'] if s['kind'] == 'hold']:
        url = v.page.evaluate('t => window.storyboard.frame(t, "image/png")', s['start'] + s['dur'] * 0.3)
        out = IMG / f'sb_{s["step"] + 1}_{s["name"]}.png'
        Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).save(out)
        print(f'  manual/img/{out.name}')
    v.close()


def landing(b, port):
    page = b.new_page(viewport={'width': 1200, 'height': 700})
    page.goto(f'http://127.0.0.1:{port}/index.html')
    page.wait_for_selector('.card')
    page.wait_for_timeout(400)
    page.screenshot(path=str(IMG / 'landing.png'))
    print('  manual/img/landing.png')
    page.close()


def cover(b, port):
    """The manual's cover: tracers while the camera turns. LAB"""
    v = Viewer(b, port, LAB, 1000, 1400)
    fit(v, nn=8, zoom=1.15)
    look(v, 16, 0.7, 0.25)
    v.set('tracer-length', 60)
    v.click('#btn-morph')           # morph off: the turn alone draws the tracers
    v.click('#btn-rotate')
    v.click('#btn-play')
    v.settle(2500)
    v.hide_ui()
    v.settle(300)
    v.shot(IMG / 'cover.png')
    v.close()


SHOTS = [('cover', cover), ('stages', stage_tiles), ('banner', banner), ('gif', gif), ('interface', interface),
         ('features', features), ('effects', render_effects), ('experimental', experimental),
         ('controls', controls), ('storyboard', storyboard), ('landing', landing)]


def main():
    srv = serve()
    port = srv.server_address[1]
    from playwright.sync_api import sync_playwright
    print(f'lab dataset: {LAB}')
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel='msedge', headless=not A.headed,
                               args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'])
        for name, fn in SHOTS:
            if want(name):
                print(name)
                fn(b, port)
        b.close()
    srv.shutdown()


if __name__ == '__main__':
    main()
