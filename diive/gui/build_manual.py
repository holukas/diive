"""
BUILD_MANUAL: render the GUI user manual to styled HTML
=======================================================

Converts ``diive/gui/MANUAL.md`` (the source of truth) into the polished,
self-contained ``diive/gui/MANUAL.html`` that **Help ▸ User manual** opens.

Run it after editing the Markdown so the two never drift::

    python -m diive.gui.build_manual        # or: python diive/gui/build_manual.py

No third-party dependencies: this is a small converter tuned to the Markdown
subset the manual actually uses (ATX headings, fenced code, blockquotes, nested
unordered / flat ordered lists, inline bold / italic / code / links). It also
adds three presentation touches the plain Markdown can't express — a gradient
hero from the title block, ``Menu ▸ Tab`` headings split into a coloured chip +
title, keyboard chords wrapped in ``<kbd>``, and the variable-list tag-kind
bullet rendered as colour pills.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import html
import re
from pathlib import Path

HERE = Path(__file__).parent
MD_PATH = HERE / "MANUAL.md"
HTML_PATH = HERE / "MANUAL.html"

# Variable-kind colour pills (mirror the GUI's classify_variable colours).
_PILLS = [
    ("nee", "NEE / FC"), ("gpp", "GPP"), ("reco", "Reco"), ("le", "LE / ET"),
    ("rad", "radiation · SW_IN · PPFD · PAR · LW"), ("ta", "TA"),
    ("vpd", "VPD"), ("swc", "SWC"), ("new", "✦ NEW"),
]
_PILL_ROW = (
    '<div class="pillrow">'
    + "".join(f'<span class="pill {c}">{html.escape(t)}</span>' for c, t in _PILLS)
    + "</div>"
)

# Keyboard chords: a named modifier/key, optionally joined by '+' to more.
# Single-character keys are matched uppercase-only so "Ctrl+click" (a lowercase
# word) isn't mistaken for the chord "Ctrl+c"; real chords use Ctrl+S, Ctrl+R, …
_KBD_RE = re.compile(
    r"(?<![\w/])((?:Ctrl|Shift|Alt|Cmd|Enter|Tab|Esc)(?:\+(?:Ctrl|Shift|Alt|Cmd|Enter|Tab|Esc|[A-Z0-9]))*)"
)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_BARE_URL_RE = re.compile(r"(?<![\"'>=])(https?://[^\s<)]+)")


def _kbdify(s: str) -> str:
    def rep(m: re.Match) -> str:
        return "+".join(f"<kbd>{p}</kbd>" for p in m.group(1).split("+"))
    return _KBD_RE.sub(rep, s)


def inline(text: str, *, autolink: bool = False) -> str:
    """Convert inline Markdown to HTML. Code spans are protected first so their
    contents are never touched by the other rules."""
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(html.escape(m.group(1)))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = _LINK_RE.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    if autolink:
        text = _BARE_URL_RE.sub(lambda m: f'<a href="{m.group(1)}">{m.group(1)}</a>', text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    text = _kbdify(text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{spans[int(m.group(1))]}</code>", text)
    return text


def _render_code(code: list[str]) -> str:
    """Escape code lines and tint a trailing shell comment. The comment is found
    on the raw line (a '#' at line start or after whitespace) *before* escaping,
    so a '#' inside an escaped entity like &#x27; is never mistaken for one."""
    out = []
    for line in code:
        m = re.search(r"(?:(?<=\s)|^)(#.*)$", line)
        if m:
            out.append(html.escape(line[:m.start(1)])
                       + f'<span class="cmt">{html.escape(m.group(1))}</span>')
        else:
            out.append(html.escape(line))
    return "\n".join(out)


def _slug(text: str, seen: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "", text.lower()) or "section"
    slug, n = base, 2
    while slug in seen:
        slug, n = f"{base}{n}", n + 1
    seen.add(slug)
    return slug


def _build_list(block: list[str]) -> str:
    """Render a block of (possibly nested) list lines to HTML."""
    items: list[list] = []  # [indent, 'ul'|'ol', text]
    item_re = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
    for line in block:
        m = item_re.match(line)
        if m:
            items.append([len(m.group(1)), "ol" if m.group(2)[0].isdigit() else "ul", m.group(3)])
        elif items:  # wrapped continuation of the current item
            items[-1][2] += " " + line.strip()

    out: list[str] = []
    stack: list[tuple[int, str]] = []
    for indent, typ, text in items:
        while stack and indent < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or indent > stack[-1][0]:
            stack.append((indent, typ))
            out.append(f"<{typ}>")
        else:  # sibling at the same level
            out.append("</li>")
        body = inline(text)
        if "Tag pills" in text:
            body += _PILL_ROW
        out.append(f"<li>{body}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def convert_body(lines: list[str]) -> tuple[str, list[tuple[int, str, str]]]:
    """Convert the body Markdown to HTML, returning (html, toc) where toc is a
    list of (level, anchor_id, label)."""
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    i, n = 0, len(lines)
    item_re = re.compile(r"^\s*([-*]|\d+\.)\s+")

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # fenced code
        if line.lstrip().startswith("```"):
            i += 1
            code: list[str] = []
            while i < n and not lines[i].lstrip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence
            out.append(f"<pre><code>{_render_code(code)}</code></pre>")
            continue

        # heading
        m = re.match(r"^(#{2,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            raw = m.group(2).strip()
            chip, title = "", raw
            if "▸" in raw:
                chip, title = (p.strip() for p in raw.split("▸", 1))
            label = re.sub(r"\s*\(.*\)$", "", title)  # drop trailing parenthetical for the TOC
            anchor = _slug(label, seen)
            chip_html = f' <span class="menu-path">{html.escape(chip)}</span>' if chip else ""
            out.append(f'<h{level} id="{anchor}">{inline(title)}{chip_html}</h{level}>')
            if level in (2, 3):
                toc.append((level, anchor, label))
            i += 1
            continue

        # horizontal rule -> section divider, skip
        if re.match(r"^---+\s*$", line):
            i += 1
            continue

        # blockquote -> note callout
        if line.lstrip().startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].lstrip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f'<div class="note">{inline(" ".join(quote))}</div>')
            continue

        # list block
        if item_re.match(line):
            block: list[str] = []
            while i < n and lines[i].strip() and (item_re.match(lines[i]) or lines[i].startswith(" ")):
                block.append(lines[i])
                i += 1
            out.append(_build_list(block))
            continue

        # paragraph
        para: list[str] = [line]
        i += 1
        while i < n and lines[i].strip() and not lines[i].lstrip().startswith(("```", "#", ">")) \
                and not re.match(r"^---+\s*$", lines[i]) and not item_re.match(lines[i]):
            para.append(lines[i])
            i += 1
        text = " ".join(p.strip() for p in para)
        if text.lstrip("*").startswith("Part of the diive library"):
            continue  # rendered in the footer instead
        out.append(f"<p>{inline(text)}</p>")

    return "\n".join(out), toc


def _render_toc(toc: list[tuple[int, str, str]]) -> str:
    links = []
    for level, anchor, label in toc:
        cls = ' class="sub"' if level == 3 else ""
        links.append(f'    <a{cls} href="#{anchor}">{html.escape(label)}</a>')
    return "\n".join(links)


CSS = """
  :root {
    color-scheme: light;
    --ink:#1b2733; --ink-soft:#4a5b6b; --ink-faint:#7c8a98;
    --line:#e4e9ee; --line-soft:#eef2f5; --canvas:#ffffff; --panel:#f7f9fb;
    --accent:#2f7d8f; --accent-2:#2563a8; --accent-soft:#e7f1f4;
    --code-bg:#f3f6f8; --code-ink:#14506b; --kbd-bg:#fbfcfd;
    --shadow:0 1px 2px rgba(27,39,51,.05),0 8px 24px rgba(27,39,51,.06);
    --radius:14px; --maxw:1180px;
    --pill-nee:#2e7d32; --pill-gpp:#1565c0; --pill-reco:#c62828;
    --pill-le:#6a1b9a; --pill-rad:#ef6c00; --pill-ta:#d84315;
    --pill-vpd:#00838f; --pill-swc:#5d4037; --pill-new:#d81b60;
  }
  /* Dark theme is opt-in via the toggle (data-theme="dark" on <html>), not the
     OS setting — the manual defaults to light. */
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ink:#e6edf3; --ink-soft:#aebccb; --ink-faint:#7d8b9c;
    --line:#243140; --line-soft:#1d2733; --canvas:#0f161e; --panel:#151e28;
    --accent:#4db6c9; --accent-2:#5a9be0; --accent-soft:#16323a;
    --code-bg:#16202b; --code-ink:#7fd3e8; --kbd-bg:#1a2531;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);
  }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; scroll-padding-top:84px; }
  body {
    margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink); background:var(--canvas); line-height:1.62; font-size:16px;
    -webkit-font-smoothing:antialiased;
  }
  .hero {
    background:linear-gradient(135deg,#1f5d8c 0%,#2f7d8f 55%,#36a3a0 100%);
    color:#fff; padding:56px 24px 64px; position:relative; overflow:hidden;
  }
  .hero-inner { max-width:var(--maxw); margin:0 auto; }
  .hero .eyebrow { text-transform:uppercase; letter-spacing:.22em; font-size:12px; font-weight:600; opacity:.85; margin:0 0 14px; }
  .hero h1 { margin:0 0 14px; font-size:clamp(30px,5vw,46px); font-weight:700; letter-spacing:-.02em; line-height:1.1; }
  .hero p { margin:0; font-size:18px; opacity:.92; max-width:620px; }
  .wip {
    display:inline-flex; align-items:center; gap:8px; margin-top:22px;
    background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28);
    color:#fff; padding:8px 14px; border-radius:999px; font-size:13.5px; font-weight:500; backdrop-filter:blur(4px);
  }
  .layout { max-width:var(--maxw); margin:0 auto; padding:0 24px; display:grid; grid-template-columns:264px minmax(0,1fr); gap:48px; }
  nav.toc { position:sticky; top:0; align-self:start; height:100vh; overflow-y:auto; padding:28px 8px 40px 0; margin-top:-28px; }
  nav.toc .toc-title { font-size:11px; text-transform:uppercase; letter-spacing:.18em; color:var(--ink-faint); font-weight:700; margin:18px 0 8px 12px; }
  nav.toc a { display:block; color:var(--ink-soft); text-decoration:none; padding:5px 12px; border-radius:8px; font-size:13.5px; border-left:2px solid transparent; line-height:1.4; }
  nav.toc a.sub { padding-left:24px; font-size:13px; color:var(--ink-faint); }
  nav.toc a:hover { background:var(--panel); color:var(--ink); }
  nav.toc a.active { color:var(--accent); background:var(--accent-soft); border-left-color:var(--accent); font-weight:600; }
  main { padding:44px 0 80px; min-width:0; }
  h2 { font-size:26px; letter-spacing:-.01em; margin:56px 0 4px; padding-bottom:10px; border-bottom:1px solid var(--line); }
  h2:first-of-type { margin-top:8px; }
  h3 { font-size:19px; margin:38px 0 6px; letter-spacing:-.01em; display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
  h3 .menu-path { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:var(--accent); background:var(--accent-soft); padding:3px 8px; border-radius:6px; white-space:nowrap; }
  p, li { color:var(--ink); }
  a { color:var(--accent-2); }
  strong { font-weight:650; color:var(--ink); }
  ul, ol { padding-left:22px; }
  li { margin:5px 0; }
  li::marker { color:var(--accent); }
  ol li::marker { color:var(--ink-faint); font-weight:600; }
  code { font-family:"SF Mono",ui-monospace,"Cascadia Code",Consolas,monospace; background:var(--code-bg); color:var(--code-ink); padding:1.5px 6px; border-radius:5px; font-size:.86em; }
  pre { background:var(--code-bg); border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px; overflow-x:auto; box-shadow:var(--shadow); }
  pre code { background:none; padding:0; color:var(--ink); font-size:14px; line-height:1.7; }
  pre .cmt { color:var(--ink-faint); }
  kbd { font-family:inherit; font-size:12.5px; font-weight:600; background:var(--kbd-bg); border:1px solid var(--line); border-bottom-width:2px; border-radius:6px; padding:2px 7px; color:var(--ink); white-space:nowrap; }
  .note { border-left:4px solid var(--accent); background:var(--accent-soft); padding:14px 18px; border-radius:0 10px 10px 0; margin:18px 0; color:var(--ink-soft); font-size:15px; }
  .note strong { color:var(--ink); }
  .pillrow { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 4px; }
  .pill { display:inline-flex; align-items:center; gap:6px; padding:3px 11px; border-radius:999px; color:#fff; font-size:12.5px; font-weight:600; }
  .pill.nee{background:var(--pill-nee)} .pill.gpp{background:var(--pill-gpp)} .pill.reco{background:var(--pill-reco)}
  .pill.le{background:var(--pill-le)} .pill.rad{background:var(--pill-rad)} .pill.ta{background:var(--pill-ta)}
  .pill.vpd{background:var(--pill-vpd)} .pill.swc{background:var(--pill-swc)} .pill.new{background:var(--pill-new)}
  footer { border-top:1px solid var(--line); margin-top:64px; padding-top:24px; color:var(--ink-faint); font-size:14px; }
  footer a { color:var(--accent); }
  .theme-toggle {
    position:fixed; top:18px; right:18px; z-index:70; width:42px; height:42px;
    display:inline-flex; align-items:center; justify-content:center;
    border-radius:999px; border:1px solid rgba(255,255,255,.35);
    background:rgba(255,255,255,.16); color:#fff; font-size:18px; line-height:1;
    cursor:pointer; backdrop-filter:blur(4px); transition:background .15s ease, transform .15s ease;
  }
  .theme-toggle:hover { background:rgba(255,255,255,.28); transform:translateY(-1px); }
  /* Once scrolled past the hero the button sits over the page, so theme it. */
  .theme-toggle.scrolled { background:var(--panel); color:var(--ink); border-color:var(--line); box-shadow:var(--shadow); }
  .theme-toggle.scrolled:hover { background:var(--accent-soft); }
  .navtoggle { display:none; }
  @media (max-width:900px) {
    .layout { grid-template-columns:1fr; gap:0; }
    nav.toc { position:fixed; top:0; left:0; bottom:0; width:290px; height:100vh; background:var(--canvas); z-index:50; box-shadow:var(--shadow); transform:translateX(-110%); transition:transform .25s ease; padding:24px 16px 40px; margin:0; border-right:1px solid var(--line); }
    nav.toc.open { transform:translateX(0); }
    .navtoggle { display:inline-flex; align-items:center; gap:8px; position:fixed; top:14px; left:14px; z-index:60; background:var(--accent); color:#fff; border:none; border-radius:999px; padding:10px 16px; font-size:14px; font-weight:600; box-shadow:var(--shadow); cursor:pointer; }
    main { padding:28px 0 64px; }
    .hero { padding-top:70px; }
    .scrim { position:fixed; inset:0; background:rgba(0,0,0,.35); z-index:40; display:none; }
    .scrim.show { display:block; }
  }
"""

SCRIPT = """
  // Theme toggle (light default, dark opt-in, persisted).
  const root = document.documentElement;
  const TKEY = 'diive-manual-theme';
  const tbtn = document.getElementById('themetoggle');
  function isDark(){ return root.getAttribute('data-theme') === 'dark'; }
  function syncToggle(){
    const dark = isDark();
    tbtn.textContent = dark ? '☀' : '☾';
    tbtn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }
  syncToggle();
  tbtn.addEventListener('click', () => {
    if (isDark()) { root.removeAttribute('data-theme'); localStorage.setItem(TKEY, 'light'); }
    else { root.setAttribute('data-theme', 'dark'); localStorage.setItem(TKEY, 'dark'); }
    syncToggle();
  });
  // The button starts over the coloured hero; once scrolled past it, theme it
  // against the page so it stays legible.
  const hero = document.querySelector('.hero');
  function placeToggle(){ tbtn.classList.toggle('scrolled', window.scrollY > (hero ? hero.offsetHeight - 40 : 200)); }
  placeToggle();
  window.addEventListener('scroll', placeToggle, { passive: true });

  const toc = document.getElementById('toc');
  const toggle = document.getElementById('navtoggle');
  const scrim = document.getElementById('scrim');
  function closeNav(){ toc.classList.remove('open'); scrim.classList.remove('show'); }
  toggle.addEventListener('click', () => { toc.classList.toggle('open'); scrim.classList.toggle('show'); });
  scrim.addEventListener('click', closeNav);
  toc.addEventListener('click', e => { if (e.target.tagName === 'A') closeNav(); });
  const links = Array.from(toc.querySelectorAll('a'));
  const byId = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const targets = Array.from(document.querySelectorAll('main h2[id], main h3[id]'));
  let current = null;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        const a = byId.get(en.target.id);
        if (a && a !== current) {
          if (current) current.classList.remove('active');
          a.classList.add('active'); current = a;
          const r = a.getBoundingClientRect();
          if (r.top < 80 || r.bottom > window.innerHeight - 20) a.scrollIntoView({ block: 'nearest' });
        }
      }
    });
  }, { rootMargin: '-72px 0px -70% 0px', threshold: 0 });
  targets.forEach(t => obs.observe(t));
"""


def build(md_text: str) -> str:
    lines = md_text.replace("\r\n", "\n").split("\n")

    # --- front matter: H1 title + first paragraph (subtitle) + WIP blockquote ---
    title, subtitle = "diive GUI — User Manual", ""
    wip_lines: list[str] = []
    body_start = 0
    for idx, line in enumerate(lines):
        if line.startswith("## "):
            body_start = idx
            break
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.lstrip().startswith(">"):
            wip_lines.append(re.sub(r"^\s*>\s?", "", line))
        elif line.strip() and not line.startswith("#") and not re.match(r"^---+\s*$", line):
            if not subtitle:
                subtitle = line.strip()
    wip = " ".join(wip_lines)

    body_html, toc = convert_body(lines[body_start:])
    toc_html = _render_toc(toc)
    wip_html = inline(wip) if wip else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script>try{{if(localStorage.getItem('diive-manual-theme')==='dark')document.documentElement.setAttribute('data-theme','dark');}}catch(e){{}}</script>
<style>{CSS}</style>
</head>
<body>

<button class="theme-toggle" id="themetoggle" aria-label="Toggle dark theme">☾</button>

<header class="hero">
  <div class="hero-inner">
    <p class="eyebrow">diive · desktop application</p>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(subtitle)}</p>
    {f'<span class="wip">{wip_html}</span>' if wip_html else ''}
  </div>
</header>

<button class="navtoggle" id="navtoggle" aria-label="Toggle contents">☰ Contents</button>
<div class="scrim" id="scrim"></div>

<div class="layout">
  <nav class="toc" id="toc">
    <div class="toc-title">Contents</div>
{toc_html}
  </nav>

  <main>
{body_html}
    <footer>
      Part of the diive library — <a href="https://github.com/holukas/diive">github.com/holukas/diive</a>
    </footer>
  </main>
</div>

<script>{SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    HTML_PATH.write_text(build(md_text), encoding="utf-8")
    print(f"Wrote {HTML_PATH} ({HTML_PATH.stat().st_size:,} bytes) from {MD_PATH.name}")


if __name__ == "__main__":
    main()
