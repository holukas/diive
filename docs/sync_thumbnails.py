"""
SYNC_THUMBNAILS: COMMIT EXAMPLE-GALLERY THUMBNAILS
==================================================

Copy the figures of a local gallery build into docs/_static/thumbs/ and point
every example at its own thumbnail.

Read the Docs builds with DIIVE_DOCS_GALLERY unset, so sphinx-gallery never runs
the examples and no figure is ever produced there: every card would show the same
stock placeholder. A committed image referenced by a per-example
`# sphinx_gallery_thumbnail_path` directive is used even when plot_gallery is
off, so generating the figures once locally and committing the thumbnails is what
gives the published gallery real pictures.

Usage (paths are resolved from this file, so the working directory does not matter)::

    uv run python docs/sync_thumbnails.py            # sync
    uv run python docs/sync_thumbnails.py --check    # report only, non-zero if out of sync

Part of the diive library: https://github.com/holukas/diive
"""

import argparse
import ast
import hashlib
import re
import sys
import tempfile
from pathlib import Path

THUMB_PREFIX = "sphx_glr_"
THUMB_SUFFIX = "_thumb.png"
DIRECTIVE_KEY = "sphinx_gallery_thumbnail_path"

REAL = "real"
PLACEHOLDER = "placeholder"
BROKEN = "broken"

# Splits on "\n" only, so a CRLF line keeps its own "\r" and every untouched line
# is reproduced byte for byte; line numbers stay aligned with ast's.
LINE_RE = re.compile(r"[^\n]*\n|[^\n]+$")


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def stock_thumbnail_hashes() -> dict:
    """Return {md5: kind} for sphinx-gallery's placeholder and broken-example images.

    Only used to put a name on a stock image; which thumbnails are stock at all is
    decided by hash grouping (see main), which needs no reference at all.

    Hashed at runtime instead of hardcoded: the stock images change between
    sphinx-gallery versions. They also cannot be hashed as they sit in the
    package - the build rescales them to thumbnail_size before writing, so the
    reference has to go through the installed version's own scaler to come out
    byte-identical. The size is sphinx-gallery's default because docs/conf.py
    does not override it.
    """
    try:
        import sphinx_gallery
        from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF
        from sphinx_gallery.utils import scale_image
    except ImportError:
        raise SystemExit("sphinx-gallery is required to classify thumbnails (uv sync).")

    static = Path(sphinx_gallery.__file__).parent / "_static"
    size = DEFAULT_GALLERY_CONF["thumbnail_size"]
    hashes = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, kind in (("no_image.png", PLACEHOLDER), ("broken_example.png", BROKEN)):
            src = static / filename
            if not src.is_file():
                continue
            hashes[_md5(src.read_bytes())] = kind  # unscaled, for the svg/gif copy path
            scaled = Path(tmpdir) / filename
            scale_image(str(src), str(scaled), *size)
            hashes[_md5(scaled.read_bytes())] = kind
    return hashes


def is_gallery_example(path: Path) -> bool:
    """Mirror the filename_pattern / ignore_pattern of docs/conf.py."""
    return path.suffix == ".py" and not path.name.startswith("_") and path.stem != "run_all_examples"


def find_examples(examples_dir: Path) -> tuple:
    """Return ({example name: source path}, [names claimed by more than one file])."""
    seen = {}
    for path in sorted(examples_dir.rglob("*.py")):
        if is_gallery_example(path):
            seen.setdefault(path.stem, []).append(path)
    found = {name: paths[0] for name, paths in seen.items() if len(paths) == 1}
    return found, sorted(name for name, paths in seen.items() if len(paths) > 1)


def find_generated(gallery_dir: Path) -> dict:
    """Return {example name: thumbnail path} from a built gallery tree.

    Nested categories (flux/lowres, flux/partitioning, preprocessing/qaqc, ...)
    each get their own images/thumb directory, hence the recursive search; the
    example name is unique across all of them and carries the mapping.
    """
    found = {}
    if not gallery_dir.is_dir():
        return found
    for png in sorted(gallery_dir.rglob(f"{THUMB_PREFIX}*{THUMB_SUFFIX}")):
        if png.parent.name == "thumb":
            found[png.name[len(THUMB_PREFIX):-len(THUMB_SUFFIX)]] = png
    return found


def directive_line(name: str) -> str:
    return f"# {DIRECTIVE_KEY} = '_static/thumbs/{name}.png'"


def _read_lines(path: Path) -> tuple:
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    return LINE_RE.findall(text), ("\r\n" if "\r\n" in text else "\n")


def _write_lines(path: Path, lines: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(lines))


def _insert_at(lines: list) -> int:
    """Line index after the module docstring, else before the first cell marker."""
    try:
        module = ast.parse("".join(lines))
    except SyntaxError:
        module = None
    if module and module.body:
        first = module.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            return first.value.end_lineno
    for i, line in enumerate(lines):
        if line.startswith("# %%"):
            return i
    return 0


def sync_directive(path: Path, name: str, want: bool, apply: bool):
    """Add, correct or drop the thumbnail directive. Returns the action, or None."""
    lines, newline = _read_lines(path)
    wanted = directive_line(name)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("#") and DIRECTIVE_KEY in stripped):
            continue
        if want:
            if stripped == wanted:
                return None
            lines[i] = wanted + line[len(line.rstrip("\r\n")):]
            action = "updated"
        else:
            del lines[i]
            # Leave no double blank line behind.
            if 0 < i < len(lines) and not lines[i - 1].strip() and not lines[i].strip():
                del lines[i]
            action = "removed"
        if apply:
            _write_lines(path, lines)
        return action

    if not want:
        return None

    at = _insert_at(lines)
    block = [wanted + newline]
    if at > 0 and lines[at - 1].strip():
        block.insert(0, newline)
    if at < len(lines) and lines[at].strip():
        block.append(newline)
    lines[at:at] = block
    if apply:
        _write_lines(path, lines)
    return "added"


def _names(label: str, names: list) -> None:
    if names:
        print(f"  {label}: {', '.join(names)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync sphinx-gallery thumbnails into docs/_static/thumbs/ and wire up the examples.")
    parser.add_argument("--check", "--dry-run", dest="check", action="store_true",
                        help="report what would change, change nothing, exit non-zero if out of sync")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="repository root (default: the checkout this script lives in)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    gallery_dir = root / "docs" / "auto_examples"
    thumbs_dir = root / "docs" / "_static" / "thumbs"
    examples_dir = root / "examples"
    apply = not args.check
    verb = "" if apply else "would be "

    if not examples_dir.is_dir():
        raise SystemExit(f"No examples directory at {examples_dir}")

    examples, ambiguous = find_examples(examples_dir)
    generated = find_generated(gallery_dir)

    # A stock image is byte-identical across every example that got it, while two
    # real figures essentially never collide, so grouping by hash finds the stock
    # images by itself - no reference image and no constant to keep up to date.
    by_hash, unmatched = {}, []
    for name, png in sorted(generated.items()):
        try:
            data = png.read_bytes()
        except OSError as ex:  # a build may be writing this tree right now
            unmatched.append(f"{name} (unreadable: {ex})")
            continue
        by_hash.setdefault(_md5(data), []).append((name, data))

    stock = stock_thumbnail_hashes() if by_hash else {}
    real, placeholders, broken, shared = {}, [], [], []
    for digest, members in by_hash.items():
        names = [name for name, _ in members]
        kind = stock.get(digest)
        if kind == PLACEHOLDER:
            placeholders.extend(names)
            continue
        if kind == BROKEN:
            broken.extend(names)
            continue
        # Real figures do collide occasionally (two examples plotting the same
        # thing), so a shared hash is kept and reported, never silently dropped.
        if len(members) > 1:
            shared.append(sorted(names))
        for name, data in members:
            if name in examples:
                real[name] = data
            else:
                unmatched.append(name)
    placeholders.sort()
    broken.sort()
    unmatched.sort()

    # Only real figures are committed: a committed placeholder would be
    # indistinguishable from content and would hide that the example draws nothing.
    copied, written = [], 0
    for name, data in sorted(real.items()):
        dest = thumbs_dir / f"{name}.png"
        if dest.is_file() and dest.read_bytes() == data:
            continue
        copied.append(name)
        written += len(data)
        if apply:
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    stale = []
    if thumbs_dir.is_dir():
        for png in sorted(thumbs_dir.glob("*.png")):
            if png.stem not in examples:
                stale.append(png.stem)
                if apply:
                    png.unlink()

    committed = {png.stem for png in thumbs_dir.glob("*.png")} if thumbs_dir.is_dir() else set()
    committed = (committed | set(copied)) - set(stale)

    actions = {"added": [], "updated": [], "removed": []}
    for name, path in sorted(examples.items()):
        action = sync_directive(path, name, name in committed, apply)
        if action:
            actions[action].append(name)

    print(f"Examples: {len(examples)} | generated thumbnails: {len(generated)} "
          f"(real {len(real)}, placeholder {len(placeholders)}, broken {len(broken)})")
    if not generated:
        print(f"  no built gallery under {gallery_dir} - run docs/build_docs.ps1 -Gallery first")
    _names("no figure produced, skipped", placeholders)
    _names("BROKEN, the example raised during the build", broken)
    for group in sorted(shared):
        _names("identical thumbnails, kept as real (check they are not a new stock image)", group)
    _names("no matching example source", unmatched)
    _names("ambiguous example name, skipped", ambiguous)
    print(f"Thumbnails: {len(copied)} {verb}written ({written:,} bytes), "
          f"{len(real) - len(copied)} unchanged, {len(stale)} stale {verb}removed")
    _names("written", copied)
    _names("removed", stale)
    print(f"Directives: {len(actions['added'])} {verb}added, {len(actions['updated'])} {verb}updated, "
          f"{len(actions['removed'])} {verb}removed")
    _names("added", actions["added"])
    _names("updated", actions["updated"])
    _names("removed", actions["removed"])

    out_of_sync = bool(copied or stale or any(actions.values()))
    if args.check and out_of_sync:
        print("Out of sync: run 'uv run python docs/sync_thumbnails.py'")
        return 1
    if broken or ambiguous:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
