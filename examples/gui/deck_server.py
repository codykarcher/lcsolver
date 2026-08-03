#!/usr/bin/env python3
"""Serve one drawing page against one solve file.

Shared by ``show_conventional.py`` and ``show_d8.py``, which are the same
program with a different page and a different expectation of the deck.

The deck is SERVED, not copied. It is mapped to a fixed URL and the page is
opened with ``?deck=`` pointing at that URL, so the file being drawn is the
file you named on the command line, wherever it lives -- rather than a copy of
it made at some earlier moment, which is the sort of thing that has you
measuring one aeroplane and looking at another.

Everything goes out ``no-store`` for the reason ``serve.py`` explains at
length: browsers cache ES modules hard, and a stale module shows geometry that
is not in the source any more, with no error anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Where the deck appears to the browser. Any path that cannot collide with a
#: real file in this directory will do; this one is obvious in the network tab.
DECK_URL = "/__deck__.json"


class DeckHandler(SimpleHTTPRequestHandler):
    """The gui directory, plus one file from wherever it actually is."""

    deck_path: Path = None                       # set by the partial, below

    def translate_path(self, path: str) -> str:
        # The base class strips the query and fragment before mapping, so an
        # exact compare is safe here.
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == DECK_URL:
            return str(self.deck_path)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:   # quiet; the summary is enough
        pass


def read_deck(path: Path) -> dict:
    """Load the solve, failing HERE rather than in the browser.

    A deck that will not parse, or that is not a solve at all, otherwise shows
    up as a red line in the page's readout with no clue as to which of the two
    it was -- so it is worth the twenty lines to say so on the terminal.
    """
    if not path.is_file():
        sys.exit(f"no such deck: {path}")
    try:
        deck = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{path}: not JSON -- {e}")
    if not isinstance(deck, dict):
        sys.exit(f"{path}: a solve is an object of named variables, not {type(deck).__name__}")
    if "n_vt" not in deck:
        sys.exit(f"{path}: does not look like a solve -- no 'n_vt' among its {len(deck)} keys")
    return deck


def check_configuration(deck: dict, path: Path, want_fins: int, what: str) -> None:
    """Warn if the deck is the other aeroplane.

    The two solves have the same 1300-odd variables and differ only in their
    values, so nothing structural tells them apart and the page will happily
    draw either. `n_vt` does: one fin on the tube-and-wing, two on the D8. This
    warns rather than refuses, because drawing a deck the other way round is a
    perfectly good thing to want to look at -- it just should not happen by
    accident.
    """
    fins = int(deck.get("n_vt", 0))
    if fins != want_fins:
        print(f"  ! {path.name} has n_vt = {fins}, and {what} expects {want_fins}.", flush=True)
        print("  ! Drawing it anyway -- if that was not deliberate, "
              "you want the other script.", flush=True)


def serve(page: str, deck_arg: str | None, default_deck: str,
          want_fins: int, what: str, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=f"Draw {what} from a solve file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"With no deck named, the example in decks/ is used:\n    {default_deck}")
    ap.add_argument("deck", nargs="?", default=deck_arg or str(HERE / default_deck),
                    help="path to a solve JSON")
    ap.add_argument("-p", "--port", type=int, default=8731, help="default 8731")
    ap.add_argument("-n", "--no-browser", action="store_true",
                    help="print the URL instead of opening it")
    args = ap.parse_args(argv)

    path = Path(args.deck).expanduser().resolve()
    deck = read_deck(path)
    check_configuration(deck, path, want_fins, what)

    handler = partial(DeckHandler, directory=str(HERE))
    DeckHandler.deck_path = path
    try:
        server = HTTPServer(("", args.port), handler)
    except OSError as e:
        sys.exit(f"cannot listen on port {args.port}: {e}\ntry: {sys.argv[0]} --port {args.port + 1}")

    url = f"http://localhost:{args.port}/{page}?deck={DECK_URL}"
    # Flushed, because this is the only output there is and the process then
    # blocks forever in serve_forever(). Redirected to a file, Python's block
    # buffering holds all three lines until the server is killed, which is the
    # one moment they are no use.
    print(f"{what} from {path}", flush=True)
    print(f"  status: {deck.get('_status', 'no status recorded')}", flush=True)
    print(f"  {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
