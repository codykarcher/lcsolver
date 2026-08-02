#!/usr/bin/env python3
"""Serve the component test pages with caching turned off.

    python3 serve.py [port]

Plain ``python3 -m http.server`` is not usable for this. Browsers cache ES
modules aggressively, and it sends nothing to say otherwise, so editing a
component and reloading can leave the old module in place -- silently, with no
error, showing geometry that no longer exists in the source. That is a
miserable thing to debug, because every measurement you take of the code
disagrees with what is on screen.

Cache-busting query strings do not fix it either: `engines.js?v=1` and
`engines.js` are different modules to the browser, so the page would get one
copy of materials.js and the components another, and shared state -- the
material objects -- would silently split in two.

So: no-store on everything, and let the module graph stay consistent.
"""
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):          # quiet; one line per start
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
    root = Path(__file__).resolve().parent
    handler = partial(NoCacheHandler, directory=str(root))
    print(f"serving {root} on http://localhost:{port}  (no-store)")
    for page in sorted(root.glob("*_test.html")):
        print(f"    http://localhost:{port}/{page.name}")
    HTTPServer(("", port), handler).serve_forever()


if __name__ == "__main__":
    main()
