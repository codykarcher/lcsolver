#!/usr/bin/env python3
"""Draw the D8 from a solve file.

    python3 show_d8.py                          # the example deck in decks/
    python3 show_d8.py ~/Desktop/my_solve.json  # any other solve
    python3 show_d8.py deck.json --port 8100 --no-browser

The page reads the deck and nothing between the solve and the drawing is
retyped, which is why the numbers in the readout agree with the solve to
machine precision rather than to however carefully they were copied.
"""
from deck_server import serve

if __name__ == "__main__":
    serve(page="d8_aircraft_test.html",
          deck_arg=None,
          default_deck="decks/b737_d8_solve.json",
          want_fins=2,                       # two fins, on the afterbody corners
          what="the D8")
