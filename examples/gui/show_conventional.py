#!/usr/bin/env python3
"""Draw the conventional tube-and-wing from a solve file.

    python3 show_conventional.py                          # the example deck
    python3 show_conventional.py ~/Desktop/my_solve.json  # any other solve
    python3 show_conventional.py deck.json --port 8100 --no-browser

The page reads the deck and nothing between the solve and the drawing is
retyped, which is why the numbers in the readout agree with the solve to
machine precision rather than to however carefully they were copied.
"""
from deck_server import serve

if __name__ == "__main__":
    serve(page="b737_test.html",
          deck_arg=None,
          default_deck="decks/b737_conventional_solve.json",
          want_fins=1,                       # one fin, on the tailcone
          what="the conventional aeroplane")
