from __future__ import annotations

from pathlib import Path
import sys

# Ensure slop directory is in sys.path
SLOP_DIR = Path(__file__).resolve().parent
if str(SLOP_DIR) not in sys.path:
    sys.path.insert(0, str(SLOP_DIR))

from top_breweries_normalized import run_ranking


if __name__ == "__main__":
    run_ranking(
        normalize_mode="abv",
        top_breweries=20,
        top_beers=5,
        min_checkins_movers=5,
        save_plot=True,
    )
