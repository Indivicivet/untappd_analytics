from pathlib import Path

import matplotlib.pyplot as plt

import untappd
import untappd_utils


@untappd_utils.show_or_save_to_out_file
def scatter_things(
    dict_of_things,
    key_x,
    key_y,
    alpha=0.05,
    s=100,
):
    plt.figure(figsize=(12.8, 7.2))
    plt.scatter(
        dict_of_things[key_x],
        dict_of_things[key_y],
        alpha=alpha,
        s=s,
    )
    plt.xlabel(key_x)
    plt.ylabel(key_y)


def save_various_scatters(checkins, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).parent / "out"
    dict_of_things = {
        "ratings": [c.rating for c in checkins],
        "comment_length": [len(c.comment) for c in checkins],
    }
    for x, y, kwargs in [
        ["comment_length", "ratings", {}],
    ]:
        scatter_things(
            dict_of_things=dict_of_things,
            key_x=x,
            key_y=y,
            out_file=out_dir / f"scatter_{y}_vs_{x}.png",
        )


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    save_various_scatters(CHECKINS)

