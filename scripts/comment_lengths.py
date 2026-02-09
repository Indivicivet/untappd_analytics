
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

import untappd


def get_checkins_by_rating(checkins):
    grouped = collections.defaultdict(list)
    for c in checkins:
        if c.rating is None:
            continue
        if c.rating < 3:
            key = "< 3"
        elif c.rating <= 3:
            key = "3"
        elif c.rating <= 3.25:
            key = "3.25"
        elif c.rating <= 3.5:
            key = "3.5"
        elif c.rating <= 3.75:
            key = "3.75"
        else:
            key = "4+"
        grouped[key].append(c)
    return grouped


def plot_comment_lengths(
    checkins_by_group,
    out_file=None,
    title="Comment Length Distribution by Rating",
):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Sort keys to make legend logical
    # Custom sort order
    order = ["< 3", "3", "3.25", "3.5", "3.75 - 4", "4.25+", "other"]
    sorted_keys = sorted(
        checkins_by_group.keys(),
        key=lambda k: order.index(k) if k in order else 999
    )

    for label in sorted_keys:
        checkins = checkins_by_group[label]
        if not checkins:
            continue
            
        lengths = [len(c.comment) for c in checkins]
        
        # Calculate frequencies for 0 to 140
        # We can go up to max length found, or fixed 140
        max_len = 140
        counts = collections.Counter(lengths)
        x_vals = np.arange(max_len + 1)
        y_vals = np.array([counts[x] for x in x_vals])
        
        # Normalize
        total = len(lengths)
        if total == 0:
            continue
        y_density = y_vals / total
        
        # Smooth the data
        y_smooth = gaussian_filter1d(y_density, sigma=5)
        
        plt.plot(
            x_vals, 
            y_smooth, 
            label=f"{label} (n={total})",
            linewidth=2,
            alpha=0.8
        )

    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Comment Length (characters)")
    plt.ylabel("Density (Log Scale)")
    plt.legend()
    plt.xlim(-1, 141)
    
    # Add some minor gridlines for log scale readability if needed
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    if out_file:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    checkins = untappd.load_latest_checkins()
    grouped = get_checkins_by_rating(checkins)
    plot_comment_lengths(
        grouped, 
        out_file=Path(__file__).parent / "out" / "comment_lengths_by_rating.png"
    )
