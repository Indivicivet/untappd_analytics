# todo :: can we do more interesting things with sentiment analysis? :)
# cf scatter_by_various.py which does scatters of (legacy) vader sentiments
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

import untappd

if not torch.cuda.is_available():
    raise RuntimeError("no CUDA :(")


def logit(v):
    clipped = np.clip(v, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


CIS = untappd.load_latest_checkins()[:1000]

MODEL_ID = "SamLowe/roberta-base-go_emotions"  # "standard robust small choice"
PIPELINE = transformers.pipeline(
    "text-classification",
    model=MODEL_ID,
    device=0,
    top_k=None,
)
emotion_scores = {}
total_scores = defaultdict(float)
for ci, analysis in zip(
    CIS,
    # must pass generator to pipeline for tqdm to work
    tqdm(PIPELINE((c.comment for c in CIS), batch_size=64), total=len(CIS)),
):
    emotion_scores[ci] = {d["label"]: d["score"] for d in analysis}
    for emotion, score in emotion_scores[ci].items():
        total_scores[emotion] += score

total_scores["neutral"] = -total_scores["neutral"]  # keep numerical value for ref
top_emotions = list(sorted(total_scores.items(), key=lambda t: t[1], reverse=True))

# for ci, emotions in sorted(emotion_scores.items()):
#     print(f"{emotions} ({c.rating}) | {c.beer} | {c.comment}")


cmap = LinearSegmentedColormap.from_list("rg", ["red", "gray", "green"])

fig, axes = plt.subplots(4, 5, figsize=(12.8, 7.2), constrained_layout=True)
print(top_emotions[: len(axes.flatten())])
for i, (emotion_name, _) in enumerate(top_emotions[: len(axes.flatten())]):
    ax = axes.flatten()[i]
    x = [c.rating for c in CIS]
    y = [logit(emotion_scores[c][emotion_name]) for c in CIS]
    x, y = np.array(x), np.array(y)
    ax.scatter(x, y, alpha=0.001, s=30)
    # "reduced major axis" 2D fit
    slope = np.sign(np.corrcoef(x, y)[0, 1]) * (np.std(y) / np.std(x))
    color = cmap(np.clip((slope + 3) / 7, 0, 1))
    ax.plot(
        x,
        slope * (x - x.mean()) + y.mean(),
        label=f"{slope:.2f}",
        color=color,
    )
    ax.set_facecolor((*color[:3], 0.05))
    ax.legend(fontsize=8)
    ax.set_title(emotion_name, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Rating", fontsize=8)
    ax.set_ylabel("Logit", fontsize=8)

fig.supxlabel("Overall Rating", fontsize=12)
fig.supylabel("Overall Logit Score", fontsize=12)
plt.show()
