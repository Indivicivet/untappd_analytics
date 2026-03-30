# todo :: can we do more interesting things with sentiment analysis? :)
# cf scatter_by_various.py which does scatters of (legacy) vader sentiments
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import transformers
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("no CUDA :(")

import untappd

CIS = untappd.load_latest_checkins()[:50]

MODEL_ID = "SamLowe/roberta-base-go_emotions"  # "standard robust small choice"
PIPELINE = transformers.pipeline(
    "text-classification", model=MODEL_ID, device=0, top_k=None,
)
comments = [c.comment for c in CIS]
total_scores = defaultdict(float)
for ci, analysis in zip(tqdm(CIS), PIPELINE(comments, batch_size=64)):
    ci._emotion_scores = {d["label"] : d["score"] for d in analysis}
    for emotion, score in ci._emotion_scores.items():
        total_scores[emotion] += score

print(total_scores)

for c in sorted(
    CIS,
    key=lambda x: max(x._emotion_scores.values()), reverse=True
):
    print(f"{c._emotion_scores} ({c.rating}) | {c.beer} | {c.comment}")

# plt.figure(figsize=(12.8, 7.2))
# plt.scatter(
#     [c.rating for c in CIS],
#     [c._compound_score for c in CIS],
#     alpha=0.05,
#     s=100,
# )
# plt.show()
