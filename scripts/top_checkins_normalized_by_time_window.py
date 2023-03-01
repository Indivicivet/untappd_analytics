import statistics

import untappd
import untappd_utils

CHECKINS = untappd.load_latest_checkins()

CHECKIN_COUNT_WINDOW = 300
assert CHECKIN_COUNT_WINDOW % 2 == 0  # laziness :)


all_mean, all_std = untappd_utils.mean_and_std(CHECKINS)

# calculate statistics to normalize against
checkins_this_block = CHECKINS[:CHECKIN_COUNT_WINDOW]
per_block_stats = (
    [untappd_utils.mean_and_std(checkins_this_block)]
    * (CHECKIN_COUNT_WINDOW // 2)
)
for ci_new in CHECKINS[CHECKIN_COUNT_WINDOW:]:
    checkins_this_block.pop(0)
    checkins_this_block.append(ci_new)
    per_block_stats.append(untappd_utils.mean_and_std(checkins_this_block))
per_block_stats += [per_block_stats[-1]] * (CHECKIN_COUNT_WINDOW // 2)

assert len(per_block_stats) == len(CHECKINS)

for ci, (block_mean, block_std) in zip(CHECKINS, per_block_stats):
    ci._normalized_rating = (
        all_mean + all_std * (ci.rating - block_mean) / block_std
    )

print(f"beers normalized by ratings from the nearest {CHECKIN_COUNT_WINDOW} checkins")
print("i.e. 4.5 ratings at the times I was least likely to... :)")
print()
cis_sorted = sorted(CHECKINS, key=lambda ci: ci._normalized_rating, reverse=True)
for i, ci in enumerate(cis_sorted[:20]):
    print(
        f"#{i + 1}, rating {ci.rating} normalized {ci._normalized_rating:.3f}"
        f" at {ci.datetime}"
    )
    print(ci.beer)
    print()
