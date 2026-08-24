import argparse
import datetime
import io
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import numpy as np
import PIL.Image
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import untappd
import untappd_utils

# Ensure UTF-8 stdout encoding on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

FLAGS_CACHE_DIR = Path(__file__).resolve().parent / "out" / "flags"

COUNTRY_META = {
    "United Kingdom": {
        "name": "United Kingdom",
        "code": "gb",
        "flag": "🇬🇧",
        "color": "#235789",
    },
    "日本": {
        "name": "日本",
        "code": "jp",
        "flag": "🇯🇵",
        "color": "#d90429",
    },
    "中国": {
        "name": "中国",
        "code": "cn",
        "flag": "🇨🇳",
        "color": "#e85d04",
    },
    "臺灣": {
        "name": "臺灣",
        "code": "tw",
        "flag": "🇹🇼",
        "color": "#2a9d8f",
    },
    "Nederland": {
        "name": "Nederland",
        "code": "nl",
        "flag": "🇳🇱",
        "color": "#7209b7",
    },
    "Singapore": {
        "name": "Singapore",
        "code": "sg",
        "flag": "🇸🇬",
        "color": "#a0522d",
    },
    "Danmark": {
        "name": "Danmark",
        "code": "dk",
        "flag": "🇩🇰",
        "color": "#c77dff",
    },
    "Malta": {
        "name": "Malta",
        "code": "mt",
        "flag": "🇲🇹",
        "color": "#00b4d8",
    },
    "Suomi": {
        "name": "Suomi",
        "code": "fi",
        "flag": "🇫🇮",
        "color": "#0077b6",
    },
    "香港": {
        "name": "香港",
        "code": "hk",
        "flag": "🇭🇰",
        "color": "#e0a96d",
    },
    "ประเทศไทย": {
        "name": "ประเทศไทย",
        "code": "th",
        "flag": "🇹🇭",
        "color": "#e9c46a",
    },
}


def _get_font_for_text(text: str) -> fm.FontProperties:
    """
    Select appropriate font based on Unicode script content (Thai, CJK, Latin).
    """
    if any("\u0e00" <= ch <= "\u0e7f" for ch in text):
        return fm.FontProperties(family=["Leelawadee UI", "Tahoma", "sans-serif"])
    if any(ord(ch) > 0x2E80 for ch in text):
        return fm.FontProperties(
            family=["Microsoft YaHei", "SimHei", "Yu Gothic", "sans-serif"]
        )
    return fm.FontProperties(family=["Segoe UI", "DejaVu Sans", "sans-serif"])


def _get_flag_image(code: str, zoom: float = 0.20) -> Optional[OffsetImage]:
    FLAGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    flag_file = FLAGS_CACHE_DIR / f"{code.lower()}.png"
    if not flag_file.exists():
        url = f"https://flagcdn.com/w80/{code.lower()}.png"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp, open(
                flag_file, "wb"
            ) as f:
                f.write(resp.read())
        except Exception:
            return None
    try:
        img = PIL.Image.open(flag_file).convert("RGBA")
        return OffsetImage(img, zoom=zoom)
    except Exception:
        return None


def build_location_timeline(cis: list[untappd.Checkin]) -> list[dict]:
    """
    Parses checkins to reconstruct continuous geographical location segments.
    Filters virtual venues and infers country for checkins lacking venue data.
    """

    def clean_country(ci: untappd.Checkin) -> Optional[str]:
        if ci.venue and ci.venue.country:
            # Untappd at Home default artifact in NC, USA
            if ci.venue.country == "United States" and "Untappd at Home" in (
                ci.venue.name or ""
            ):
                return None
            return ci.venue.country
        return None

    known_indices = [i for i, ci in enumerate(cis) if clean_country(ci) is not None]

    records = []
    for i, ci in enumerate(cis):
        known = clean_country(ci)
        if known:
            raw_sub = ci.venue.state or ci.venue.city or ""
            records.append(
                {
                    "ci": ci,
                    "country": known,
                    "subregion": raw_sub,
                    "is_inferred": False,
                    "is_transition": False,
                }
            )
        else:
            prev_k = [k for k in known_indices if k < i]
            next_k = [k for k in known_indices if k > i]
            prev_c = clean_country(cis[prev_k[-1]]) if prev_k else "United Kingdom"
            next_c = clean_country(cis[next_k[0]]) if next_k else "United Kingdom"

            if prev_c == next_c:
                records.append(
                    {
                        "ci": ci,
                        "country": prev_c,
                        "subregion": "",
                        "is_inferred": True,
                        "is_transition": False,
                    }
                )
            else:
                records.append(
                    {
                        "ci": ci,
                        "country": f"{prev_c} -> {next_c}",
                        "prev_country": prev_c,
                        "next_country": next_c,
                        "subregion": "",
                        "is_inferred": True,
                        "is_transition": True,
                    }
                )

    # Consolidate continuous country segments
    segments = []
    curr_seg = [records[0]]
    for rec in records[1:]:
        if rec["country"] == curr_seg[-1]["country"]:
            curr_seg.append(rec)
        else:
            segments.append(curr_seg)
            curr_seg = [rec]
    segments.append(curr_seg)

    seg_dicts = []
    for s in segments:
        raw_c = s[0]["country"]
        is_trans = s[0]["is_transition"]
        s_dt = s[0]["ci"].datetime
        e_dt = s[-1]["ci"].datetime

        # Subregions visited in chronological order (unique)
        sub_list = []
        for x in s:
            sub = x["subregion"]
            if sub and sub not in sub_list:
                sub_list.append(sub)

        duration_days = max((e_dt - s_dt).total_seconds() / 86400.0, 0.0)

        seg_dicts.append(
            {
                "raw_country": raw_c,
                "is_transition": is_trans,
                "prev_country": s[0].get("prev_country"),
                "next_country": s[0].get("next_country"),
                "start_dt": s_dt,
                "end_dt": e_dt,
                "duration_days": duration_days,
                "checkins": s,
                "count": len(s),
                "venue_count": sum(1 for x in s if not x["is_inferred"]),
                "subregions": sub_list,
            }
        )
    return seg_dicts


def print_timeline(seg_dicts: list[dict]):
    """
    Prints a formatted geographical timeline log to console.
    """
    print("=" * 105)
    print("                                  GEOGRAPHICAL LOCATION TIMELINE")
    print("=" * 105)
    print(
        f"{'DATES':<23} | {'PLACE / DESTINATION':<28} | {'DAYS':>6} | {'CHECKINS':>10} | {'SUBREGIONS'}"
    )
    print("-" * 105)

    country_time = defaultdict(float)
    country_cis = defaultdict(int)

    for seg in seg_dicts:
        s_str = seg["start_dt"].strftime("%Y-%m-%d")
        e_str = seg["end_dt"].strftime("%Y-%m-%d")
        date_col = f"{s_str} .. {e_str}"
        days = seg["duration_days"]
        days_str = f"{days:5.1f}d" if days >= 0.1 else "<1d"
        cnt_str = f"{seg['count']:>4} ({seg['venue_count']:>3}v)"

        if seg["is_transition"]:
            p_name = COUNTRY_META.get(seg["prev_country"], {}).get(
                "name", seg["prev_country"]
            )
            n_name = COUNTRY_META.get(seg["next_country"], {}).get(
                "name", seg["next_country"]
            )
            place_col = f"✈️  {p_name} ➔ {n_name}"
            subs = "(transit / uncertain boundary)"
        else:
            meta = COUNTRY_META.get(
                seg["raw_country"],
                {"name": seg["raw_country"], "flag": "📍"},
            )
            place_col = f"{meta['flag']}  {meta['name']}"
            subs = ", ".join(seg["subregions"][:4]) if seg["subregions"] else ""
            country_time[meta["name"]] += days
            country_cis[meta["name"]] += seg["count"]

        print(
            f"{date_col:<23} | {place_col:<28} | {days_str:>6} | {cnt_str:>10} | {subs}"
        )

    print("=" * 105)
    print("\n--- SUMMARY OF VISITED COUNTRIES ---")
    sorted_countries = sorted(country_time.items(), key=lambda t: t[1], reverse=True)
    for c_name, t_days in sorted_countries:
        meta = next(
            (v for v in COUNTRY_META.values() if v["name"] == c_name),
            {"flag": "📍"},
        )
        print(
            f"  {meta['flag']}  {c_name:<20} : {t_days:>7.1f} days  |  {country_cis[c_name]:>5} check-ins"
        )
    print()


@untappd_utils.show_or_save_to_out_file
def plot_location_timeline(seg_dicts: list[dict]):
    """
    Renders a multi-row year-by-year visual timeline with color-coded country segments,
    flags, exact date ranges, duration indicators, and visited sub-regions.
    """
    min_year = min(s["start_dt"].year for s in seg_dicts)
    max_year = max(s["end_dt"].year for s in seg_dicts)
    start_year = min_year if min_year >= 2017 else 2017
    years = list(range(start_year, max_year + 1))
    n_rows = len(years)

    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.15 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f6f8fa")

    for row_idx, year in enumerate(years):
        ax = axes[row_idx]
        ax.set_facecolor("#ffffff")

        r_start = datetime.datetime(year, 1, 1)
        r_end = datetime.datetime(year, 12, 31, 23, 59, 59)
        ax.set_xlim(r_start, r_end)
        ax.set_ylim(-0.25, 2.95)
        ax.get_yaxis().set_visible(False)

        # Month gridlines
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=list(range(1, 13))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(axis="x", linestyle=":", alpha=0.5, color="#d0d0d0")

        # Year label on left margin
        ax.text(
            -0.035,
            0.35,
            str(year),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="#333333",
            va="center",
            ha="right",
        )

        row_segs = [
            s for s in seg_dicts if s["start_dt"] <= r_end and s["end_dt"] >= r_start
        ]

        # Non-UK segments that will have callout annotations
        non_uk_segs = [
            s
            for s in row_segs
            if s["raw_country"] != "United Kingdom" and not s["is_transition"]
        ]

        # Cluster non-UK segments to compute collision-free offsets
        clusters = []
        if non_uk_segs:
            curr_c = [non_uk_segs[0]]
            for s in non_uk_segs[1:]:
                prev_s = curr_c[-1]
                if (s["start_dt"] - prev_s["end_dt"]).days < 28:
                    curr_c.append(s)
                else:
                    clusters.append(curr_c)
                    curr_c = [s]
            clusters.append(curr_c)

        # Map each segment to its (x_shift_days, y_level)
        callout_positions = {}
        for c in clusters:
            if len(c) == 1:
                callout_positions[id(c[0])] = (0.0, 1.25)
            elif len(c) == 2:
                callout_positions[id(c[0])] = (-16.0, 1.45)
                callout_positions[id(c[1])] = (16.0, 0.85)
            elif len(c) >= 3:
                callout_positions[id(c[0])] = (-22.0, 1.55)
                callout_positions[id(c[1])] = (0.0, 0.82)
                callout_positions[id(c[2])] = (22.0, 1.55)
                for extra_idx, extra_seg in enumerate(c[3:], start=3):
                    callout_positions[id(extra_seg)] = (
                        26.0 * (extra_idx - 1),
                        0.85,
                    )

        for seg in row_segs:
            s_start = max(seg["start_dt"], r_start)
            s_end = min(seg["end_dt"], r_end)

            start_num = mdates.date2num(s_start)
            end_num = mdates.date2num(s_end)
            actual_width = max(end_num - start_num, 0.05)

            bar_height = 0.42
            bar_y = 0.08

            if seg["is_transition"]:
                rect = patches.FancyBboxPatch(
                    (start_num, bar_y),
                    max(actual_width, 1.0),
                    bar_height,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    facecolor="#e4e7eb",
                    edgecolor="#a0aec0",
                    linewidth=0.8,
                    hatch="//",
                    alpha=0.85,
                    zorder=3,
                )
                ax.add_patch(rect)
                continue

            meta = COUNTRY_META.get(
                seg["raw_country"],
                {
                    "name": seg["raw_country"],
                    "color": "#4a7c59",
                    "code": "un",
                },
            )
            color = meta["color"]
            name = meta["name"]
            code = meta.get("code")

            # Draw the visual bar on the timeline
            visual_bar_width = max(actual_width, 1.2)
            rect = patches.FancyBboxPatch(
                (start_num, bar_y),
                visual_bar_width,
                bar_height,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor="#1a202c",
                linewidth=1.1,
                alpha=0.95,
                zorder=3,
            )
            ax.add_patch(rect)

            mid_date = s_start + (s_end - s_start) / 2
            mid_num = mdates.date2num(mid_date)

            if seg["raw_country"] != "United Kingdom":
                x_shift_days, callout_y = callout_positions.get(id(seg), (0.0, 1.25))
                card_x_num = mid_num + x_shift_days
                flag_y = callout_y + 0.58

                # Draw a marker pin for short trips (< 4 days)
                if seg["duration_days"] < 4.0:
                    ax.plot(
                        [mid_num, mid_num],
                        [bar_y + bar_height, bar_y + bar_height + 0.15],
                        color=color,
                        lw=1.5,
                        zorder=4,
                    )
                    ax.scatter(
                        [mid_num],
                        [bar_y + bar_height + 0.15],
                        color=color,
                        s=25,
                        zorder=4,
                        edgecolor="#ffffff",
                        linewidth=0.8,
                    )

                # Embed country flag icon cleanly above card
                if code:
                    imagebox = _get_flag_image(code, zoom=0.20)
                    if imagebox:
                        ab = AnnotationBbox(
                            imagebox,
                            (card_x_num, flag_y),
                            frameon=True,
                            pad=0.08,
                            bboxprops=dict(
                                boxstyle="round,pad=0.08",
                                facecolor="white",
                                edgecolor="#cccccc",
                                lw=0.8,
                            ),
                            zorder=6,
                        )
                        ax.add_artist(ab)

                # Format duration string
                dur = seg["duration_days"]
                if dur < 1.0:
                    dur_str = "<1 day"
                elif abs(dur - round(dur)) < 0.15:
                    dur_str = f"{int(round(dur))} days" if round(dur) > 1 else "1 day"
                else:
                    dur_str = f"{dur:.1f} days"

                # Format dates
                if seg["start_dt"].strftime("%Y-%m-%d") == seg["end_dt"].strftime(
                    "%Y-%m-%d"
                ):
                    date_str = seg["start_dt"].strftime("%d %b")
                else:
                    date_str = f"{seg['start_dt'].strftime('%d %b')} – {seg['end_dt'].strftime('%d %b')}"

                subs_text = (
                    ", ".join(seg["subregions"][:2]) if seg["subregions"] else ""
                )
                lines = [f"{name}"]
                if subs_text:
                    lines.append(subs_text)
                lines.append(f"[{dur_str}]  {date_str} ({seg['count']} check-ins)")
                label_text = "\n".join(lines)

                text_font = _get_font_for_text(label_text)
                text_font.set_size(8)
                text_font.set_weight("medium")

                ax.annotate(
                    label_text,
                    xy=(mid_num, bar_y + bar_height),
                    xytext=(card_x_num, callout_y),
                    textcoords="data",
                    ha="center",
                    va="center",
                    fontproperties=text_font,
                    color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.32",
                        facecolor="#ffffff",
                        edgecolor=color,
                        alpha=0.96,
                        linewidth=1.4,
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        connectionstyle=(
                            "arc3,rad=0.1" if x_shift_days != 0 else "arc3,rad=0"
                        ),
                        color=color,
                        lw=1.1,
                    ),
                    zorder=5,
                )
            elif seg["raw_country"] == "United Kingdom" and (s_end - s_start).days > 30:
                ax.text(
                    mid_num,
                    bar_y + bar_height / 2,
                    "United Kingdom",
                    ha="center",
                    va="center",
                    color="#ffffff",
                    fontweight="bold",
                    fontsize=9,
                    zorder=4,
                )

    fig.suptitle(
        f"Geographical Travel Timeline from Untappd Check-ins ({start_year} – {max_year})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
        color="#1a1a1a",
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.03, hspace=0.48)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geographical location timeline from Untappd check-ins"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path to save figure (e.g. scripts/out/location_timeline.png)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the text timeline to console without plotting",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate the visual plot without printing text table",
    )
    args = parser.parse_args()

    checkins = untappd.load_latest_checkins(ignore_unrated=False)
    timeline_segments = build_location_timeline(checkins)

    if not args.plot_only:
        print_timeline(timeline_segments)

    if not args.print_only:
        plot_location_timeline(timeline_segments, out_file=args.out)
