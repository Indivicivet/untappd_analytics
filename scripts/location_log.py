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
    "대한민국": {
        "name": "대한민국",
        "code": "kr",
        "flag": "🇰🇷",
        "color": "#1d3557",
    },
    "South Korea": {
        "name": "대한민국",
        "code": "kr",
        "flag": "🇰🇷",
        "color": "#1d3557",
    },
}


def _get_font_for_text(text: str) -> fm.FontProperties:
    """
    Select appropriate font based on Unicode script content (Hangul, Thai, CJK, Latin).
    """
    return fm.FontProperties(
        family=[
            "Microsoft JhengHei",
            "Microsoft YaHei",
            "SimHei",
            "Malgun Gothic",
            "Leelawadee UI",
            "Yu Gothic",
            "Segoe UI",
            "DejaVu Sans",
        ]
    )


def _get_flag_image(code: str, zoom: float = 0.17) -> Optional[OffsetImage]:
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
    Renders a compact, unified multi-row visual timeline with shared vertical month gridlines,
    large readable fonts, collision-free multi-destination tour cards, flags,
    exact date ranges, duration indicators, and visited sub-regions.
    """
    min_year = min(s["start_dt"].year for s in seg_dicts)
    max_year = max(s["end_dt"].year for s in seg_dicts)
    start_year = min_year if min_year >= 2017 else 2017
    years = list(range(start_year, max_year + 1))
    n_rows = len(years)

    base_ref_year = 2024
    ref_start = datetime.datetime(base_ref_year, 1, 1)
    ref_end = datetime.datetime(base_ref_year, 12, 31, 23, 59, 59)

    def to_ref_dt(dt: datetime.datetime) -> datetime.datetime:
        doy = dt.timetuple().tm_yday
        is_leap = (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0)
        if not is_leap and doy > 59:
            doy += 1
        sec = dt.hour * 3600 + dt.minute * 60 + dt.second
        return datetime.datetime(base_ref_year, 1, 1) + datetime.timedelta(
            days=doy - 1, seconds=sec
        )

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(18, 1.15 * n_rows + 0.6),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    if n_rows == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f6f8fa")

    for row_idx, year in enumerate(years):
        ax = axes[row_idx]
        ax.set_facecolor("#ffffff")

        r_start = datetime.datetime(year, 1, 1)
        r_end = datetime.datetime(year, 12, 31, 23, 59, 59)

        ax.set_xlim(ref_start, ref_end)
        ax.set_ylim(-0.12, 2.25)
        ax.get_yaxis().set_visible(False)

        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=list(range(1, 13))))
        ax.grid(axis="x", linestyle="--", alpha=0.45, color="#b0b8c4", zorder=1)

        if row_idx == 0:
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(axis="x", labelsize=11, pad=3, length=3, color="#888888")
        elif row_idx == n_rows - 1:
            ax.xaxis.tick_bottom()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(axis="x", labelsize=11, pad=3, length=3, color="#888888")
        else:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                labeltop=False,
            )

        ax.text(
            -0.035,
            0.18,
            str(year),
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color="#333333",
            va="center",
            ha="right",
        )

        row_segs = [
            s for s in seg_dicts if s["start_dt"] <= r_end and s["end_dt"] >= r_start
        ]

        # Draw all track bars
        for seg in row_segs:
            s_start = max(seg["start_dt"], r_start)
            s_end = min(seg["end_dt"], r_end)

            ref_s_start = to_ref_dt(s_start)
            ref_s_end = to_ref_dt(s_end)

            start_num = mdates.date2num(ref_s_start)
            end_num = mdates.date2num(ref_s_end)
            actual_width = max(end_num - start_num, 0.05)

            bar_height = 0.32
            bar_y = 0.03

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

            mid_date = ref_s_start + (ref_s_end - ref_s_start) / 2
            mid_num = mdates.date2num(mid_date)

            # Short trip marker pin
            if seg["raw_country"] != "United Kingdom" and seg["duration_days"] < 4.0:
                ax.plot(
                    [mid_num, mid_num],
                    [bar_y + bar_height, bar_y + bar_height + 0.08],
                    color=color,
                    lw=1.5,
                    zorder=4,
                )
                ax.scatter(
                    [mid_num],
                    [bar_y + bar_height + 0.08],
                    color=color,
                    s=22,
                    zorder=4,
                    edgecolor="#ffffff",
                    linewidth=0.8,
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
                    fontsize=9.5,
                    zorder=4,
                )

        # Group non-UK segments into Tours (if gap between trips is < 5 days)
        non_uk_segs = [
            s
            for s in row_segs
            if s["raw_country"] != "United Kingdom" and not s["is_transition"]
        ]
        tours = []
        if non_uk_segs:
            curr_tour = [non_uk_segs[0]]
            for s in non_uk_segs[1:]:
                prev_s = curr_tour[-1]
                if (s["start_dt"] - prev_s["end_dt"]).total_seconds() / 86400.0 < 5.0:
                    curr_tour.append(s)
                else:
                    tours.append(curr_tour)
                    curr_tour = [s]
            tours.append(curr_tour)

        # Separate adjacent tour cards if centers are close horizontally
        tour_offsets = {}
        for t_idx, tour in enumerate(tours):
            t_start = to_ref_dt(tour[0]["start_dt"])
            t_end = to_ref_dt(tour[-1]["end_dt"])
            t_mid = mdates.date2num(t_start + (t_end - t_start) / 2)

            x_shift = 0.0
            y_shift = 0.0
            if t_idx > 0:
                prev_t = tours[t_idx - 1]
                p_start = to_ref_dt(prev_t[0]["start_dt"])
                p_end = to_ref_dt(prev_t[-1]["end_dt"])
                p_mid = mdates.date2num(p_start + (p_end - p_start) / 2)

                diff_days = t_mid - p_mid
                if diff_days < 72.0:
                    shift_val = max((72.0 - diff_days) / 2.0, 10.0)
                    prev_x, prev_y = tour_offsets.get(t_idx - 1, (0.0, 0.0))
                    tour_offsets[t_idx - 1] = (prev_x - shift_val, prev_y)
                    x_shift = shift_val
            tour_offsets[t_idx] = (x_shift, y_shift)

        for t_idx, tour in enumerate(tours):
            tour_start_dt = tour[0]["start_dt"]
            tour_end_dt = tour[-1]["end_dt"]
            ref_tour_start = to_ref_dt(tour_start_dt)
            ref_tour_end = to_ref_dt(tour_end_dt)

            raw_mid_num = mdates.date2num(
                ref_tour_start + (ref_tour_end - ref_tour_start) / 2
            )
            x_shift, y_shift = tour_offsets.get(t_idx, (0.0, 0.0))
            mid_num = raw_mid_num + x_shift

            total_days = max(
                (tour_end_dt - tour_start_dt).total_seconds() / 86400.0, 0.0
            )
            total_cis = sum(s["count"] for s in tour)

            def fmt_d(d):
                if d < 1.0:
                    return "<1d"
                elif abs(d - round(d)) < 0.15:
                    return f"{int(round(d))}d"
                return f"{d:.1f}d"

            if len(tour) == 1:
                # Single destination trip card
                seg = tour[0]
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

                dur = seg["duration_days"]
                dur_str = f"[{fmt_d(dur)}]"

                if seg["start_dt"].strftime("%Y-%m-%d") == seg["end_dt"].strftime(
                    "%Y-%m-%d"
                ):
                    date_str = seg["start_dt"].strftime("%d %b")
                else:
                    date_str = f"{seg['start_dt'].strftime('%d %b')} – {seg['end_dt'].strftime('%d %b')}"

                subs_text = (
                    ", ".join(seg["subregions"][:2]) if seg["subregions"] else ""
                )
                title_line = f"{name} ({subs_text})" if subs_text else f"{name}"
                lines = [
                    title_line,
                    f"{dur_str}  {date_str} ({seg['count']} check-ins)",
                ]
                card_text = "\n".join(lines)

                text_font = _get_font_for_text(card_text)
                text_font.set_size(9.0)
                text_font.set_weight("medium")

                callout_y = 0.88 + y_shift
                flag_y = callout_y + 0.46

                if code:
                    imagebox = _get_flag_image(code, zoom=0.18)
                    if imagebox:
                        ab = AnnotationBbox(
                            imagebox,
                            (mid_num, flag_y),
                            frameon=True,
                            pad=0.06,
                            bboxprops=dict(
                                boxstyle="round,pad=0.06",
                                facecolor="white",
                                edgecolor="#cccccc",
                                lw=0.8,
                            ),
                            zorder=6,
                        )
                        ax.add_artist(ab)

                ax.annotate(
                    card_text,
                    xy=(raw_mid_num, bar_y + bar_height),
                    xytext=(mid_num, callout_y),
                    textcoords="data",
                    ha="center",
                    va="center",
                    fontproperties=text_font,
                    color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.28",
                        facecolor="#ffffff",
                        edgecolor=color,
                        alpha=0.96,
                        linewidth=1.4,
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        connectionstyle=(
                            "arc3,rad=0.1" if x_shift != 0 else "arc3,rad=0"
                        ),
                        color=color,
                        lw=1.1,
                    ),
                    zorder=5,
                )
            else:
                # Multi-destination Tour Card: colored by main destination (highest duration)
                main_seg = max(tour, key=lambda s: s["duration_days"])
                main_color = COUNTRY_META.get(main_seg["raw_country"], {}).get(
                    "color", "#4a7c59"
                )

                tour_header = " -> ".join(
                    COUNTRY_META.get(s["raw_country"], {}).get("name", s["raw_country"])
                    for s in tour
                )

                stop_lines = []
                for s in tour:
                    c_name = COUNTRY_META.get(s["raw_country"], {}).get(
                        "name", s["raw_country"]
                    )
                    s_subs = ", ".join(s["subregions"][:1]) if s["subregions"] else ""
                    s_dt_str = (
                        s["start_dt"].strftime("%d %b")
                        if s["start_dt"].strftime("%Y-%m-%d")
                        == s["end_dt"].strftime("%Y-%m-%d")
                        else f"{s['start_dt'].strftime('%d')}–{s['end_dt'].strftime('%d %b')}"
                    )

                    loc_desc = f"{c_name} ({s_subs})" if s_subs else c_name
                    stop_lines.append(
                        f"• {loc_desc}: {s_dt_str} [{fmt_d(s['duration_days'])}, {s['count']} ci]"
                    )

                lines = [
                    f"{tour_header}  [{fmt_d(total_days)}, {total_cis} total check-ins]",
                    *stop_lines,
                ]
                card_text = "\n".join(lines)

                text_font = _get_font_for_text(card_text)
                text_font.set_size(8.5)
                text_font.set_weight("medium")

                callout_y = 0.78 + 0.07 * len(lines) + y_shift
                tour_flag_y = callout_y + 0.085 * len(lines) + 0.36

                # Include all flags in visits in sequential order, including duplicates
                all_tour_codes = [
                    COUNTRY_META.get(s["raw_country"], {}).get("code")
                    for s in tour
                    if COUNTRY_META.get(s["raw_country"], {}).get("code")
                ]

                n_flags = len(all_tour_codes)
                flag_spacing_days = 7.0
                flag_start_x = mid_num - ((n_flags - 1) * flag_spacing_days) / 2.0

                for f_idx, f_code in enumerate(all_tour_codes):
                    imagebox = _get_flag_image(f_code, zoom=0.16)
                    if imagebox:
                        ab = AnnotationBbox(
                            imagebox,
                            (
                                flag_start_x + f_idx * flag_spacing_days,
                                tour_flag_y,
                            ),
                            frameon=True,
                            pad=0.05,
                            bboxprops=dict(
                                boxstyle="round,pad=0.05",
                                facecolor="white",
                                edgecolor="#cccccc",
                                lw=0.7,
                            ),
                            zorder=6,
                        )
                        ax.add_artist(ab)

                ax.annotate(
                    card_text,
                    xy=(raw_mid_num, bar_y + bar_height),
                    xytext=(mid_num, callout_y),
                    textcoords="data",
                    ha="center",
                    va="center",
                    fontproperties=text_font,
                    color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.32",
                        facecolor="#ffffff",
                        edgecolor=main_color,
                        alpha=0.96,
                        linewidth=1.4,
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        connectionstyle=(
                            "arc3,rad=0.1" if x_shift != 0 else "arc3,rad=0"
                        ),
                        color=main_color,
                        lw=1.1,
                    ),
                    zorder=5,
                )

    fig.suptitle(
        f"Geographical Travel Timeline from Untappd Check-ins ({start_year} – {max_year})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
        color="#1a1a1a",
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.950, bottom=0.040)


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
