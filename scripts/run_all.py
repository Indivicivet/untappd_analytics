from pathlib import Path
import matplotlib
import untappd

# Use non-interactive backend for batch generation
matplotlib.use("Agg")

import comment_lengths
import consecutive_checkin_rating_correlation
import cumulative_checkins
import festival_home_country
import high_abv_ratings
import intoxication
import probability_continue_drinking
import rating_histogram_by_various
import rating_vs_abv_stats
import ratings_by_country
import scatter_by_various
import statistics_over_time_periods
import style_frequency_over_time_periods
import style_ratings_by_year
import taproom_brewery_vs_thirdparty
import time_of_day_kdes
import time_of_day_mean
import top_beers_normalized_by_abv
import top_breweries
import top_rated_checkins
import unique_ratio_by_date
import venues_by_time
import venues_by_time_inverse


def run_all(out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkins...")
    checkins = untappd.load_latest_checkins()
    print(f"Loaded {len(checkins)} checkins.\n")

    tasks = [
        (
            "Comment lengths",
            lambda: comment_lengths.plot_comment_lengths(
                comment_lengths.get_checkins_by_rating(checkins),
                out_file=out_dir / "comment_lengths.png",
            ),
        ),
        (
            "Consecutive checkin correlation",
            lambda: consecutive_checkin_rating_correlation.plot_consecutive_checkin_rating_correlation(
                checkins,
                out_file=out_dir / "consecutive_checkin_rating_correlation.png",
            ),
        ),
        (
            "Cumulative checkins",
            lambda: cumulative_checkins.plot_cumulative_checkins(
                out_file=out_dir / "cumulative_checkins.png",
            ),
        ),
        (
            "Checkin rate",
            lambda: cumulative_checkins.plot_checkin_rate(
                out_file=out_dir / "checkin_rate.png",
            ),
        ),
        (
            "High ABV ratings",
            lambda: high_abv_ratings.plot_high_abv_ratings(
                checkins,
                out_file=out_dir / "high_abv_ratings.png",
            ),
        ),
        (
            "Intoxication curve",
            lambda: intoxication.plot_intoxication(
                checkins,
                out_file=out_dir / "intoxication.png",
            ),
        ),
        (
            "Unique ratio over time",
            lambda: unique_ratio_by_date.plot_unique_ratio_by_date(
                checkins,
                out_file=out_dir / "unique_ratio_by_date.png",
            ),
        ),
        (
            "Top rated checkins",
            lambda: top_rated_checkins.plot_top_rated_checkins(
                checkins,
                out_file=out_dir / "top_rated_checkins.png",
            ),
        ),
        (
            "Festival home countries",
            lambda: festival_home_country.plot_festival_home_country(
                checkins,
                out_file=out_dir / "festival_home_country.png",
            ),
        ),
        (
            "Ratings by country",
            lambda: ratings_by_country.country_pie_and_ratings(
                checkins,
                out_file=out_dir / "ratings_by_country.png",
            ),
        ),
        (
            "Ratings vs ABV stats",
            lambda: rating_vs_abv_stats.plot_all_rating_vs_abv(
                checkins,
                out_file=out_dir / "rating_vs_abv_stats.png",
            ),
        ),
        (
            "Style frequency over time",
            lambda: style_frequency_over_time_periods.plot_style_frequency_over_time_periods(
                checkins,
                out_file=out_dir / "style_frequency_over_time.png",
            ),
        ),
        (
            "Style ratings by year",
            lambda: style_ratings_by_year.plot_style_ratings_by_year(
                checkins,
                out_file=out_dir / "style_ratings_by_year.png",
            ),
        ),
        (
            "Taproom first party vs third party",
            lambda: taproom_brewery_vs_thirdparty.plot_taproom_brewery_vs_thirdparty(
                checkins,
                out_file=out_dir / "taproom_brewery_vs_thirdparty.png",
            ),
        ),
        (
            "Time of day KDEs",
            lambda: time_of_day_kdes.plot_time_of_day_kdes(
                checkins,
                out_file=out_dir / "time_of_day_kdes.png",
            ),
        ),
        (
            "Time of day mean",
            lambda: time_of_day_mean.show_average_rating_by_time(
                checkins,
                out_file=out_dir / "time_of_day_mean.png",
            ),
        ),
        (
            "Venues by time",
            lambda: venues_by_time.plot_venues_by_time(
                checkins,
                out_file=out_dir / "venues_by_time.png",
            ),
        ),
        (
            "Venues by time inverse",
            lambda: venues_by_time_inverse.plot_venues_by_time_inverse(
                checkins,
                out_file=out_dir / "venues_by_time_inverse.png",
            ),
        ),
        (
            "Drinking probabilities",
            lambda: probability_continue_drinking.plot_drinking_probabilities(
                checkins,
                out_file=out_dir / "drinking_probabilities.png",
            ),
        ),
        (
            "Top beers normalized by ABV",
            lambda: top_beers_normalized_by_abv.plot_abv_normalization(
                out_file=out_dir / "top_beers_normalized_by_abv.png",
            ),
        ),
        (
            "Rating histograms & violins by various",
            lambda: rating_histogram_by_various.save_various_plots(
                checkins,
                out_dir=out_dir,
                violin=False,
            ),
        ),
        (
            "Scatter plots by various",
            lambda: scatter_by_various.save_various_scatters(
                checkins,
                out_dir=out_dir,
            ),
        ),
    ]

    for name, fn in tasks:
        print(f"Running: {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  Warning: {name} encountered an error: {e}")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    run_all()
