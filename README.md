Various scripts to make graphs and/or dump out data about a user's untappd beer checkins. These are written for personal interest only so to-do list may never get ticked off. You will need an untappd data file, which requires being an untappd subscriber (or finding a friend with one).

    pip install -e .

`untappd.py` contains core functionality to load check-in data from a json file. The other python files in scripts/ make graphs or print some data.

## Scripts

The `scripts/` directory contains tools to plot and analyse Untappd check-in exports:

- `comment_lengths.py` - Plots comment length distributions across rating buckets.
- `consecutive_checkin_rating_correlation.py` - Heatmap and PCA ellipse analysis comparing ratings of consecutive check-ins within an 8-hour window.
- `cumulative_checkins.py` - Plots cumulative check-in counts over time (total, unique, repeat, and non-taster) with polynomial extrapolation.
- `daily_beer.py` - Prints the top 20 rated IPAs under 9% ABV checked in since January 2024.
- `detect_possible_duplicates.py` - Flags check-ins for the same beer recorded within 30 minutes of each other.
- `festival_home_country.py` - Bar charts showing the breakdown of brewery home countries across festivals.
- `generate_sample_data_source.py` - Generates mock Untappd JSON check-in data for testing.
- `high_abv_ratings.py` - Scatter plot of ratings versus ABV for beers with ABV of 12% or higher.
- `intoxication.py` - Models and plots estimated blood alcohol concentration over a specified date range using a simple absorption and clearance curve.
- `location_log.py` - Reconstructs travel timelines from check-in venue locations, plotting country stays and travel segments with country flags and city summaries.
- `overrated_and_underrated.py` - Lists beers with the highest positive and negative difference between personal rating and global Untappd rating.
- `overrated_and_underrated_breweries.py` - Computes and plots breweries whose beers you rate higher or lower than the global Untappd average.
- `overrated_and_underrated_by_ranking.py` - Compares beer rankings by personal rating against rankings by global Untappd rating.
- `probability_continue_drinking.py` - Plots probabilities of having another drink given current drink rating and session average rating.
- `rating_histogram_by_various.py` - Generates rating histograms and violin plots grouped by style category, hour, session position, ABV strength, date segment, brewery, or venue.
- `rating_histogram_over_time.py` - Animated histogram showing how rating distributions evolve across check-in history.
- `rating_vs_abv_by_category_wip.py` - Scatter plot of mean rating versus mean ABV grouped by festival and year.
- `rating_vs_abv_stats.py` - Plots mean and standard deviation of ratings across ABV bins, overall and broken down by style.
- `ratings_by_country.py` - Pie chart and violin plot of check-in counts and rating distributions by brewery country.
- `run_all.py` - Runs plotting scripts in batch and saves output figures to `scripts/out/`.
- `scatter_by_various.py` - Generates pairwise scatter plots for attributes including rating, global rating, ABV, sentiment score, comment length, and date.
- `scatter_plots_by_category.py` - Scatter plots comparing personal rating against global rating and ABV across top venues.
- `sentiments_bert.py` - Evaluates comment text using a RoBERTa GoEmotions transformer model and plots predicted emotion logits against check-in ratings.
- `sentiments_nltk_vader.py` - Evaluates comment text using NLTK VADER sentiment scoring and plots compound sentiment against check-in ratings.
- `statistical_analysis.py` - Tests rating distributions for normality using a discretized normal model and runs Monte Carlo simulations on festival subsets.
- `statistics_over_time_periods.py` - Computes rolling window statistics (ratings, ABV, uniqueness) over time and saves line plots.
- `style_frequency_over_time_periods.py` - Plots rolling check-in counts or percentages by beer style category over time.
- `style_ratings_by_year.py` - Line plot showing average rating trends per style category across calendar years.
- `taproom_brewery_vs_thirdparty.py` - Stacked bar chart comparing brewery-owned venue check-ins against third-party venues over time.
- `time_of_day_kdes.py` - Kernel density estimates showing check-in time of day distributions split by rating quartiles.
- `time_of_day_mean.py` - Plots mean rating and check-in count by hour of the day.
- `top_beers_normalized_by_abv.py` - Ranks top beers after normalising ratings against expected scores for their ABV bin.
- `top_beers_normalized_by_style_category.py` - Ranks top beers after normalising ratings against mean and standard deviation for their style category.
- `top_breweries.py` - Ranks breweries by combining their top beer ratings and overall average rating with configurable weighting.
- `top_checkins_normalized_by_time_window.py` - Ranks check-ins after normalising ratings against a moving window to account for scoring shifts over time.
- `top_festival_beers.py` - Prints top 5 rated beers for each festival attended.
- `top_rated_checkins.py` - Plots cumulative timeline curves for top rating thresholds (e.g. 4.25 and 4.5).
- `top_venue_repeats.py` - Lists most visited venues and the beers most frequently re-ordered at each.
- `unique_ratio_by_date.py` - Plots the moving average ratio of unique beers over time.
- `venues_by_time.py` - Pie charts of top venues visited on each day of the week.
- `venues_by_time_inverse.py` - Day-of-week distribution plots for each top venue.
- `word_cloud.py` - Builds a word cloud from check-in comments with words coloured by average rating.
- `world_map.py` - Geocodes brewery check-in locations and renders an interactive Folium map coloured by rating metrics.

The `scripts/slop/` directory contains experimental and scratch scripts.

# todos

- make it easy to run all scripts in batch and have them all save out their results

