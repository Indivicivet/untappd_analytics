from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import numpy as np
from scipy import stats

import untappd
import untappd_categorise

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def get_discretized_normal_probs(
    mu: float, std: float, unique_ratings: list[float]
) -> np.ndarray:
    """
    Computes the probabilities of each rating category from a normal distribution
    discretized by setting boundaries exactly halfway between adjacent observed rating values.
    """
    probs = []
    n = len(unique_ratings)
    if n == 1:
        return np.array([1.0])

    for i, r in enumerate(unique_ratings):
        if i == 0:
            mid = (unique_ratings[0] + unique_ratings[1]) / 2.0
            probs.append(stats.norm.cdf(mid, mu, std))
        elif i == n - 1:
            mid = (unique_ratings[n - 2] + unique_ratings[n - 1]) / 2.0
            probs.append(1.0 - stats.norm.cdf(mid, mu, std))
        else:
            mid_prev = (unique_ratings[i - 1] + r) / 2.0
            mid_next = (r + unique_ratings[i + 1]) / 2.0
            probs.append(
                stats.norm.cdf(mid_next, mu, std) - stats.norm.cdf(mid_prev, mu, std)
            )
    return np.array(probs)


def analyze_global_normality(ratings: np.ndarray, unique_ratings: list[float]):
    """
    Fits a discretized normal distribution to the global check-in ratings
    and runs a goodness-of-fit test to quantify how close it is to normal.
    """
    mu = np.mean(ratings)
    std = np.std(ratings, ddof=1)
    n = len(ratings)

    # Compute observed counts and expected counts under normal
    obs_counts_dict = Counter(ratings)
    obs_counts = np.array([obs_counts_dict.get(r, 0) for r in unique_ratings])

    # Expected probabilities
    exp_probs = get_discretized_normal_probs(mu, std, unique_ratings)
    exp_counts = exp_probs * n

    # Metrics to quantify deviation from normality
    mae = np.mean(np.abs(obs_counts / n - exp_probs))
    rmse = np.sqrt(np.mean((obs_counts / n - exp_probs) ** 2))

    # Skewness and Kurtosis
    skew = stats.skew(ratings)
    kurt = stats.kurtosis(ratings)  # Fisher's kurtosis (normal -> 0.0)

    # Chi-Squared Goodness of Fit
    # Subtract 2 degrees of freedom because mu and std are estimated from the sample
    chi2, p_val = stats.chisquare(obs_counts, f_exp=exp_counts, ddof=2)

    print("======================================================================")
    print("GLOBAL DISTRIBUTION NORMALITY ANALYSIS")
    print("======================================================================")
    print(f"Total check-ins: {n}")
    print(f"Global Mean:     {mu:.4f}")
    print(f"Global Std Dev:  {std:.4f}")
    print(f"Global Skewness: {skew:.4f} (Ideal normal: 0)")
    print(f"Global Kurtosis: {kurt:.4f} (Ideal excess kurtosis: 0)")
    print("\nGoodness-of-Fit (Discretized Gaussian):")
    print(f"  Mean Absolute Error (MAE) per bin: {mae:.5f}")
    print(f"  Root Mean Square Error (RMSE):      {rmse:.5f}")
    print(f"  Chi-Square Statistic:               {chi2:.4f}")
    print(f"  Chi-Square p-value:                 {p_val:.4g}")
    if p_val < 0.05:
        print("  Conclusion: We reject the null hypothesis of normality (p < 0.05).")
        print(
            "              The global rating distribution deviates significantly from a Gaussian."
        )
    else:
        print("  Conclusion: The global distribution is consistent with a Gaussian.")
    print("\nTop 5 Rating Bins (Observed vs Expected under Gaussian):")
    sorted_bins = sorted(
        zip(unique_ratings, obs_counts, exp_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"  {'Rating':<8} | {'Observed':<8} | {'Expected':<8}")
    print(f"  {'-'*29}")
    for r, obs, exp in sorted_bins[:5]:
        print(f"  {r:<8.2f} | {obs:<8d} | {exp:<8.2f}")
    print()


def analyze_subset(
    global_ratings: np.ndarray,
    subset_ratings: np.ndarray,
    unique_ratings: list[float],
    filter_name: str = "Subset",
):
    """
    Runs a series of statistical comparisons and hypothesis tests to determine
    whether the subset is consistent with the global background distribution (H0).
    """
    n_glob = len(global_ratings)
    n_sub = len(subset_ratings)

    if n_sub == 0:
        print(f"No check-ins found matching filter '{filter_name}'.")
        return

    mu_glob = np.mean(global_ratings)
    std_glob = np.std(global_ratings, ddof=1)

    mu_sub = np.mean(subset_ratings)
    std_sub = np.std(subset_ratings, ddof=1)

    # 1. Dynamic identification of top rating categories
    subset_counts = Counter(subset_ratings)
    observed_subset_ratings = sorted(subset_counts.keys())

    # R_max is the highest rating present in the subset
    r_max = observed_subset_ratings[-1]
    c_max = subset_counts[r_max]

    # Find the next higher rating in the global scale (usually r_max + 0.25)
    global_index = unique_ratings.index(r_max)
    if global_index < len(unique_ratings) - 1:
        r_next = unique_ratings[global_index + 1]
    else:
        r_next = r_max + 0.25  # Fallback increment

    # Count subset check-ins that are >= r_next
    c_next_plus = sum(count for r, count in subset_counts.items() if r >= r_next)

    print("======================================================================")
    print(f"SUBSET ANALYSIS: {filter_name}")
    print("======================================================================")
    print(f"Global Sample Size: {n_glob}  |  Subset Sample Size: {n_sub}")
    print(f"Global Mean:        {mu_glob:.4f}  |  Subset Mean:        {mu_sub:.4f}")
    print(f"Global Std Dev:     {std_glob:.4f}  |  Subset Std Dev:     {std_sub:.4f}")
    print()
    print("Top Rating Observations:")
    print(f"  Highest Rating in Subset (R_max):    {r_max:.2f}")
    print(f"  Count at R_max (C_max):              {c_max}")
    print(f"  Next rating increment (R_next):      {r_next:.2f}")
    print(f"  Count >= R_next in Subset:           {c_next_plus}")
    print()

    # 2. Hypothesis Testing: Is Subset consistent with Global?
    print("Hypothesis Testing (H0: Subset matches global background):")

    # t-test
    t_stat, t_p = stats.ttest_ind(global_ratings, subset_ratings, equal_var=False)
    print(
        f"  Welch's t-test:       t = {t_stat:.4f}, p-value = {t_p:.4g} "
        f"({'Reject H0' if t_p < 0.05 else 'Cannot reject H0'})"
    )

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(
        global_ratings, subset_ratings, alternative="two-sided"
    )
    print(
        f"  Mann-Whitney U test:  U = {u_stat:.1f}, p-value = {u_p:.4g} "
        f"({'Reject H0' if u_p < 0.05 else 'Cannot reject H0'})"
    )

    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.ks_2samp(global_ratings, subset_ratings)
    print(
        f"  Kolmogorov-Smirnov:   D = {ks_stat:.4f}, p-value = {ks_p:.4g} "
        f"({'Reject H0' if ks_p < 0.05 else 'Cannot reject H0'})"
    )

    # Monte Carlo Resampling (empirical p-values under H0)
    print("\nMonte Carlo Resampling (100,000 runs):")
    n_runs = 100000
    np.random.seed(42)
    mc_samples = np.random.choice(global_ratings, size=(n_runs, n_sub))

    # Mean rating resampling
    mc_means = np.mean(mc_samples, axis=1)
    p_mean = np.mean(mc_means >= mu_sub)
    print(f"  Prob of getting a mean rating >= {mu_sub:.4f} under H0: {p_mean:.5f}")

    # Top ratings resampling (joint probability of >= c_max of r_max AND 0 >= r_next)
    mc_c_max = np.sum(mc_samples == r_max, axis=1)
    mc_c_next = np.sum(mc_samples >= r_next, axis=1)
    p_top_anom = np.mean((mc_c_max >= c_max) & (mc_c_next == 0))
    print(
        f"  Prob of getting >= {c_max} beers rated {r_max:.2f} and 0 rated >= {r_next:.2f} under H0: {p_top_anom:.5f}"
    )

    if p_mean < 0.05 or p_top_anom < 0.05:
        print(
            "  Conclusion: There is strong evidence to reject H0. The subset's rating\n"
            "              statistics are NOT consistent with the global background."
        )
    else:
        print(
            "  Conclusion: There is NOT sufficient evidence to reject H0. The subset is\n"
            "              consistent with a random draw from the global background."
        )

    # 3. Discretized Gaussian Modeling of the Subset
    print("\n======================================================================")
    print(f"DISCRETIZED GAUSSIAN MODELING OF THE SUBSET")
    print("======================================================================")
    sub_probs = get_discretized_normal_probs(mu_sub, std_sub, unique_ratings)
    sub_obs = np.array([subset_counts.get(r, 0) for r in unique_ratings])
    sub_exp = sub_probs * n_sub

    # Goodness of fit on subset
    sub_chi2, sub_p = stats.chisquare(sub_obs, f_exp=sub_exp, ddof=2)
    print(f"Fitting N({mu_sub:.4f}, {std_sub:.4f}^2) to subset:")
    print(f"  Chi-Square Statistic: {sub_chi2:.4f}, p-value = {sub_p:.4g}")
    if sub_p < 0.05:
        print(
            "  Conclusion: Even for the subset, we reject the Gaussian fit (p < 0.05)."
        )
    else:
        print(
            "  Conclusion: The subset's ratings are well modeled by a Gaussian distribution\n"
            f"              with mean = {mu_sub:.4f} and std dev = {std_sub:.4f} (p = {sub_p:.4f} > 0.05)."
        )

    # Probabilities under the fitted normal distribution
    r_max_index = unique_ratings.index(r_max)
    # Bin boundaries for R_max and R_next
    if r_max_index == 0:
        mid_max = (unique_ratings[0] + unique_ratings[1]) / 2.0
        p_r_max = stats.norm.cdf(mid_max, mu_sub, std_sub)
    else:
        mid_prev = (unique_ratings[r_max_index - 1] + r_max) / 2.0
        mid_next = (r_max + unique_ratings[r_max_index + 1]) / 2.0
        p_r_max = stats.norm.cdf(mid_next, mu_sub, std_sub) - stats.norm.cdf(
            mid_prev, mu_sub, std_sub
        )

    p_r_next_plus = 1.0 - stats.norm.cdf(mid_next, mu_sub, std_sub)

    print(
        f"\nTheoretical probabilities for a single beer under N({mu_sub:.4f}, {std_sub:.4f}^2):"
    )
    print(f"  P(Rating == {r_max:.2f}):      {p_r_max:.5f}")
    print(f"  P(Rating >= {r_next:.2f}):      {p_r_next_plus:.5f}")

    # Probability in N trials
    p_binom_max = 1.0 - stats.binom.cdf(c_max - 1, n_sub, p_r_max)
    p_binom_next = (1.0 - p_r_next_plus) ** n_sub

    # Let's run a simulation of the fitted normal to compute the joint and conditional probabilities
    sim_draws = np.random.normal(mu_sub, std_sub, size=(n_runs, n_sub))
    sim_counts_max = np.sum((sim_draws > mid_prev) & (sim_draws <= mid_next), axis=1)
    sim_counts_next = np.sum(sim_draws > mid_next, axis=1)

    p_sim_max = np.mean(sim_counts_max >= c_max)
    p_sim_next = np.mean(sim_counts_next == 0)
    p_sim_both = np.mean((sim_counts_max >= c_max) & (sim_counts_next == 0))
    p_sim_cond = p_sim_both / p_sim_max if p_sim_max > 0 else 0.0

    print(
        f"\nIn a sample of {n_sub} beers from this distribution (Monte Carlo simulation):"
    )
    print(
        f"  Prob of getting >= {c_max} ratings of {r_max:.2f}:             {p_binom_max:.5f} (sim: {p_sim_max:.5f})"
    )
    print(
        f"  Prob of getting 0 ratings of >= {r_next:.2f}:               {p_binom_next:.5f} (sim: {p_sim_next:.5f})"
    )
    print(f"  Joint Prob of BOTH:                                    {p_sim_both:.5f}")
    print(
        f"  Conditional Prob of 0 ratings >= {r_next:.2f} GIVEN >= {c_max} of {r_max:.2f}:  {p_sim_cond:.5f}"
    )

    print("\nFinal Interpretation:")
    if p_sim_cond >= 0.05:
        print(
            f"  Given that you drank at least {c_max} beers rated {r_max:.2f}, there is a {p_sim_cond*100:.1f}% chance\n"
            f"  of drinking 0 beers rated >= {r_next:.2f}. This is very common and not anomalous.\n"
            f"  Thus, it is highly feasible that you really did drink {c_max} {r_max:.2f} beers and no {r_next:.2f}s."
        )
    else:
        print(
            f"  Given that you drank at least {c_max} beers rated {r_max:.2f}, there is only a {p_sim_cond*100:.1f}% chance\n"
            f"  of drinking 0 beers rated >= {r_next:.2f}. This is statistically anomalous (p < 0.05)."
        )


if __name__ == "__main__":
    # Load all rated check-ins
    checkins = untappd.load_latest_checkins()
    global_ratings = np.array([c.rating for c in checkins if c.rating is not None])
    unique_ratings = sorted(list(set(global_ratings)))

    # Global analysis
    analyze_global_normality(global_ratings, unique_ratings)

    # Filter to CAMRA Beer Festival 2026
    def camra_2026_filter(c):
        return (
            untappd_categorise.festival_with_year(c)
            == "Cambridge CAMRA Beer Festival\n(2026)"
        )

    camra_ratings = np.array(
        [c.rating for c in checkins if camra_2026_filter(c) and c.rating is not None]
    )

    # Subset analysis for CAMRA 2026
    analyze_subset(
        global_ratings,
        camra_ratings,
        unique_ratings,
        filter_name="Cambridge CAMRA Beer Festival (2026)",
    )
