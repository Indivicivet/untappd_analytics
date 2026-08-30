import datetime
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import untappd
import untappd_utils

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MAX_GAP_HOURS = 6.0


def compute_sessions(
    checkins: Sequence[untappd.Checkin],
    max_gap_hours: float = MAX_GAP_HOURS,
) -> list[list[untappd.Checkin]]:
    """Segment check-ins into sessions based on maximum gap threshold."""
    sessions: list[list[untappd.Checkin]] = []
    current: list[untappd.Checkin] = []

    for c in checkins:
        if c.datetime is None or c.rating is None:
            continue
        if not current:
            current.append(c)
            continue
        prev = current[-1]
        gap = c.datetime - prev.datetime  # type: ignore[operator]
        if gap <= datetime.timedelta(hours=max_gap_hours):
            current.append(c)
        else:
            sessions.append(current)
            current = [c]
    if current:
        sessions.append(current)
    return sessions


def drink_units(ci: untappd.Checkin) -> float:
    """
    Assumed alcohol units for a checkin based on ABV and serving size assumptions:
    - 0.05 * ABV for tasters
    - 0.33 * ABV for purchased bottles/cans
    - 0.15 * ABV for standard draft / venue servings
    """
    abv = ci.beer.abv if ci.beer.abv is not None else 0.0
    vol = (
        0.05
        if ci.serving_type and ci.serving_type.lower() == "taster"
        else 0.33 if ci.purchase_venue else 0.15
    )
    return abv * vol


def get_circadian_hour(dt: datetime.datetime) -> float:
    """
    Circadian drinking hour where drinking day starts at 06:00 AM.
    Post-midnight hours (00:00 to 05:59) map to 24.0 to 29.99.
    """
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return h if h >= 6.0 else h + 24.0


def build_hazard_dataset(
    sessions: list[list[untappd.Checkin]],
) -> dict:
    """
    Builds a discrete-time hazard dataset where each checkin k in a session is an observation:
    - Event Y = 1 if it is the terminal drink of the session, Y = 0 if another drink followed.
    - Features: drink index k, rating R_k, rating delta (R_k - R_{k-1}), cumulative units,
      circadian hour of day, and weekend indicator (Fri/Sat vs Sun-Thu).
    """
    k_list = []
    rating_list = []
    rating_delta_list = []
    cum_units_list = []
    hour_list = []
    weekend_list = []
    terminal_list = []

    for session in sessions:
        cum_units = 0.0
        session_len = len(session)
        for idx, c in enumerate(session):
            k = idx + 1
            is_terminal = 1.0 if idx == session_len - 1 else 0.0
            r_k = float(c.rating)  # type: ignore[arg-type]
            r_delta = (
                float(c.rating - session[idx - 1].rating)  # type: ignore[operator]
                if idx > 0
                else 0.0
            )
            cum_units += drink_units(c)
            circ_hour = get_circadian_hour(c.datetime)  # type: ignore[arg-type]
            is_weekend = (
                1.0 if c.datetime.weekday() in (4, 5) else 0.0  # type: ignore[union-attr]
            )

            k_list.append(float(k))
            rating_list.append(r_k)
            rating_delta_list.append(r_delta)
            cum_units_list.append(cum_units)
            hour_list.append(circ_hour)
            weekend_list.append(is_weekend)
            terminal_list.append(is_terminal)

    return {
        "k": np.array(k_list),
        "rating": np.array(rating_list),
        "rating_delta": np.array(rating_delta_list),
        "cum_units": np.array(cum_units_list),
        "hour": np.array(hour_list),
        "is_weekend": np.array(weekend_list),
        "terminal": np.array(terminal_list),
    }


def fit_logistic_hazard(
    hazard_data: dict,
) -> dict:
    """
    Fits discrete-time logistic hazard model:
      logit P(Terminate at k) = beta_0 + sum(beta_j * X_j)
    via Iteratively Reweighted Least Squares (IRLS).
    """
    feature_names = [
        "Intercept",
        "Drink Index in Session (k)",
        "Current Drink Rating (R_k)",
        "Rating Delta (R_k - R_{k-1})",
        "Cumulative Alcohol Units",
        "Hour of Day (Circadian)",
        "Weekend (Fri/Sat vs Sun-Thu)",
    ]

    n_obs = len(hazard_data["terminal"])
    X = np.column_stack(
        [
            np.ones(n_obs),
            hazard_data["k"],
            hazard_data["rating"],
            hazard_data["rating_delta"],
            hazard_data["cum_units"],
            hazard_data["hour"],
            hazard_data["is_weekend"],
        ]
    )
    y = hazard_data["terminal"]

    # Fit logistic regression via Newton-Raphson / IRLS
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        W = p * (1.0 - p)
        grad = X.T @ (y - p)
        H = -(X.T * W) @ X
        step = np.linalg.solve(-H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break

    cov = np.linalg.inv(-H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p_vals = 2.0 * stats.norm.sf(np.abs(z))
    odds_ratios = np.exp(beta)
    ci_lower_beta = beta - 1.96 * se
    ci_upper_beta = beta + 1.96 * se
    ci_lower_or = np.exp(ci_lower_beta)
    ci_upper_or = np.exp(ci_upper_beta)

    # Goodness-of-fit metrics
    p_null = np.mean(y)
    ll_null = np.sum(y * np.log(p_null) + (1.0 - y) * np.log(1.0 - p_null))
    p_fitted = np.clip(1.0 / (1.0 + np.exp(-X @ beta)), 1e-15, 1.0 - 1e-15)
    ll_model = np.sum(y * np.log(p_fitted) + (1.0 - y) * np.log(1.0 - p_fitted))
    pseudo_r2 = 1.0 - (ll_model / ll_null)

    return {
        "feature_names": feature_names,
        "beta": beta,
        "se": se,
        "z": z,
        "p_values": p_vals,
        "odds_ratios": odds_ratios,
        "ci_lower_or": ci_lower_or,
        "ci_upper_or": ci_upper_or,
        "n_obs": n_obs,
        "n_events": int(np.sum(y)),
        "ll_null": ll_null,
        "ll_model": ll_model,
        "pseudo_r2": pseudo_r2,
    }


def print_hazard_summary(results: dict) -> None:
    """Prints the regression table and formal statistical interpretation."""
    print("=" * 95)
    print("DISCRETE-TIME LOGISTIC HAZARD REGRESSION FOR SESSION TERMINATION")
    print("=" * 95)
    print(f"Total check-in observations (N): {results['n_obs']}")
    print(f"Session termination events (Y=1): {results['n_events']}")
    print(f"Log-Likelihood (Null Model):      {results['ll_null']:.2f}")
    print(f"Log-Likelihood (Fitted Model):    {results['ll_model']:.2f}")
    print(f"McFadden's Pseudo R-squared:      {results['pseudo_r2']:.4f}")
    print("-" * 95)
    print(
        f"{'Feature':<32} | {'Coef':<9} | {'StdErr':<8} | {'z':<7} | "
        f"{'p-value':<10} | {'Odds Ratio':<11} | {'95% CI (OR)':<20}"
    )
    print("-" * 95)
    for name, b, s, z_stat, p_v, or_val, ci_l, ci_u in zip(
        results["feature_names"],
        results["beta"],
        results["se"],
        results["z"],
        results["p_values"],
        results["odds_ratios"],
        results["ci_lower_or"],
        results["ci_upper_or"],
    ):
        p_str = f"{p_v:.4e}" if p_v < 1e-4 else f"{p_v:.4f}"
        print(
            f"{name:<32} | {b:<9.4f} | {s:<8.4f} | {z_stat:<7.2f} | "
            f"{p_str:<10} | {or_val:<11.4f} | [{ci_l:.4f}, {ci_u:.4f}]"
        )
    print("=" * 95)

    # Rule Interpretations
    beta_rating = results["beta"][2]
    p_rating = results["p_values"][2]
    beta_delta = results["beta"][3]
    p_delta = results["p_values"][3]
    beta_units = results["beta"][4]
    p_units = results["p_values"][4]
    beta_hour = results["beta"][5]
    p_hour = results["p_values"][5]
    or_hour = results["odds_ratios"][5]
    beta_wknd = results["beta"][6]
    p_wknd = results["p_values"][6]
    or_wknd = results["odds_ratios"][6]

    print("\nSTATISTICAL INTERPRETATION OF SESSION STOPPING RULES:")
    print("-" * 95)

    # 1. Satisfaction vs Frustration stopping
    print("1. Satisfaction vs. Frustration Stopping Rule:")
    if p_rating < 0.05 and beta_rating > 0:
        print(
            f"   - SATISFACTION STOPPING SUPPORTED (beta = {beta_rating:+.4f}, p = {p_rating:.4g}):\n"
            "     Higher rated drinks significantly increase the probability of stopping (quitting while ahead)."
        )
    elif p_rating < 0.05 and beta_rating < 0:
        print(
            f"   - FRUSTRATION STOPPING SUPPORTED (beta = {beta_rating:+.4f}, p = {p_rating:.4g}):\n"
            "     Lower rated drinks significantly increase the probability of stopping (abandoning a session in disgust)."
        )
    else:
        print(
            f"   - NEITHER SATISFACTION NOR FRUSTRATION STOPPING (beta = {beta_rating:+.4f}, p = {p_rating:.4f}):\n"
            "     Drink rating has no statistically significant effect on session termination.\n"
            f"     Rating momentum/delta is also non-significant (beta = {beta_delta:+.4f}, p = {p_delta:.4f})."
        )

    # 2. Curfew / Circadian stopping
    print("\n2. Curfew / Circadian Stopping Rule:")
    if p_hour < 0.05 and beta_hour > 0:
        print(
            f"   - STRONG CIRCADIAN CURFEW EFFECT (beta = {beta_hour:+.4f}, p = {p_hour:.4e}):\n"
            f"     Each additional hour into the night increases the odds of session termination by "
            f"{(or_hour - 1.0) * 100:.1f}% (OR = {or_hour:.4f})."
        )
    else:
        print(
            f"   - Circadian hour has no significant effect (beta = {beta_hour:+.4f}, p = {p_hour:.4f})."
        )

    # 3. Weekend Extension Effect
    print("\n3. Weekend vs. Weekday Context:")
    if p_wknd < 0.05:
        print(
            f"   - WEEKEND SESSION EXTENSION (beta = {beta_wknd:+.4f}, p = {p_wknd:.4e}):\n"
            f"     On weekends (Fri/Sat), the odds of ending the session at any given drink are "
            f"reduced by {(1.0 - or_wknd) * 100:.1f}% (OR = {or_wknd:.4f}), allowing longer sessions."
        )

    # 4. Physiological Inebriation
    print("\n4. Physiological / Cumulative Alcohol Inebriation:")
    if p_units < 0.05:
        print(
            f"   - CUMULATIVE UNITS EFFECT (beta = {beta_units:+.4f}, p = {p_units:.4g}):\n"
            "     Total alcohol consumed significantly shifts termination hazard after controlling for time."
        )
    else:
        print(
            f"   - Cumulative units effect is modest/non-significant once time of night is controlled "
            f"(beta = {beta_units:+.4f}, p = {p_units:.4f})."
        )

    print("-" * 95)


@untappd_utils.show_or_save_to_out_file
def plot_session_termination_hazard(
    sessions: list[list[untappd.Checkin]],
    hazard_data: dict,
    results: dict,
) -> None:
    """Generates the 4-panel diagnostic plot saved to out_file."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Drinking Session Termination Hazard & Stopping Rule Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Kaplan-Meier Survival Curve & Empirical Hazard Rate
    ax1 = axes[0, 0]
    session_lengths = np.array([len(s) for s in sessions])
    max_k = 15
    k_vals = np.arange(1, max_k + 1)
    n_at_risk = np.array([np.sum(session_lengths >= k) for k in k_vals])
    n_events = np.array([np.sum(session_lengths == k) for k in k_vals])
    hazard = n_events / n_at_risk
    survival = np.cumprod(1.0 - hazard)

    ax1.step(
        k_vals,
        survival,
        where="post",
        color="#1f77b4",
        linewidth=2.5,
        label="Survival S(k) = P(Length ≥ k)",
    )
    ax1.plot(
        k_vals,
        hazard,
        "o--",
        color="#d62728",
        linewidth=1.5,
        markersize=5,
        label="Hazard h(k) = P(Stop at k | Reached k)",
    )
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Drink Index in Session (k)", fontsize=11)
    ax1.set_ylabel("Probability", fontsize=11)
    ax1.set_title(
        "A. Session Length Survival & Hazard Function",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlim(1, max_k)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(k_vals)
    ax1.legend(loc="upper right", frameon=True)

    median_len = np.median(session_lengths)
    mean_len = np.mean(session_lengths)
    ax1.text(
        0.05,
        0.25,
        f"Total Sessions: {len(sessions):,}\n"
        f"Median Length: {median_len:.0f} drinks\n"
        f"Mean Length: {mean_len:.2f} drinks\n"
        f"Max Length: {np.max(session_lengths)} drinks",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.9,
            edgecolor="#ccc",
        ),
    )

    # Panel B: Odds Ratios Forest Plot
    ax2 = axes[0, 1]
    pred_indices = range(1, len(results["feature_names"]))
    feature_labels = [results["feature_names"][i] for i in pred_indices]
    ors = [results["odds_ratios"][i] for i in pred_indices]
    ci_lows = [results["ci_lower_or"][i] for i in pred_indices]
    ci_highs = [results["ci_upper_or"][i] for i in pred_indices]
    p_values_pred = [results["p_values"][i] for i in pred_indices]
    y_positions = np.arange(len(feature_labels))

    colors = ["#2ca02c" if p < 0.05 else "#7f7f7f" for p in p_values_pred]

    for y_pos, or_v, low, high, col, p_v in zip(
        y_positions, ors, ci_lows, ci_highs, colors, p_values_pred
    ):
        ax2.errorbar(
            or_v,
            y_pos,
            xerr=[[or_v - low], [high - or_v]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2,
            capsize=5,
            capthick=1.5,
            markersize=7,
        )
        sig_label = "p < 0.001" if p_v < 0.001 else f"p = {p_v:.3f}"
        ax2.text(
            max(high, or_v) * 1.03,
            y_pos,
            f"OR={or_v:.2f} ({sig_label})",
            va="center",
            fontsize=9,
            color=col,
            fontweight="bold" if p_v < 0.05 else "normal",
        )

    ax2.axvline(1.0, color="black", linestyle="--", alpha=0.7, linewidth=1.2)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(feature_labels, fontsize=10)
    ax2.set_xlabel("Odds Ratio for Session Termination (95% CI)", fontsize=11)
    ax2.set_title(
        "B. Stopping Rule Predictors (Logistic Hazard ORs)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlim(0.4, 1.8)
    ax2.invert_yaxis()

    # Panel C: Termination Probability vs Hour of Night (Circadian Clock)
    ax3 = axes[1, 0]
    hour_grid = np.linspace(14.0, 28.0, 100)
    k_ref = 3.0
    r_ref = 3.75
    rd_ref = 0.0
    u_ref = 4.0

    beta = results["beta"]

    def predict_prob(h_vals, is_wknd):
        X_pred = np.column_stack(
            [
                np.ones_like(h_vals),
                np.full_like(h_vals, k_ref),
                np.full_like(h_vals, r_ref),
                np.full_like(h_vals, rd_ref),
                np.full_like(h_vals, u_ref),
                h_vals,
                np.full_like(h_vals, is_wknd),
            ]
        )
        logits = X_pred @ beta
        return 1.0 / (1.0 + np.exp(-logits))

    p_weekday = predict_prob(hour_grid, 0.0)
    p_weekend = predict_prob(hour_grid, 1.0)

    ax3.plot(
        hour_grid,
        p_weekday,
        color="#d62728",
        linewidth=2.5,
        label="Weekday (Sun–Thu Model)",
    )
    ax3.plot(
        hour_grid,
        p_weekend,
        color="#1f77b4",
        linewidth=2.5,
        label="Weekend (Fri–Sat Model)",
    )

    all_hours = hazard_data["hour"]
    all_terms = hazard_data["terminal"]
    all_wknd = hazard_data["is_weekend"]

    h_bins = np.arange(14, 29, 1)
    bin_centers = h_bins[:-1] + 0.5
    emp_wd = []
    emp_we = []
    for b_start, b_end in zip(h_bins[:-1], h_bins[1:]):
        mask_wd = (all_hours >= b_start) & (all_hours < b_end) & (all_wknd == 0)
        mask_we = (all_hours >= b_start) & (all_hours < b_end) & (all_wknd == 1)
        emp_wd.append(np.mean(all_terms[mask_wd]) if np.sum(mask_wd) >= 10 else np.nan)
        emp_we.append(np.mean(all_terms[mask_we]) if np.sum(mask_we) >= 10 else np.nan)

    ax3.scatter(
        bin_centers,
        emp_wd,
        color="#d62728",
        alpha=0.6,
        s=30,
        zorder=4,
        label="Empirical Weekday",
    )
    ax3.scatter(
        bin_centers,
        emp_we,
        color="#1f77b4",
        alpha=0.6,
        s=30,
        zorder=4,
        label="Empirical Weekend",
    )

    tick_hours = [14, 16, 18, 20, 22, 24, 26, 28]
    tick_labels = [
        "14:00\n(2pm)",
        "16:00\n(4pm)",
        "18:00\n(6pm)",
        "20:00\n(8pm)",
        "22:00\n(10pm)",
        "00:00\n(Mid)",
        "02:00\n(2am)",
        "04:00\n(4am)",
    ]
    ax3.set_xticks(tick_hours)
    ax3.set_xticklabels(tick_labels, fontsize=9)
    ax3.set_xlabel("Circadian Time of Day / Hour of Night", fontsize=11)
    ax3.set_ylabel("P(Session Termination at Drink k)", fontsize=11)
    ax3.set_title(
        "C. Curfew Effect: Termination Probability vs Hour of Night",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_ylim(0, 1.0)
    ax3.legend(loc="upper left", frameon=True)

    # Panel D: Termination Probability vs Cumulative Alcohol Units
    ax4 = axes[1, 1]
    unit_grid = np.linspace(0.5, 12.0, 100)

    for h_val, h_name, col in [
        (20.0, "20:00 (8pm)", "#2ca02c"),
        (23.0, "23:00 (11pm)", "#ff7f0e"),
        (25.5, "01:30 (1:30am)", "#9467bd"),
    ]:
        X_pred = np.column_stack(
            [
                np.ones_like(unit_grid),
                np.full_like(unit_grid, k_ref),
                np.full_like(unit_grid, r_ref),
                np.full_like(unit_grid, rd_ref),
                unit_grid,
                np.full_like(unit_grid, h_val),
                np.full_like(unit_grid, 0.5),
            ]
        )
        logits = X_pred @ beta
        p_units = 1.0 / (1.0 + np.exp(-logits))
        ax4.plot(
            unit_grid,
            p_units,
            linewidth=2.2,
            color=col,
            label=f"At {h_name}",
        )

    ax4.set_xlabel("Cumulative Alcohol Units in Session", fontsize=11)
    ax4.set_ylabel("P(Session Termination at Drink k)", fontsize=11)
    ax4.set_title(
        "D. Termination Hazard vs Cumulative Units at Fixed Hours",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc="upper left", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == "__main__":
    checkins = untappd.load_latest_checkins()
    sessions = compute_sessions(checkins, max_gap_hours=MAX_GAP_HOURS)
    hazard_data = build_hazard_dataset(sessions)
    results = fit_logistic_hazard(hazard_data)

    print_hazard_summary(results)

    out_file = Path(__file__).parent / "out" / "session_termination_hazard.png"
    plot_session_termination_hazard(
        sessions,
        hazard_data,
        results,
        out_file=out_file,
    )
