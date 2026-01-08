"""Statistical methods for cohort comparisons."""

from lifelines import CoxPHFitter  # type: ignore
import pandas as pd  # type: ignore


def calculate_cox_hazard_ratio(
    cohort_a_data: pd.DataFrame, cohort_b_data: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate hazard ratio using Cox proportional hazards model.

    Parameters
    ----------
    cohort_a_data : pd.DataFrame
        Must have columns: PATIENT_ID, time, event
    cohort_b_data : pd.DataFrame
        Must have columns: PATIENT_ID, time, event

    Returns
    -------
    dict[str, float]
        Keys: hazard_ratio, confidence_interval_lower,
              confidence_interval_upper, p_value
    """
    # Combine cohorts and add treatment indicator
    cohort_a_data = cohort_a_data.copy()
    cohort_a_data["cohort"] = 1  # cohort A = treatment

    cohort_b_data = cohort_b_data.copy()
    cohort_b_data["cohort"] = 0  # cohort B = reference

    combined = pd.concat([cohort_a_data, cohort_b_data])

    # Select only columns needed for Cox model (exclude PATIENT_ID)
    cox_data = combined[["time", "event", "cohort"]]

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="time", event_col="event")

    # Extract results
    hr = cph.hazard_ratios_["cohort"]
    ci = cph.confidence_intervals_.loc["cohort"]
    p_value = cph.summary.loc["cohort", "p"]

    return {
        "hazard_ratio": float(hr),
        "confidence_interval_lower": float(ci.iloc[0]),
        "confidence_interval_upper": float(ci.iloc[1]),
        "p_value": float(p_value),
    }
