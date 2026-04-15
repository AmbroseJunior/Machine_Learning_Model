"""
Generates synthetic survey-style data to use until real data is collected.
Each row represents one respondent's session answers.
"""
import numpy as np
import pandas as pd

def generate_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(16, 65, size=n)
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])

    # Slider questions (0–100)
    q1_focus       = rng.integers(0, 101, size=n)   # "How focused do you feel?"
    q2_stress      = rng.integers(0, 101, size=n)   # "Current stress level"
    q3_motivation  = rng.integers(0, 101, size=n)   # "Motivation to learn"
    q4_satisfaction = rng.integers(0, 101, size=n)  # "Satisfaction with results"

    # Checkbox / categorical questions
    q5_sleep = rng.choice(["<5h", "5-7h", "7-9h", ">9h"], size=n, p=[0.1, 0.3, 0.45, 0.15])
    q6_exercise = rng.choice(["Never", "1-2x/week", "3-4x/week", "Daily"], size=n, p=[0.2, 0.35, 0.3, 0.15])

    # Derived target: "performance_category"  (Low / Medium / High)
    # Based loosely on focus, motivation, satisfaction minus stress
    score = (q1_focus * 0.3 + q3_motivation * 0.3 + q4_satisfaction * 0.2 - q2_stress * 0.2)
    score += rng.normal(0, 5, size=n)  # noise

    labels = pd.cut(
        score,
        bins=[-np.inf, 30, 55, np.inf],
        labels=["Low", "Medium", "High"],
    )

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "focus_score": q1_focus,
        "stress_level": q2_stress,
        "motivation": q3_motivation,
        "satisfaction": q4_satisfaction,
        "sleep_hours": q5_sleep,
        "exercise_freq": q6_exercise,
        "performance_category": labels,
    })

    return df


if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv("sample_data.csv", index=False)
    print(f"Saved {len(df)} rows to sample_data.csv")
    print(df["performance_category"].value_counts())
