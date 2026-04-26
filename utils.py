import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_fairness(
    X_test_raw,
    y_test,
    y_pred,
    y_prob=None,
    group_col="Gender",
    group_map=None
):
    # Build evaluation dataframe
    eval_df = X_test_raw.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred

    if y_prob is not None:
        eval_df["y_prob"] = y_prob

    ## Optional mapping (e.g., 0/1 → Female/Male)
    if group_map is not None:
        eval_df[group_col + "_label"] = eval_df[group_col].map(group_map)
        group_col = group_col + "_label"

    rows = []

    for group_name, group_data in eval_df.groupby(group_col, observed=True):
        rows.append({
            "group": str(group_name),
            "count": len(group_data),
            "accuracy": accuracy_score(group_data["y_true"], group_data["y_pred"]),
            "precision": precision_score(group_data["y_true"], group_data["y_pred"], zero_division=0),
            "recall": recall_score(group_data["y_true"], group_data["y_pred"], zero_division=0),
            "f1": f1_score(group_data["y_true"], group_data["y_pred"], zero_division=0),
        })

    fairness_df = pd.DataFrame(rows)
    print(f"\nFairness evaluation by {group_col}:")
    print(fairness_df)

    return fairness_df


def evaluate_gender_fairness( X_test, y_test, y_pred, y_prob):
    return evaluate_fairness(
    X_test,
    y_test,
    y_pred,
    y_prob,
    group_col="Gender",
    group_map={0: "Female", 1: "Male"}
)

def evaluate_age_fairness( X_test, y_test, y_pred, y_prob):
    X_test["Age_group"] = pd.cut(
        X_test["Age"],
        bins=[0, 18, 40, 65, 120],
        labels=["0-17", "18-39", "40-64", "65+"],
        right=False
    )
    return evaluate_fairness(
    X_test,
    y_test,
    y_pred,
    y_prob,
    group_col="Age_group"
)