from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd


def generate_formula_json(
    context_name: str,
    summary: pd.DataFrame,
    coef_summary: pd.DataFrame,
    n_train_samples: int,
    output_path: str,
) -> Dict:
    """
    Generate a formula.json file for the best model (lowest MAE).

    Args:
        context_name: Name of the context (e.g., 'waha', 'lln', 'ixelle').
        summary: DataFrame with aggregated metrics per model (from lofo_cv).
        coef_summary: DataFrame with averaged coefficients per model.
        n_train_samples: Total number of training samples used.
        output_path: Where to save the JSON file.

    Returns:
        The formula dictionary (also saved to output_path).
    """
    if summary.empty or coef_summary.empty:
        raise ValueError("Cannot generate formula from empty summary or coef_summary")

    # Best model = first row of summary (already sorted by MAE ascending)
    best_row = summary.iloc[0]
    model_name = best_row["model"]
    features_str = best_row["features"]
    features = [f.strip() for f in features_str.split(",")]

    # Get coefficient row for this model
    coef_row = coef_summary[coef_summary["model"] == model_name]
    if coef_row.empty:
        raise ValueError(f"No coefficient data found for model: {model_name}")

    coef_row = coef_row.iloc[0]

    # Build coefficients dict
    coefficients = {"intercept": float(coef_row["intercept_mean"])}
    for feat in features:
        col_name = f"coef_{feat}_mean"
        if col_name in coef_row:
            coefficients[f"coef_{feat}"] = float(coef_row[col_name])

    # Build performance metrics
    performance_metrics = {
        "mae_mean_m": float(best_row["mae_mean_m"]),
        "rmse_mean_m": float(best_row["rmse_mean_m"]),
        "mape_mean": float(best_row["mape_mean"]),
        "folds": int(best_row["folds"]),
        "train_samples": int(n_train_samples),
    }

    # Auto-generate description and notes based on features
    description, notes = _generate_description_and_notes(features)

    formula = {
        "context": context_name,
        "model": model_name,
        "description": description,
        "features": features,
        "coefficients": coefficients,
        "performance_metrics": performance_metrics,
        "notes": notes,
    }

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formula, f, indent=2, ensure_ascii=False)

    return formula


def _generate_description_and_notes(features: list) -> tuple:
    """Auto-generate description and notes based on feature types."""
    has_polynomial = any("_sq" in f for f in features)
    has_interaction = any("_x_" in f for f in features)
    has_earfcn = "earfcn_k" in features

    if has_polynomial and has_interaction:
        description = "Combined polynomial and interaction regression model"
    elif has_polynomial:
        description = "Polynomial regression model with squared terms"
    elif has_interaction:
        interaction_term = next((f for f in features if "_x_" in f), None)
        if interaction_term:
            parts = interaction_term.split("_x_")
            description = f"Nonlinear regression model with interaction term {parts[0]} * {parts[1]}"
        else:
            description = "Regression model with interaction terms"
    else:
        description = "Linear regression model"

    # Build notes
    notes_parts = []

    if has_earfcn:
        notes_parts.append("earfcn_k = EARFCN / 1000")

    # Document squared terms
    squared_terms = [f for f in features if f.endswith("_sq")]
    if squared_terms:
        squared_docs = []
        for term in squared_terms:
            base = term.replace("_sq", "")
            squared_docs.append(f"{term} = {base}^2")
        notes_parts.append("Polynomial terms: " + ", ".join(squared_docs))

    # Document interaction terms
    interaction_terms = [f for f in features if "_x_" in f]
    if interaction_terms:
        interaction_docs = []
        for term in interaction_terms:
            parts = term.split("_x_")
            interaction_docs.append(f"{term} = {parts[0]} * {parts[1]}")
        notes_parts.append("Interaction terms: " + ", ".join(interaction_docs))

    notes = ". ".join(notes_parts) + "." if notes_parts else "Linear model using base features."

    return description, notes
