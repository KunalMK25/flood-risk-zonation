"""
Per-cell explainability for the Flood Risk Zonation System.

Provides ``build_cell_explanation`` which turns a single scored-grid row
into a (tooltip_html, popup_html) pair used by the Folium map.

Design principles
-----------------
- No re-computation of risk scores — only formats data already in the row.
- Factor contributions reflect the normalised value for THIS cell relative
  to the dataset range (model_bounds), not a fixed global scale.
- The dominant factor is whichever has the highest risk contribution for
  this specific cell — the tooltip label reflects the actual direction.
- Every cell produces a unique, data-driven summary sentence.
- Water cells get a dedicated explanation (no factor breakdown needed).
- Coastal cells show the ⚠️ Tsunami Risk badge.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from flood_risk_zonation.scoring.susceptibility import FACTOR_WEIGHTS
from flood_risk_zonation.visualization.layers import RISK_COLOR_MAP

# ── Factor metadata ────────────────────────────────────────────────────────────
# (label, unit, icon, low-risk phrase, high-risk phrase)
# low-risk phrase: used when this cell's value pushes risk DOWN
# high-risk phrase: used when this cell's value pushes risk UP
_FACTOR_META: dict[str, tuple[str, str, str, str, str]] = {
    "elevation_m":         ("Elevation",         "m",    "⛰",
                            "high elevation",          "low elevation"),
    "slope_deg":           ("Slope",             "°",    "📐",
                            "steep slope drains water","flat terrain pools water"),
    "twi":                 ("Wetness (TWI)",     "",     "💧",
                            "low wetness index",       "high wetness index"),
    "rainfall_mean_mm":    ("Annual Rainfall",   "mm",   "🌧",
                            "low annual rainfall",     "high annual rainfall"),
    "rainfall_max_24h_mm": ("Peak Rainfall",     "mm",   "⛈",
                            "low peak rainfall",       "extreme peak rainfall"),
    "dist_water_m":        ("Dist. to Water",    "m",    "🏞",
                            "far from water bodies",   "very close to water"),
    "drainage_capacity":   ("Drainage",          "",     "🚰",
                            "good drainage capacity",  "poor drainage capacity"),
    "population_density":  ("Population",        "/km²", "👥",
                            "low population density",  "high population density"),
    "curvature":           ("Curvature",         "",     "〰",
                            "convex terrain drains",   "concave terrain collects water"),
}

_WATER_TYPE_LABELS: dict[str, str] = {
    "coastline": "ocean/sea (coastline)",
    "bay": "bay/sea", "sea": "sea", "ocean": "ocean",
    "water": "lake/pond", "river": "river", "canal": "canal",
    "stream": "stream", "drain": "drainage channel",
    "reservoir": "reservoir", "basin": "basin",
}

_FALLBACK_BOUNDS: dict[str, tuple[float, float]] = {
    "elevation_m":         (0.0,    500.0),
    "slope_deg":           (0.0,    45.0),
    "twi":                 (0.0,    20.0),
    "rainfall_mean_mm":    (200.0,  3000.0),
    "rainfall_max_24h_mm": (10.0,   300.0),
    "dist_water_m":        (0.0,    5000.0),
    "drainage_capacity":   (0.0,    1.0),
    "population_density":  (0.0,    5000.0),
    "curvature":           (-10.0,  10.0),
}


def _normalise(value: float, feature: str, lo: float, hi: float) -> float:
    """
    Return the risk-contribution normalised value in [0, 1] for a factor.
    0 = low risk contribution, 1 = high risk contribution.
    Mirrors WeightedSusceptibilityModel per-factor logic exactly.
    """
    direction = FACTOR_WEIGHTS[feature][1]
    if hi <= lo:
        return 0.5
    n = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    return (1.0 - n) if direction < 0 else n


def _get_bounds(feature: str, model_bounds: dict | None) -> tuple[float, float]:
    if model_bounds and feature in model_bounds:
        return model_bounds[feature]
    return _FALLBACK_BOUNDS.get(feature, (0.0, 1.0))


def _html_bar(risk_pct: float, width_px: int = 88) -> str:
    pct = int(min(100, max(0, risk_pct)))
    color = "#e74c3c" if pct > 66 else "#f39c12" if pct > 33 else "#2ecc71"
    return (
        f'<div style="display:inline-block;background:#e0e0e0;border-radius:3px;'
        f'height:7px;width:{width_px}px;vertical-align:middle">'
        f'<div style="background:{color};width:{pct}%;height:7px;border-radius:3px">'
        f'</div></div>'
    )


def _format_val(val: float, unit: str) -> str:
    if unit == "m":
        return f"{val:.0f} m"
    if unit == "mm":
        return f"{val:.0f} mm"
    if unit == "°":
        return f"{val:.1f}°"
    if unit == "/km²":
        return f"{val:.0f}/km²"
    return f"{val:.2f}"


def _compute_factors(
    row: Any,
    model_bounds: dict | None,
) -> list[tuple[float, float, str, str, str, str, str]]:
    """
    Compute per-factor data for a cell.

    Returns a list of tuples:
        (n, contribution, feature, label, unit, icon, phrase)
    where
        n            = normalised risk contribution [0, 1] for this cell
        contribution = n * weight  (used to rank factors)
        feature      = column name
        label        = display label
        unit         = value unit string
        icon         = emoji icon
        phrase       = specific plain-language phrase for THIS cell
    """
    results = []
    for feature, (weight, _) in FACTOR_WEIGHTS.items():
        if feature not in _FACTOR_META:
            continue
        val = row.get(feature, None)
        if val is None:
            continue
        val = float(val)
        if np.isnan(val):
            continue
        lo, hi = _get_bounds(feature, model_bounds)
        n = _normalise(val, feature, lo, hi)
        contribution = n * weight
        label, unit, icon, low_phrase, high_phrase = _FACTOR_META[feature]
        # Pick the phrase that matches THIS cell's actual direction
        phrase = high_phrase if n >= 0.5 else low_phrase
        results.append((n, contribution, feature, label, unit, icon, phrase))
    return results


def _build_summary(
    risk_class: str,
    risk_score: float,
    factors: list[tuple[float, float, str, str, str, str, str]],
    row: Any,
) -> str:
    """
    Generate a unique, data-driven summary sentence for this specific cell.
    Uses the actual factor values and their actual direction for this cell.
    """
    if not factors:
        return f"Risk score: {risk_score:.0f}/100."

    # Sort by contribution descending
    by_contrib = sorted(factors, key=lambda x: x[1], reverse=True)
    top = by_contrib[:3]

    # Collect risky factors (n > 0.5) and safe factors (n < 0.5)
    risky = [(n, contrib, feat, label, unit, icon, phrase)
             for n, contrib, feat, label, unit, icon, phrase in factors if n > 0.5]
    safe = [(n, contrib, feat, label, unit, icon, phrase)
            for n, contrib, feat, label, unit, icon, phrase in factors if n <= 0.5]

    risky_by_contrib = sorted(risky, key=lambda x: x[1], reverse=True)
    safe_by_contrib = sorted(safe, key=lambda x: x[1], reverse=False)

    def _val_phrase(feat, unit):
        """Return a concrete value string like '12 m' or '0.28'."""
        val = row.get(feat, None)
        if val is None or (isinstance(val, float) and np.isnan(float(val))):
            return ""
        return _format_val(float(val), unit)

    if risk_class == "High":
        if risky_by_contrib:
            top_risky = risky_by_contrib[0]
            n0, c0, f0, l0, u0, i0, p0 = top_risky
            v0 = _val_phrase(f0, u0)
            val_note = f" ({v0})" if v0 else ""
            sentence = f"{l0}{val_note} is the primary driver — {p0}"
            if len(risky_by_contrib) > 1:
                t2 = risky_by_contrib[1]
                l1, p1 = t2[3], t2[6]
                sentence += f", worsened by {l1.lower()} ({p1})"
            if safe_by_contrib:
                s = safe_by_contrib[0]
                ls, ps = s[3], s[6]
                sentence += f". Note: {ls.lower()} ({ps}) partially offsets risk"
        else:
            # High risk but no single strongly risky factor — combined effect
            names = [f[3] for f in by_contrib[:3]]
            sentence = (
                f"Multiple moderate factors combine to produce high risk: "
                f"{', '.join(n.lower() for n in names)}"
            )

    elif risk_class == "Medium":
        if risky_by_contrib:
            top_risky = risky_by_contrib[0]
            l0, u0, f0, p0 = top_risky[3], top_risky[4], top_risky[2], top_risky[6]
            v0 = _val_phrase(f0, u0)
            val_note = f" ({v0})" if v0 else ""
            sentence = f"{l0}{val_note} raises vulnerability — {p0}"
            if safe_by_contrib:
                s = safe_by_contrib[0]
                ls, ps = s[3], s[6]
                sentence += f", but {ls.lower()} ({ps}) keeps risk moderate"
        else:
            sentence = "Moderate risk — no dominant vulnerability factor identified"

    else:  # Low
        if safe_by_contrib:
            top_safe = safe_by_contrib[0]  # lowest contribution = most protective
            l0, u0, f0, p0 = top_safe[3], top_safe[4], top_safe[2], top_safe[6]
            v0 = _val_phrase(f0, u0)
            val_note = f" ({v0})" if v0 else ""
            sentence = f"{l0}{val_note} — {p0}"
            if len(safe_by_contrib) > 1:
                s2 = safe_by_contrib[1]
                ls, ps = s2[3], s2[6]
                sentence += f". {ls} also favourable ({ps})"
        else:
            sentence = "Low risk — most conditioning factors are within safe ranges"

    return sentence


def build_cell_explanation(
    row: Any,
    model_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[str, str]:
    """
    Build (tooltip_html, popup_html) for a single scored grid row.

    Every cell produces unique, data-driven text based on its actual
    feature values — not a fixed template.

    Parameters
    ----------
    row : pd.Series or dict-like
        One row from the scored GeoDataFrame.
    model_bounds : dict | None
        Fitted normalisation bounds {feature: (lower_5th_pct, upper_95th_pct)}.

    Returns
    -------
    (tooltip_html, popup_html)
    """
    risk_class = str(row.get("risk_class", "Low"))
    risk_score = float(row.get("risk_score", 0.0))
    risk_color = RISK_COLOR_MAP.get(risk_class, "#999999")
    is_coastal = bool(row.get("is_coastal_tsunami_risk", False))

    # ── Water cells ───────────────────────────────────────────────────────────
    if risk_class == "Water":
        wtype_raw = str(row.get("water_type", "")).lower()
        wtype_label = _WATER_TYPE_LABELS.get(wtype_raw, "permanent water body")
        tooltip_html = (
            f'<div style="font-family:Arial,sans-serif;font-size:12px;'
            f'padding:3px 6px;background:{risk_color};color:white;border-radius:3px">'
            f'<b>💧 Water</b></div>'
        )
        popup_html = (
            f'<div style="font-family:Arial,sans-serif;font-size:12px;'
            f'min-width:190px;max-width:240px">'
            f'<div style="background:{risk_color};color:white;padding:5px 8px;'
            f'border-radius:4px;margin-bottom:6px"><b>💧 Water Body</b></div>'
            f'<div style="color:#555;font-size:11px;line-height:1.5">'
            f'This cell is a <b>{wtype_label}</b>.<br>'
            f'Masked as permanent water — flood risk scoring does not apply.'
            f'</div></div>'
        )
        return tooltip_html, popup_html

    # ── Compute per-factor data for this specific cell ────────────────────────
    factors = _compute_factors(row, model_bounds)

    # Dominant factor for tooltip: highest risk contribution for THIS cell
    risky_factors = sorted(
        [(n, c, f, l, u, i, p) for n, c, f, l, u, i, p in factors if n >= 0.5],
        key=lambda x: x[1], reverse=True
    )
    all_by_contrib = sorted(factors, key=lambda x: x[1], reverse=True)

    if risky_factors:
        top_n, top_c, top_feat, top_label, top_unit, top_icon, top_phrase = risky_factors[0]
        top_val = row.get(top_feat, None)
        val_str = _format_val(float(top_val), top_unit) if top_val is not None else ""
        tooltip_factor = f"{top_icon} {top_label}: {val_str} — {top_phrase}" if val_str else f"{top_icon} {top_phrase}"
    elif all_by_contrib:
        # Low risk cell — show most protective factor
        protective = sorted(factors, key=lambda x: x[1])  # lowest contribution first
        top_n, top_c, top_feat, top_label, top_unit, top_icon, top_phrase = protective[0]
        top_val = row.get(top_feat, None)
        val_str = _format_val(float(top_val), top_unit) if top_val is not None else ""
        tooltip_factor = f"{top_icon} {top_label}: {val_str} — {top_phrase}" if val_str else f"{top_icon} {top_phrase}"
    else:
        tooltip_factor = "No factor data available"

    coastal_note = " · ⚠️ Coastal" if is_coastal else ""
    tooltip_html = (
        f'<div style="font-family:Arial,sans-serif;font-size:12px;padding:4px 7px;'
        f'background:{risk_color};color:white;border-radius:3px;max-width:180px">'
        f'<b>{risk_class} Risk</b> — {risk_score:.0f}/100{coastal_note}'
        f'</div>'
    )

    # ── Full popup ────────────────────────────────────────────────────────────
    lines: list[str] = []

    # Header
    lines.append(
        f'<div style="background:{risk_color};color:white;padding:5px 8px;'
        f'border-radius:4px;margin-bottom:5px;font-size:13px">'
        f'<b>{risk_class} Risk</b>'
        f'<span style="font-size:11px;opacity:0.85;float:right">'
        f'Score: {risk_score:.0f}/100</span></div>'
    )

    # Coastal badge
    if is_coastal:
        lines.append(
            '<div style="background:#e67e22;color:white;padding:3px 7px;'
            'border-radius:3px;margin-bottom:5px;font-size:11px;font-weight:bold">'
            '⚠️ Coastal — Tsunami Risk &nbsp;'
            '<span style="font-weight:normal;font-size:10px">'
            'Adjacent to ocean/sea</span></div>'
        )

    # Factor table — sorted by risk contribution descending (most risky first)
    factors_sorted = sorted(factors, key=lambda x: x[1], reverse=True)

    lines.append(
        '<table style="width:100%;border-collapse:collapse;font-size:11px">'
        '<tr style="background:#f4f4f4">'
        '<th style="text-align:left;padding:3px 4px;font-weight:600">Factor</th>'
        '<th style="padding:3px 4px;font-weight:600">Level</th>'
        '<th style="text-align:right;padding:3px 4px;font-weight:600">Value</th>'
        '</tr>'
    )

    for n, contrib, feature, label, unit, icon, phrase in factors_sorted:
        bar = _html_bar(n * 100)
        raw_val = row.get(feature, None)
        val_display = _format_val(float(raw_val), unit) if raw_val is not None else "—"
        # Row background: highlight risky factors subtly
        row_bg = "background:#fff5f5" if n > 0.66 else ""
        lines.append(
            f'<tr style="border-bottom:1px solid #f0f0f0;{row_bg}">'
            f'<td style="padding:3px 4px;white-space:nowrap">'
            f'{icon} {label}</td>'
            f'<td style="padding:3px 4px">{bar}</td>'
            f'<td style="padding:3px 4px;text-align:right;'
            f'color:#333;font-size:10px">{val_display}</td>'
            f'</tr>'
        )

    lines.append('</table>')

    # Unique summary sentence derived from actual cell data
    summary = _build_summary(risk_class, risk_score, factors, row)
    lines.append(
        f'<div style="margin-top:5px;font-size:11px;color:#333;'
        f'padding:4px 6px;background:#f8f9fa;border-left:3px solid {risk_color};'
        f'border-radius:0 3px 3px 0;line-height:1.5">'
        f'{summary}.</div>'
    )

    popup_html = (
        f'<div style="font-family:Arial,sans-serif;font-size:12px;'
        f'min-width:250px;max-width:310px">'
        + "".join(lines)
        + '</div>'
    )

    return tooltip_html, popup_html
