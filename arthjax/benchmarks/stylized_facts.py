"""Stylized macro facts computed from simulation metrics history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class StylizedFactResult:
    name: str
    value: float
    expected_range: tuple[float, float]
    passed: bool
    note: str = ""


def _autocorr(series: np.ndarray, lag: int = 1) -> float:
    if len(series) <= lag + 1:
        return 0.0
    x = series[:-lag]
    y = series[lag:]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rolling_std(series: np.ndarray, window: int = 40) -> np.ndarray:
    out = []
    for i in range(window, len(series)):
        out.append(float(np.std(series[i - window : i])))
    return np.array(out) if out else np.array([0.0])


def compute_stylized_facts(metrics_np: dict, burn: int = 25) -> Dict[str, StylizedFactResult]:
    """Compute pass/fail stylized facts from metrics history."""
    gdp = metrics_np["gdp"][burn:]
    infl = metrics_np["inflation"][burn:]
    unemp = metrics_np["unemployment"][burn:]
    sentiment = metrics_np["sentiment"][burn:]
    credit = metrics_np["credit_stress"][burn:]
    vol = metrics_np.get("avg_volatility")
    vol = vol[burn:] if vol is not None else None

    gdp_diff = np.diff(gdp)
    step_downs = int(np.sum(gdp_diff < -20.0))

    vol_std = _rolling_std(gdp, 40)
    vol_cluster_score = float(np.std(vol_std)) if len(vol_std) else 0.0

    phillips_slope = 0.0
    if len(unemp) > 10 and np.std(unemp) > 1e-6:
        phillips_slope = float(np.polyfit(unemp, infl, 1)[0])

    unemp_autocorr = _autocorr(unemp, lag=5)
    credit_spike = float(np.max(credit) - np.median(credit))

    facts = {
        "gdp_step_downs": StylizedFactResult(
            "gdp_step_downs",
            float(step_downs),
            (1.0, 15.0),
            1.0 <= step_downs <= 15.0,
            "Discrete output contractions (not flat line)",
        ),
        "vol_clustering": StylizedFactResult(
            "vol_clustering",
            vol_cluster_score,
            (5.0, 500.0),
            vol_cluster_score > 5.0,
            "Rolling GDP volatility varies over time",
        ),
        "phillips_slope": StylizedFactResult(
            "phillips_slope",
            phillips_slope,
            (-5.0, 2.0),
            -5.0 < phillips_slope < 2.0,
            "Inflation-unemployment co-movement (sign varies in toy model)",
        ),
        "unemployment_autocorr_lag5": StylizedFactResult(
            "unemployment_autocorr_lag5",
            unemp_autocorr,
            (0.3, 0.99),
            unemp_autocorr > 0.3,
            "Unemployment persists (positive autocorrelation)",
        ),
        "credit_stress_spike": StylizedFactResult(
            "credit_stress_spike",
            credit_spike,
            (0.02, 0.5),
            credit_spike > 0.02,
            "Credit stress shows at least one elevated episode",
        ),
        "sentiment_decline": StylizedFactResult(
            "sentiment_decline",
            float(sentiment[0] - sentiment[-1]),
            (-0.5, 0.6),
            float(sentiment[0] - sentiment[-1]) > -0.1,
            "Sentiment does not monotonically explode upward",
        ),
    }

    if "stock_index" in metrics_np and len(metrics_np["stock_index"]) > 50:
        si = metrics_np["stock_index"][burn:]
        rolling_ret_vol = []
        for i in range(40, len(si)):
            window = si[i - 40 : i + 1]
            rolling_ret_vol.append(float(np.std(np.diff(np.log(window + 1e-8)))))
        rolling_ret_vol = np.array(rolling_ret_vol)
        vol_autocorr = _autocorr(rolling_ret_vol, lag=3) if len(rolling_ret_vol) > 4 else 0.0
        facts["volatility_autocorr"] = StylizedFactResult(
            "volatility_autocorr",
            vol_autocorr,
            (0.2, 0.99),
            vol_autocorr > 0.2,
            "Rolling stock-index return volatility persistence",
        )
    elif vol is not None and len(vol) > 10 and np.std(vol) > 1e-6:
        vol_autocorr = _autocorr(vol, lag=3)
        facts["volatility_autocorr"] = StylizedFactResult(
            "volatility_autocorr",
            vol_autocorr,
            (0.2, 0.99),
            vol_autocorr > 0.2,
            "Volatility clustering in market vol series",
        )

    return facts


def summarize_facts(facts: Dict[str, StylizedFactResult]) -> str:
    lines = ["Stylized facts:"]
    passed = sum(1 for f in facts.values() if f.passed)
    lines.append(f"  Passed {passed}/{len(facts)}")
    for f in facts.values():
        mark = "PASS" if f.passed else "FAIL"
        lines.append(
            f"  [{mark}] {f.name}: {f.value:.4f} "
            f"(expected {f.expected_range[0]}–{f.expected_range[1]}) — {f.note}"
        )
    return "\n".join(lines)
