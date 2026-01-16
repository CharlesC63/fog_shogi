"""
Analysis tools for Fog of War Shogi.

Provides:
- Regret bound computation
- Convergence rate estimation
- Exploitability analysis
- Sample complexity analysis
"""

from .regret import (
    RegretAnalyzer,
    RegretBound,
    SampleComplexityAnalyzer,
    compare_solvers,
)

__all__ = [
    "RegretAnalyzer",
    "RegretBound",
    "SampleComplexityAnalyzer",
    "compare_solvers",
]
