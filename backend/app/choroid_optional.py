"""Optional Choroidalyzer integration protocol.

The model requires separately downloaded weights and PyTorch. This module never
silently substitutes offset-based choroid estimates as neural predictions.
"""
from __future__ import annotations


def availability() -> dict:
    try:
        from choroidalyze import Choroidalyzer  # noqa: F401
        return {'installed':True,'status':'Model package importable; weights and local inference need independent validation.'}
    except ImportError:
        return {'installed':False,'status':'Choroidalyzer model not installed; current choroid projection uses heuristic offsets.'}
