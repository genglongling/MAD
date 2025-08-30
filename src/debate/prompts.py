# src/debate/prompts.py
from __future__ import annotations
import json
from typing import Dict, Any, Optional, Tuple

EPS = 1e-9

def get_choice_keys(choices: Dict[str, Any]) -> list[str]:
    """Extract choice keys from choices dict (e.g., ['A', 'B', 'C', 'D'] or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])"""
    if 'label' in choices:
        return choices['label']
    elif 'text' in choices:
        # If we have text choices, generate labels A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
        text_choices = choices['text']
        if isinstance(text_choices, list):
            return [chr(ord('A') + i) for i in range(len(text_choices))]
    # Fallback to A, B, C, D
    return ["A", "B", "C", "D"]

# -----------------------------
# Probability helpers
# -----------------------------
def _vec(d: Dict[str, float], choice_keys: list[str]) -> list[float]:
    return [max(0.0, float(d.get(k, 0.0))) for k in choice_keys]

def normalize_probs(d: Dict[str, float], choice_keys: list[str]) -> Dict[str, float]:
    """
    Clamp negatives to 0; renormalize to sum ~ 1.
    Fallback to uniform if vector is all zeros/invalid.
    """
    v = _vec(d, choice_keys)
    s = float(sum(v))
    if s <= 0.0:
        return {k: 1.0/len(choice_keys) for k in choice_keys}
    return {k: (v[i] / s) for i, k in enumerate(choice_keys)}

def is_valid_prob_dict(d: Dict[str, float], choice_keys: list[str], tol: float = 1e-6) -> bool:
    if not isinstance(d, dict):
        return False
    try:
        v = _vec(d, choice_keys)
        if any(x < -1e-9 for x in v):  # negative (beyond tiny numeric noise)
            return False
        s = sum(v)
        return abs(s - 1.0) <= tol or s > 0.0  # accept nonzero (we normalize anyway)
    except Exception:
        return False

# -----------------------------
# Debater parsing
# -----------------------------
def parse_json_or_fallback(text: str, choice_keys: list[str]) -> Dict[str, Any]:
    """
    Parse a debater message.

    New strict schema (preferred):
    {
      "output": {"A": pA, "B": pB, "C": pC, "D": pD, ...},
      "reason": {"A": rA, "B": rB, "C": rC, "D": rD, ...}
    }

    Legacy schemas (still accepted):
    - {"probs": {...}, "rationale": "..."}
    - {"probs": {...}, "reason": {...}}

    Returns:
      {
        "probs": Dict[str, float],        # normalized over choice_keys
        "rationale": str,                 # concatenated per-choice reasons (if present)
        "reasons": Dict[str, str] | {},   # per-choice reasons if provided
      }
    """
    # print(f"PARSING TEXT: {text}")
    # Default fallback
    uniform_prob = 1.0 / len(choice_keys)
    fallback = {
        "probs": {k: uniform_prob for k in choice_keys},
        "rationale": text,
    }

    try:
        obj = json.loads(text)
        # print(f"JSON PARSE SUCCESS: {obj}")
    except Exception as e:
        # print(f"JSON PARSE FAILED: {e}")
        return fallback

    if not isinstance(obj, dict):
        # Treat other JSON as rationale-only
        return {
            "probs": {k: uniform_prob for k in choice_keys},
            "rationale": json.dumps(obj, ensure_ascii=False),
        }

    # New strict schema
    if "output" in obj or "reason" in obj:
        # print("USING NEW STRICT SCHEMA")
        probs_raw = obj.get("output", {})
        reasons = obj.get("reason", {}) or {}
        probs = normalize_probs(probs_raw if isinstance(probs_raw, dict) else {}, choice_keys)
        rationale = " ".join(
            (str(reasons.get(k, "")).strip()) for k in choice_keys if reasons.get(k)
        )
        result = {
            "probs": probs,
            "rationale": rationale,
            "reasons": reasons if isinstance(reasons, dict) else {},
        }
        # print(f"NEW SCHEMA RESULT: {result}")
        return result

    # Legacy
    if "probs" in obj:
        # print("USING LEGACY SCHEMA")
        probs = normalize_probs(obj.get("probs", {}), choice_keys)
        rationale = obj.get("rationale", "")
        reasons = obj.get("reason", {})
        if isinstance(reasons, dict) and not rationale:
            rationale = " ".join(
                (str(reasons.get(k, "")).strip()) for k in choice_keys if reasons.get(k)
            )
        out = {"probs": probs, "rationale": rationale}
        if isinstance(reasons, dict):
            out["reasons"] = reasons
        # print(f"LEGACY SCHEMA RESULT: {out}")
        return out

    # Unknown dict shape → rationale-only
    return {
        "probs": {k: uniform_prob for k in choice_keys},
        "rationale": json.dumps(obj, ensure_ascii=False),
    }

# -----------------------------
# Judge parsing
# -----------------------------
def parse_judge_json(text: str, choice_keys: list[str]) -> Dict[str, Any]:
    """
    Parse judge outputs. Supports three shapes:

    1) Per-round judge (latest):
       {
         "outputA": {...}, "outputB": {...},
         "CRIT_A": float, "CRIT_B": float
       }

    2) Older alt (you mentioned): {"outputPA": {...}, "notes": str}
       - We map outputPA -> outputA and leave outputB empty.

    3) Final judge legacy:
       {"final_probs": {...}, "notes": str}
       - We map final_probs -> outputA (as the final) and leave outputB empty.

    Returns (normalized):
      {
        "outputA": Dict[str, float] | None,
        "outputB": Dict[str, float] | None,
        "CRIT_A": float | None,
        "CRIT_B": float | None,
        "notes": str | ""
      }
    """
    out = {"outputA": None, "outputB": None, "CRIT_A": None, "CRIT_B": None, "notes": ""}

    try:
        obj = json.loads(text)
    except Exception:
        return out

    if not isinstance(obj, dict):
        return out

    # Case 1: latest per-round schema
    if "outputA" in obj or "outputB" in obj or "CRIT_A" in obj or "CRIT_B" in obj:
        if isinstance(obj.get("outputA"), dict):
            out["outputA"] = normalize_probs(obj["outputA"], choice_keys)
        if isinstance(obj.get("outputB"), dict):
            out["outputB"] = normalize_probs(obj["outputB"], choice_keys)
        if isinstance(obj.get("CRIT_A"), (int, float)):
            out["CRIT_A"] = float(obj["CRIT_A"])
        if isinstance(obj.get("CRIT_B"), (int, float)):
            out["CRIT_B"] = float(obj["CRIT_B"])
        if isinstance(obj.get("notes"), str):
            out["notes"] = obj["notes"]
        return out

    # Case 2: older alt (outputPA + notes)
    if "outputPA" in obj:
        if isinstance(obj.get("outputPA"), dict):
            out["outputA"] = normalize_probs(obj["outputPA"], choice_keys)
        if isinstance(obj.get("notes"), str):
            out["notes"] = obj["notes"]
        return out

    # Case 3: legacy final (final_probs + notes)
    if "final_probs" in obj:
        if isinstance(obj.get("final_probs"), dict):
            out["outputA"] = normalize_probs(obj["final_probs"], choice_keys)
        if isinstance(obj.get("notes"), str):
            out["notes"] = obj["notes"]
        return out

    return out

# -----------------------------
# Validation helpers for IO
# -----------------------------
def ensure_debater_schema(d: Dict[str, Any], choice_keys: list[str]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Ensure the debater dict contains normalized probs and a full reason dict.
    Returns (probs, reasons).
    """
    probs = normalize_probs(d.get("probs", {}), choice_keys)
    reasons = d.get("reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    # Make sure all keys exist (possibly empty strings)
    reasons = {k: str(reasons.get(k, "") or "").strip() for k in choice_keys}
    return probs, reasons

def ensure_prob_dist(d: Optional[Dict[str, float]], choice_keys: list[str]) -> Dict[str, float]:
    """
    Normalize or fallback to uniform if None.
    """
    if not isinstance(d, dict):
        return {k: 1.0/len(choice_keys) for k in choice_keys}
    return normalize_probs(d, choice_keys)

# -----------------------------
# Pretty serialization (optional)
# -----------------------------
def to_strict_json_output(probs: Dict[str, float], reasons: Dict[str, str], choice_keys: list[str]) -> str:
    """
    Produce the strict debater JSON string:
      {"output": {...}, "reason": {...}}
    """
    obj = {
        "output": {k: float(probs.get(k, 0.0)) for k in choice_keys},
        "reason": {k: str(reasons.get(k, "") or "") for k in choice_keys},
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
