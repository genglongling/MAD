"""
Information-theoretic metrics for debate rounds on 4-way MC distributions.

Metrics implemented:
- KLD: symmetric KL divergence (J-divergence) in bits
- JSD: Jensen–Shannon divergence in bits
- WD : Wasserstein-1 distance on an ordinal grid A(0),B(1),C(2),D(3)
- MI : Mutual Information via JSD for a 50–50 mixture (in bits)
- H(A), H(B): Shannon entropy (base-2) of each agent's distribution
- IG(A), IG(B): Information gain from previous round (normalized by log(4))
- CE : Cross-Entropy H(PA, PB) (optional, not part of the "8", but available)
- AvgCRIT: passthrough of average CRIT score if provided

All probabilities are normalized defensively; zero-safety clamps avoid log(0).
"""
from typing import Dict, Optional
import math

KEYS = ["A","B","C","D"]

def _vec(p: Dict[str, float]):
    return [max(0.0, float(p.get(k, 0.0))) for k in KEYS]

def _normalize(v):
    s = sum(v) or 1.0
    return [x/s for x in v]

def normalize_probs(p: Dict[str, float]) -> Dict[str, float]:
    v = _normalize(_vec(p))
    return {k: v[i] for i,k in enumerate(KEYS)}

def entropy(p: Dict[str,float], base: float=2.0) -> float:
    v = _normalize(_vec(p))
    s = 0.0
    for x in v:
        if x > 0:
            s -= x * math.log(x, base)
    return s

def info_gain(curr: Dict[str,float], prev: Optional[Dict[str,float]], base: float=2.0) -> Optional[float]:
    if prev is None:
        return None
    Hprev = entropy(prev, base=base)
    Hcurr = entropy(curr, base=base)
    denom = math.log(len(KEYS), base)  # log |Y| with |Y|=4
    return (Hprev - Hcurr) / denom if denom > 0 else None

def kl_div(p: Dict[str,float], q: Dict[str,float], base: float=2.0) -> float:
    P = _normalize(_vec(p)); Q = _normalize(_vec(q))
    eps = 1e-12
    s = 0.0
    for i in range(len(P)):
        if P[i] > 0:
            s += P[i] * math.log(P[i]/max(Q[i], eps), base)
    return s

def sym_kl(p: Dict[str,float], q: Dict[str,float], base: float=2.0) -> float:
    return 0.5*(kl_div(p,q,base=base) + kl_div(q,p,base=base))

def js_divergence(p: Dict[str,float], q: Dict[str,float], base: float=2.0) -> float:
    P = _normalize(_vec(p)); Q = _normalize(_vec(q))
    M = [(P[i]+Q[i])/2.0 for i in range(len(P))]
    def _kl(A,B):
        eps = 1e-12
        s = 0.0
        for i in range(len(A)):
            if A[i] > 0:
                s += A[i] * math.log(A[i]/max(B[i], eps), base)
        return s
    return 0.5*_kl(P,M) + 0.5*_kl(Q,M)

def wasserstein_1d(p: Dict[str,float], q: Dict[str,float]) -> float:
    """Earth Mover's Distance on ordered categories A=0,B=1,C=2,D=3."""
    P = _normalize(_vec(p)); Q = _normalize(_vec(q))
    cumP = 0.0; cumQ = 0.0; w1 = 0.0
    for i in range(len(P)):
        cumP += P[i]; cumQ += Q[i]
        w1 += abs(cumP - cumQ)
    return w1

def cross_entropy(p: Dict[str,float], q: Dict[str,float], base: float=2.0) -> float:
    """H(P,Q) = -sum_x P(x) log Q(x)."""
    P = _normalize(_vec(p)); Q = _normalize(_vec(q))
    eps = 1e-12
    s = 0.0
    for i in range(len(P)):
        if P[i] > 0:
            s -= P[i] * math.log(max(Q[i], eps), base)
    return s

def mutual_information_via_jsd(p: Dict[str,float], q: Dict[str,float], base: float=2.0) -> float:
    """For a 50-50 mixture of P and Q, I(S;X) = JSD(P||Q) (in bits for base=2)."""
    return js_divergence(p,q,base=base)

def compute_round_metrics(
    PA: Dict[str,float],
    PB: Dict[str,float],
    prev_PA: Optional[Dict[str,float]]=None,
    prev_PB: Optional[Dict[str,float]]=None,
    critA: Optional[float]=None,
    critB: Optional[float]=None,
    include_ce: bool=False
) -> Dict[str, float]:
    """Compute the 8 standard metrics (+ optional CE) and AvgCRIT.
    Returns keys: KLD, JSD, WD, MI, H(A), IG(A), H(B), IG(B), [CE], AvgCRIT
    """
    # Divergences & distances
    kld = sym_kl(PA, PB, base=2.0)
    jsd = js_divergence(PA, PB, base=2.0)
    mi  = mutual_information_via_jsd(PA, PB, base=2.0)
    wd  = wasserstein_1d(PA, PB)

    # Entropies & info gain
    Ha = entropy(PA, base=2.0)
    Hb = entropy(PB, base=2.0)
    IGa = info_gain(PA, prev_PA, base=2.0)
    IGb = info_gain(PB, prev_PB, base=2.0)

    out = {
        "KLD": kld,
        "JSD": jsd,
        "WD": wd,
        "MI": mi,
        "H(A)": Ha,
        "IG(A)": IGa if IGa is not None else float("nan"),
        "H(B)": Hb,
        "IG(B)": IGb if IGb is not None else float("nan"),
        "AvgCRIT": float("nan")
    }

    if include_ce:
        out["CE"] = cross_entropy(PA, PB, base=2.0)

    # AvgCRIT passthrough (if caller computed CRIT per agent this round)
    vals = [v for v in (critA, critB) if v is not None]
    if vals:
        out["AvgCRIT"] = sum(vals)/len(vals)

    return out
