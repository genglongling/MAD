
from typing import List, Dict, Tuple, Optional
import re, json

# Tokenization and helpers
STOP = set("the a an is are was were be been being to of in on at by for from with and or as that this these those which who whom whose".split())
HEDGES = set("maybe might could possibly perhaps around roughly approximately".split())
SUPPORT_CUES = r"\b(because|since|due to|thus|therefore|hence|so)\b"
ELIMIN_CUES  = r"\b(not|cannot|isn['’]t|no|except|ruled out|eliminate|unlikely)\b"

def tokenize(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())

def sent_split(s: str) -> List[str]:
    return [t.strip() for t in re.split(r"[.?!]\s+", s) if t.strip()]

def overlap(a_tokens, b_tokens):
    A = set(a_tokens) - STOP; B = set(b_tokens) - STOP
    if not A or not B:
        return 0.0
    return len(A & B) / (len(A | B) or 1.0)

# BM25 (optional)
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

class SymbolicIndex:
    def __init__(self, docs: List[Dict[str,str]]):
        self.docs = docs
        self.tokens = [tokenize(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(self.tokens) if BM25Okapi else None
    def search(self, query: str, k: int = 5) -> List[Dict]:
        toks = tokenize(query)
        if self.bm25 is None:
            # Jaccard fallback
            Q = set(toks)
            scores = [len(Q & set(T))/float(len(Q | set(T)) or 1) for T in self.tokens]
        else:
            scores = self.bm25.get_scores(toks)
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"id": self.docs[i].get("id", str(i)), "text": self.docs[i]["text"], "score": float(scores[i])} for i in idx]

def build_index_from_jsonl(path: str) -> Optional[SymbolicIndex]:
    docs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "text" in obj:
                    docs.append({"id": obj.get("id", str(len(docs))), "text": obj["text"]})
    except Exception:
        return None
    if not docs:
        return None
    return SymbolicIndex(docs)

# CLAIM: extract main claim and atomic reasons
def split_atomic(sentences: List[str]) -> List[str]:
    out = []
    for s in sentences:
        parts = re.split(r"(?:;|,|\b(?:because|since|therefore|thus|hence)\b)", s, flags=re.I)
        out += [p.strip() for p in parts if p.strip()]
    return out

def CLAIM(message: Dict) -> Tuple[str, List[Dict]]:
    Omega = max(message["probs"], key=message["probs"].get)
    rationale = message.get("rationale", "")
    sents = sent_split(rationale)
    atoms = split_atomic(sents)
    R = []
    for a in atoms:
        t = "support" if re.search(SUPPORT_CUES, a, re.I) else ("eliminate" if re.search(ELIMIN_CUES, a, re.I) else "support")
        target = Omega
        for k, v in message["choices"].items():
            if re.search(rf"\b({re.escape(k)}|{re.escape(v)})\b", a, re.I):
                target = k
                break
        R.append({"text": a, "type": t, "target": target})
    return Omega, R

def FINDDOC(reason_text: str, question: str, choices: Dict[str,str], index: Optional[SymbolicIndex], k: int = 5):
    if index is None:
        return []
    query = f"{reason_text} {question} {' '.join(choices.values())}"
    return index.search(query, k=k)

def VALIDATE(reason: Dict, Omega: str, choices: Dict[str,str], docs: Optional[List[Dict]] = None):
    r_tok = tokenize(reason["text"])
    tgt_choice = choices[reason["target"]]
    tgt_tok = tokenize(tgt_choice)

    if reason["type"] == "support":
        align = overlap(r_tok, tgt_tok)
    else:
        align = overlap(r_tok, tgt_tok)
        if not re.search(ELIMIN_CUES, reason["text"], re.I):
            align *= 0.6

    fact_boost = 0.0
    if docs:
        top = max(docs, key=lambda d: d["score"])
        dtok = tokenize(top["text"])
        fact_boost = 0.3 * min(1.0, overlap(r_tok, dtok) + overlap(tgt_tok, dtok))

    contra = 0.0
    if reason["type"] == "support" and re.search(ELIMIN_CUES, reason["text"], re.I):
        contra = 0.3

    gamma = max(0.0, min(1.0, 0.6*align + fact_boost - contra))

    n = len(r_tok)
    hedge_pen = 0.15 if any(t in HEDGES for t in r_tok) else 0.0
    length_pen = 0.0 if n <= 30 else min(0.3, 0.01*(n-30))
    theta = max(0.0, 1.0 - hedge_pen - length_pen)

    return gamma, theta

def _agg(pairs):
    if not pairs:
        return 0.0
    num = sum(g*t for g,t in pairs)
    den = sum(max(1e-9, t) for _,t in pairs)
    return num/den

def CRIT(message: Dict, question: str, choices: Dict[str,str], index: Optional[SymbolicIndex]=None, lambda_counter: float=1.0) -> float:
    Omega, R = CLAIM(message)
    R_prime = [r for r in R if r["type"]=="eliminate" or r["target"]!=Omega]
    R_supp  = [r for r in R if r["type"]=="support" and r["target"]==Omega]

    scored_R = []
    for r in R_supp:
        docs = FINDDOC(r["text"], question, choices, index) if index else None
        scored_R.append(VALIDATE(r, Omega, choices, docs))

    scored_Rp = []
    for r in R_prime:
        docs = FINDDOC(r["text"], question, choices, index) if index else None
        scored_Rp.append(VALIDATE(r, Omega, choices, docs))

    pos = _agg(scored_R)
    neg = _agg(scored_Rp)
    Gamma = max(0.0, min(1.0, pos - lambda_counter*neg))
    return Gamma
