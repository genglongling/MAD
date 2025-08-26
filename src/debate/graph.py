
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .models import LLMFactory
from .prompts import parse_json_or_fallback, normalize_probs
from .metrics import compute_round_metrics
from .crit_rulebased import CRIT, build_index_from_jsonl

class DebateState(dict):
    pass

def _ask(model, system: str, user: str) -> Dict[str, Any]:
    resp = model.invoke([{"role":"system","content":system}, {"role":"user","content":user}])
    data = parse_json_or_fallback(getattr(resp, "content", str(resp)))
    if "probs" in data:
        data["probs"] = normalize_probs(data["probs"])
    else:
        data["probs"] = {"A":0.25,"B":0.25,"C":0.25,"D":0.25}
    return data

def build_graph(cfg_prompts, cfg_models, pairing, with_judge: bool=True):
    A = LLMFactory.make(**cfg_models[pairing]['A'])
    B = LLMFactory.make(**cfg_models[pairing]['B'])
    J = LLMFactory.make(**cfg_models[pairing]['judge'], temperature=0.2) if with_judge else None

    g = StateGraph(DebateState)

    def start_A1(s: DebateState):
        s.setdefault('A', {}); s.setdefault('B', {})
        s['round_metrics'] = []
        facts_path = s.get('facts_jsonl')
        s['facts_index'] = build_index_from_jsonl(facts_path) if facts_path else None
        choices_csv = ", ".join([f"{k}) {v}" for k,v in s['choices'].items()])
        s['A']['r1'] = _ask(A, s['sys_debater'], s['u_r1_A'].format(question=s['question'], choices_csv=choices_csv))
        s.setdefault('crit', {}); s['crit'].setdefault('A', {})
        try: s['crit']['A']['r1'] = CRIT(s['A']['r1'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['A']['r1'] = None
        return s

    def do_B1(s: DebateState):
        s['B']['r1'] = _ask(B, s['sys_debater'], s['u_r1_B'].format(A_json=s['A']['r1']))
        s.setdefault('crit', {}); s['crit'].setdefault('B', {})
        try: s['crit']['B']['r1'] = CRIT(s['B']['r1'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['B']['r1'] = None
        rm = compute_round_metrics(s['A']['r1']['probs'], s['B']['r1']['probs'], None, None,
                                   critA=s['crit']['A']['r1'], critB=s['crit']['B']['r1'])
        s['round_metrics'].append({"round": 1, **rm})
        return s

    def do_A2(s: DebateState):
        s['A']['r2'] = _ask(A, s['sys_debater'], s['u_r2_A'].format(B_json=s['B']['r1']))
        try: s['crit']['A']['r2'] = CRIT(s['A']['r2'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['A']['r2'] = None
        return s

    def do_B2(s: DebateState):
        s['B']['r2'] = _ask(B, s['sys_debater'], s['u_r2_B'].format(A_json=s['A']['r2']))
        try: s['crit']['B']['r2'] = CRIT(s['B']['r2'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['B']['r2'] = None
        rm = compute_round_metrics(s['A']['r2']['probs'], s['B']['r2']['probs'], s['A']['r1']['probs'], s['B']['r1']['probs'],
                                   critA=s['crit']['A']['r2'], critB=s['crit']['B']['r2'])
        s['round_metrics'].append({"round": 2, **rm})
        return s

    def do_A3(s: DebateState):
        s['A']['r3'] = _ask(A, s['sys_debater'], s['u_r3_A'].format(B_json=s['B']['r2']))
        try: s['crit']['A']['r3'] = CRIT(s['A']['r3'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['A']['r3'] = None
        return s

    def do_B3(s: DebateState):
        s['B']['r3'] = _ask(B, s['sys_debater'], s['u_r3_B'].format(A_json=s['A']['r3']))
        try: s['crit']['B']['r3'] = CRIT(s['B']['r3'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['B']['r3'] = None
        rm = compute_round_metrics(s['A']['r3']['probs'], s['B']['r3']['probs'], s['A']['r2']['probs'], s['B']['r2']['probs'],
                                   critA=s['crit']['A']['r3'], critB=s['crit']['B']['r3'])
        s['round_metrics'].append({"round": 3, **rm})
        return s

    def do_A4(s: DebateState):
        s['A']['r4'] = _ask(A, s['sys_debater'], s['u_r4_A'].format(B_json=s['B']['r3']))
        try: s['crit']['A']['r4'] = CRIT(s['A']['r4'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['A']['r4'] = None
        return s

    def do_B4(s: DebateState):
        s['B']['r4'] = _ask(B, s['sys_debater'], s['u_r4_B'].format(A_json=s['A']['r4']))
        try: s['crit']['B']['r4'] = CRIT(s['B']['r4'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['B']['r4'] = None
        rm = compute_round_metrics(s['A']['r4']['probs'], s['B']['r4']['probs'], s['A']['r3']['probs'], s['B']['r3']['probs'],
                                   critA=s['crit']['A']['r4'], critB=s['crit']['B']['r4'])
        s['round_metrics'].append({"round": 4, **rm})
        return s

    def do_A5(s: DebateState):
        s['A']['r5'] = _ask(A, s['sys_debater'], s['u_r5_A'].format(B_json=s['B']['r4']))
        try: s['crit']['A']['r5'] = CRIT(s['A']['r5'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['A']['r5'] = None
        return s

    def do_B5(s: DebateState):
        s['B']['r5'] = _ask(B, s['sys_debater'], s['u_r5_B'].format(A_json=s['A']['r5']))
        try: s['crit']['B']['r5'] = CRIT(s['B']['r5'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['B']['r5'] = None
        rm = compute_round_metrics(s['A']['r5']['probs'], s['B']['r5']['probs'], s['A']['r4']['probs'], s['B']['r4']['probs'],
                                   critA=s['crit']['A']['r5'], critB=s['crit']['B']['r5'])
        s['round_metrics'].append({"round": 5, **rm})
        return s

    def do_A6(s: DebateState):
        s['A']['r6'] = _ask(A, s['sys_debater'], s['u_r6_A'].format(A_json=s['A']['r5'], B_json=s['B']['r5']))
        try: s['crit']['A']['r6'] = CRIT(s['A']['r6'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['A']['r6'] = None
        return s

    def do_B6(s: DebateState):
        s['B']['r6'] = _ask(B, s['sys_debater'], s['u_r6_B'].format(B_json=s['B']['r5'], A_json=s['A']['r5']))
        try: s['crit']['B']['r6'] = CRIT(s['B']['r6'], s['question'], s['choices'], index=s['facts_index'])
        except Exception: s['crit']['B']['r6'] = None
        rm = compute_round_metrics(s['A']['r6']['probs'], s['B']['r6']['probs'], s['A']['r5']['probs'], s['B']['r5']['probs'],
                                   critA=s['crit']['A']['r6'], critB=s['crit']['B']['r6'])
        s['round_metrics'].append({"round": 6, **rm})
        return s

    def judge_node(s: DebateState):
        if J is None:
            fa = s['A']['r6']['probs']; fb = s['B']['r6']['probs']
            fp = {k: 0.5*(fa[k]+fb[k]) for k in fa}
            s['final'] = {"probs": fp, "notes": "mean(A6,B6); judge disabled"}
            return s
        payload = {"question": s['question'], "choices": s['choices'], "A": s['A'], "B": s['B']}
        resp = _ask(J, s['sys_judge'], s['u_judge'].format(payload_json=str(payload)))
        final_probs = resp.get("final_probs", None) or resp.get("probs", None) or s['A']['r6']['probs']
        s['final'] = {"probs": final_probs, "notes": resp.get("notes","")}
        return s

    g.add_node("A1", start_A1)
    g.add_node("B1", do_B1)
    g.add_node("A2", do_A2)
    g.add_node("B2", do_B2)
    g.add_node("A3", do_A3)
    g.add_node("B3", do_B3)
    g.add_node("A4", do_A4)
    g.add_node("B4", do_B4)
    g.add_node("A5", do_A5)
    g.add_node("B5", do_B5)
    g.add_node("A6", do_A6)
    g.add_node("B6", do_B6)
    g.add_node("Judge", judge_node)

    g.set_entry_point("A1")
    g.add_edge("A1","B1")
    g.add_edge("B1","A2"); g.add_edge("A2","B2")
    g.add_edge("B2","A3"); g.add_edge("A3","B3")
    g.add_edge("B3","A4"); g.add_edge("A4","B4")
    g.add_edge("B4","A5"); g.add_edge("A5","B5")
    g.add_edge("B5","A6"); g.add_edge("A6","B6")
    g.add_edge("B6","Judge"); g.add_edge("Judge", END)
    return g
