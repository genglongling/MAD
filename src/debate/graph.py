
from typing import Dict, Any
import json
from langgraph.graph import StateGraph, END
from .models import LLMFactory
from .prompts import parse_json_or_fallback, normalize_probs, get_choice_keys, parse_judge_json
from .metrics import compute_round_metrics

class DebateState(dict):
    pass

def _ask(model, system: str, user: str, choice_keys: list[str]) -> Dict[str, Any]:
    print(f"MAKING API CALL...")
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            resp = model.invoke([{"role":"system","content":system}, {"role":"user","content":user}])
            print(f"API RESPONSE OBJECT: {resp}")
            content = getattr(resp, "content", str(resp))
            print(f"RAW LLM RESPONSE: {content}")
            print(f"RESPONSE TYPE: {type(content)}")
            print(f"RESPONSE LENGTH: {len(content) if content else 0}")
            
            # Check if we got a valid response
            if content and len(content.strip()) > 0:
                break
            else:
                print(f"Empty response on attempt {attempt + 1}, retrying...")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                
        except Exception as e:
            print(f"API CALL FAILED on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            content = ""
    
    data = parse_json_or_fallback(content, choice_keys)
    print(f"PARSED DATA: {json.dumps(data, indent=2)}")
    # The parse_json_or_fallback function already handles the new schema and returns "probs"
    # We just need to ensure it exists
    if "probs" not in data:
        data["probs"] = {k: 1.0/len(choice_keys) for k in choice_keys}
    return data

def _ask_judge(model, system: str, user: str, choice_keys: list[str]) -> Dict[str, Any]:
    print(f"MAKING JUDGE API CALL...")
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            resp = model.invoke([{"role":"system","content":system}, {"role":"user","content":user}])
            print(f"JUDGE API RESPONSE OBJECT: {resp}")
            content = getattr(resp, "content", str(resp))
            print(f"JUDGE RAW LLM RESPONSE: {content}")
            print(f"JUDGE RESPONSE TYPE: {type(content)}")
            print(f"JUDGE RESPONSE LENGTH: {len(content) if content else 0}")
            
            # Check if we got a valid response
            if content and len(content.strip()) > 0:
                break
            else:
                print(f"Empty judge response on attempt {attempt + 1}, retrying...")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                
        except Exception as e:
            print(f"JUDGE API CALL FAILED on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            content = ""
    
    data = parse_judge_json(content, choice_keys)
    print(f"JUDGE PARSED DATA: {json.dumps(data, indent=2)}")
    return data

def build_graph(cfg_prompts, cfg_models, pairing, with_judge: bool=True):
    # Handle both old and new config structures
    if 'pairings' in cfg_models:
        pairing_config = cfg_models['pairings'][pairing]
    else:
        pairing_config = cfg_models[pairing]
    
    A = LLMFactory.make(**pairing_config['A'])
    B = LLMFactory.make(**pairing_config['B'])
    J = LLMFactory.make(**pairing_config['judge']) if with_judge else None

    g = StateGraph(DebateState)

    def start_A1(s: DebateState):
        s.setdefault('A', {}); s.setdefault('B', {})
        s['round_metrics'] = []
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        # Create dynamic JSON examples
        choice_dict = ", ".join([f'"{k}": p{k}' for k in choice_keys])
        reason_dict = ", ".join([f'"{k}": r{k}' for k in choice_keys])
        # Escape curly braces in choices to prevent format string issues
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r1_A'].format(question=s['question'], choices_csv=choices_csv, choice_dict=choice_dict, reason_dict=reason_dict)
        print(f"\n=== ROUND 1 - AGENT A ===")
        print(f"Question: {s['question']}")
        print(f"Choices: {choices_csv}")
        print(f"Prompt: {prompt}")
        s['A']['r1'] = _ask(A, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['A']['r1'], indent=2)}")
        return s

    def judge_r1(s: DebateState):
        if J is None:
            s.setdefault('crit', {}); s['crit'].setdefault('A', {}); s['crit'].setdefault('B', {})
            s['crit']['A']['r1'] = None; s['crit']['B']['r1'] = None
            return s
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        # Create dynamic JSON examples
        choice_dict = ", ".join([f'"{k}": float' for k in choice_keys])
        uniform_prob = 1.0 / len(choice_keys)
        uniform_dict = ", ".join([f'"{k}": {uniform_prob}' for k in choice_keys])
        # Escape curly braces in choices to prevent format string issues
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_judge_r1'].format(
            question=s['question'], 
            choices_csv=choices_csv,
            A_output_json_r1=json.dumps(s['A']['r1']['probs']),
            A_reason_json_r1=json.dumps(s['A']['r1'].get('reasons', {})),
            B_output_json_r1=json.dumps(s['B']['r1']['probs']),
            B_reason_json_r1=json.dumps(s['B']['r1'].get('reasons', {})),
            judge_crit_instructions=s['judge_crit_instructions'],
            choice_dict=choice_dict,
            uniform_dict=uniform_dict
        )
        print(f"\n=== ROUND 1 - JUDGE ===")
        print(f"Prompt: {prompt}")
        resp = _ask_judge(J, s['sys_judge'], prompt, choice_keys)
        print(f"Response: {json.dumps(resp, indent=2)}")
        s.setdefault('crit', {}); s['crit'].setdefault('A', {}); s['crit'].setdefault('B', {})
        s['crit']['A']['r1'] = resp.get('CRIT_A', None)
        s['crit']['B']['r1'] = resp.get('CRIT_B', None)
        rm = compute_round_metrics(s['A']['r1']['probs'], s['B']['r1']['probs'], None, None,
                                   critA=s['crit']['A']['r1'], critB=s['crit']['B']['r1'])
        s['round_metrics'].append({"round": 1, **rm})
        return s

    def do_B1(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        # Create dynamic JSON examples
        choice_dict = ", ".join([f'"{k}": p{k}' for k in choice_keys])
        reason_dict = ", ".join([f'"{k}": r{k}' for k in choice_keys])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r1_B'].format(question=s['question'], choices_csv=choices_csv, A_json=s['A']['r1'], choice_dict=choice_dict, reason_dict=reason_dict)
        print(f"\n=== ROUND 1 - AGENT B ===")
        print(f"Prompt: {prompt}")
        s['B']['r1'] = _ask(B, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['B']['r1'], indent=2)}")
        return s

    def do_A2(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        # Create dynamic JSON examples
        choice_dict = ", ".join([f'"{k}": p{k}' for k in choice_keys])
        reason_dict = ", ".join([f'"{k}": r{k}' for k in choice_keys])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r2_A'].format(question=s['question'], choices_csv=choices_csv, B_json=s['B']['r1'], choice_dict=choice_dict, reason_dict=reason_dict)
        print(f"\n=== ROUND 2 - AGENT A ===")
        print(f"Prompt: {prompt}")
        s['A']['r2'] = _ask(A, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['A']['r2'], indent=2)}")
        return s

    def judge_r2(s: DebateState):
        if J is None:
            s['crit']['A']['r2'] = None; s['crit']['B']['r2'] = None
            return s
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        # Create dynamic JSON examples
        choice_dict = ", ".join([f'"{k}": float' for k in choice_keys])
        uniform_prob = 1.0 / len(choice_keys)
        uniform_dict = ", ".join([f'"{k}": {uniform_prob}' for k in choice_keys])
        choices_csv = ", ".join([f"{k}) {v}" for k,v in s['choices'].items()])
        resp = _ask_judge(J, s['sys_judge'], s['u_judge_r2'].format(
            question=s['question'], 
            choices_csv=choices_csv,
            A_output_json_r2=json.dumps(s['A']['r2']['probs']),
            A_reason_json_r2=json.dumps(s['A']['r2'].get('reasons', {})),
            B_output_json_r2=json.dumps(s['B']['r2']['probs']),
            B_reason_json_r2=json.dumps(s['B']['r2'].get('reasons', {})),
            judge_crit_instructions=s['judge_crit_instructions'],
            choice_dict=choice_dict,
            uniform_dict=uniform_dict
        ), choice_keys)
        s['crit']['A']['r2'] = resp.get('CRIT_A', None)
        s['crit']['B']['r2'] = resp.get('CRIT_B', None)
        rm = compute_round_metrics(s['A']['r2']['probs'], s['B']['r2']['probs'], s['A']['r2']['probs'], s['B']['r1']['probs'],
                                   critA=s['crit']['A']['r2'], critB=s['crit']['B']['r2'])
        s['round_metrics'].append({"round": 2, **rm})
        return s

    def do_B2(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        # Create dynamic JSON examples
        choice_dict = ", ".join([f'"{k}": p{k}' for k in choice_keys])
        reason_dict = ", ".join([f'"{k}": r{k}' for k in choice_keys])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r2_B'].format(question=s['question'], choices_csv=choices_csv, A_json=s['A']['r2'], choice_dict=choice_dict, reason_dict=reason_dict)
        s['B']['r2'] = _ask(B, s['sys_debater'], prompt, choice_keys)
        return s

    def do_A3(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r3_A'].format(question=s['question'], choices_csv=choices_csv, B_json=s['B']['r2'])
        print(f"\n=== ROUND 3 - AGENT A ===")
        print(f"Prompt: {prompt}")
        s['A']['r3'] = _ask(A, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['A']['r3'], indent=2)}")
        return s

    def judge_r3(s: DebateState):
        if J is None:
            s['crit']['A']['r3'] = None; s['crit']['B']['r3'] = None
            return s
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {v}" for k,v in s['choices'].items()])
        prompt = s['u_judge_r3'].format(
            question=s['question'], 
            choices_csv=choices_csv,
            A_output_json_r3=json.dumps(s['A']['r3']['probs']),
            A_reason_json_r3=json.dumps(s['A']['r3'].get('reasons', {})),
            B_output_json_r3=json.dumps(s['B']['r3']['probs']),
            B_reason_json_r3=json.dumps(s['B']['r3'].get('reasons', {})),
            judge_crit_instructions=s['judge_crit_instructions']
        )
        print(f"\n=== ROUND 3 - JUDGE ===")
        print(f"Prompt: {prompt}")
        resp = _ask_judge(J, s['sys_judge'], prompt, choice_keys)
        print(f"Response: {json.dumps(resp, indent=2)}")
        s['crit']['A']['r3'] = resp.get('CRIT_A', None)
        s['crit']['B']['r3'] = resp.get('CRIT_B', None)
        rm = compute_round_metrics(s['A']['r3']['probs'], s['B']['r3']['probs'], s['A']['r2']['probs'], s['B']['r2']['probs'],
                                   critA=s['crit']['A']['r3'], critB=s['crit']['B']['r3'])
        s['round_metrics'].append({"round": 3, **rm})
        return s

    def do_B3(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r3_B'].format(question=s['question'], choices_csv=choices_csv, A_json=s['A']['r3'])
        print(f"\n=== ROUND 3 - AGENT B ===")
        print(f"Prompt: {prompt}")
        s['B']['r3'] = _ask(B, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['B']['r3'], indent=2)}")
        return s

    def do_A4(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r4_A'].format(question=s['question'], choices_csv=choices_csv, B_json=s['B']['r3'])
        print(f"\n=== ROUND 4 - AGENT A ===")
        print(f"Prompt: {prompt}")
        s['A']['r4'] = _ask(A, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['A']['r4'], indent=2)}")
        return s

    def judge_r4(s: DebateState):
        if J is None:
            s['crit']['A']['r4'] = None; s['crit']['B']['r4'] = None
            return s
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {v}" for k,v in s['choices'].items()])
        prompt = s['u_judge_r4'].format(
            question=s['question'], 
            choices_csv=choices_csv,
            A_output_json_r4=json.dumps(s['A']['r4']['probs']),
            A_reason_json_r4=json.dumps(s['A']['r4'].get('reasons', {})),
            B_output_json_r4=json.dumps(s['B']['r4']['probs']),
            B_reason_json_r4=json.dumps(s['B']['r4'].get('reasons', {})),
            judge_crit_instructions=s['judge_crit_instructions']
        )
        print(f"\n=== ROUND 4 - JUDGE ===")
        print(f"Prompt: {prompt}")
        resp = _ask_judge(J, s['sys_judge'], prompt, choice_keys)
        print(f"Response: {json.dumps(resp, indent=2)}")
        s['crit']['A']['r4'] = resp.get('CRIT_A', None)
        s['crit']['B']['r4'] = resp.get('CRIT_B', None)
        rm = compute_round_metrics(s['A']['r4']['probs'], s['B']['r4']['probs'], s['A']['r3']['probs'], s['B']['r3']['probs'],
                                   critA=s['crit']['A']['r4'], critB=s['crit']['B']['r4'])
        s['round_metrics'].append({"round": 4, **rm})
        return s

    def do_B4(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r4_B'].format(question=s['question'], choices_csv=choices_csv, A_json=s['A']['r4'])
        print(f"\n=== ROUND 4 - AGENT B ===")
        print(f"Prompt: {prompt}")
        s['B']['r4'] = _ask(B, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['B']['r4'], indent=2)}")
        return s

    def do_A5(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r5_A'].format(question=s['question'], choices_csv=choices_csv, B_json=s['B']['r4'])
        print(f"\n=== ROUND 5 - AGENT A ===")
        print(f"Prompt: {prompt}")
        s['A']['r5'] = _ask(A, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['A']['r5'], indent=2)}")
        return s

    def judge_r5(s: DebateState):
        if J is None:
            s['crit']['A']['r5'] = None; s['crit']['B']['r5'] = None
            return s
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {v}" for k,v in s['choices'].items()])
        prompt = s['u_judge_r5'].format(
            question=s['question'], 
            choices_csv=choices_csv,
            A_output_json_r5=json.dumps(s['A']['r5']['probs']),
            A_reason_json_r5=json.dumps(s['A']['r5'].get('reasons', {})),
            B_output_json_r5=json.dumps(s['B']['r5']['probs']),
            B_reason_json_r5=json.dumps(s['B']['r5'].get('reasons', {})),
            judge_crit_instructions=s['judge_crit_instructions']
        )
        print(f"\n=== ROUND 5 - JUDGE ===")
        print(f"Prompt: {prompt}")
        resp = _ask_judge(J, s['sys_judge'], prompt, choice_keys)
        print(f"Response: {json.dumps(resp, indent=2)}")
        s['crit']['A']['r5'] = resp.get('CRIT_A', None)
        s['crit']['B']['r5'] = resp.get('CRIT_B', None)
        rm = compute_round_metrics(s['A']['r5']['probs'], s['B']['r5']['probs'], s['A']['r4']['probs'], s['B']['r4']['probs'],
                                   critA=s['crit']['A']['r5'], critB=s['crit']['B']['r5'])
        s['round_metrics'].append({"round": 5, **rm})
        return s

    def do_B5(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r5_B'].format(question=s['question'], choices_csv=choices_csv, A_json=s['A']['r5'])
        print(f"\n=== ROUND 5 - AGENT B ===")
        print(f"Prompt: {prompt}")
        s['B']['r5'] = _ask(B, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['B']['r5'], indent=2)}")
        return s

    def do_A6(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r6_A'].format(question=s['question'], choices_csv=choices_csv, A_json=s['A']['r5'], B_json=s['B']['r5'])
        print(f"\n=== ROUND 6 (FINAL) - AGENT A ===")
        print(f"Prompt: {prompt}")
        s['A']['r6'] = _ask(A, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['A']['r6'], indent=2)}")
        return s

    def judge_r6(s: DebateState):
        if J is None:
            s['crit']['A']['r6'] = None; s['crit']['B']['r6'] = None
            return s
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {v}" for k,v in s['choices'].items()])
        prompt = s['u_judge_r6'].format(
            question=s['question'], 
            choices_csv=choices_csv,
            A_output_json_r6=json.dumps(s['A']['r6']['probs']),
            A_reason_json_r6=json.dumps(s['A']['r6'].get('reasons', {})),
            B_output_json_r6=json.dumps(s['B']['r6']['probs']),
            B_reason_json_r6=json.dumps(s['B']['r6'].get('reasons', {})),
            judge_crit_instructions=s['judge_crit_instructions']
        )
        print(f"\n=== ROUND 6 (FINAL) - JUDGE ===")
        print(f"Prompt: {prompt}")
        resp = _ask_judge(J, s['sys_judge'], prompt, choice_keys)
        print(f"Response: {json.dumps(resp, indent=2)}")
        s['crit']['A']['r6'] = resp.get('CRIT_A', None)
        s['crit']['B']['r6'] = resp.get('CRIT_B', None)
        rm = compute_round_metrics(s['A']['r6']['probs'], s['B']['r6']['probs'], s['A']['r5']['probs'], s['B']['r5']['probs'],
                                   critA=s['crit']['A']['r6'], critB=s['crit']['B']['r6'])
        s['round_metrics'].append({"round": 6, **rm})
        return s

    def do_B6(s: DebateState):
        # Get choice keys dynamically
        choice_keys = get_choice_keys(s['choices'])
        choices_csv = ", ".join([f"{k}) {str(v).replace('{', '{{').replace('}', '}}')}" for k,v in s['choices'].items()])
        prompt = s['u_r6_B'].format(question=s['question'], choices_csv=choices_csv, B_json=s['B']['r5'], A_json=s['A']['r5'])
        print(f"\n=== ROUND 6 (FINAL) - AGENT B ===")
        print(f"Prompt: {prompt}")
        s['B']['r6'] = _ask(B, s['sys_debater'], prompt, choice_keys)
        print(f"Response: {json.dumps(s['B']['r6'], indent=2)}")
        return s

    def final_judge(s: DebateState):
        if J is None:
            fa = s['A']['r6']['probs']; fb = s['B']['r6']['probs']
            fp = {k: 0.5*(fa[k]+fb[k]) for k in fa}
            s['final'] = {"probs": fp, "notes": "mean(A6,B6); judge disabled"}
            return s
        # Use the final round probabilities and CRIT scores for final judgment
        fa = s['A']['r6']['probs']; fb = s['B']['r6']['probs']
        fp = {k: 0.5*(fa[k]+fb[k]) for k in fa}
        s['final'] = {"probs": fp, "notes": f"mean(A6,B6); CRIT_A={s['crit']['A']['r6']}, CRIT_B={s['crit']['B']['r6']}"}
        return s

    # Add all nodes
    g.add_node("A1", start_A1)
    g.add_node("Judge1", judge_r1)
    g.add_node("B1", do_B1)
    g.add_node("A2", do_A2)
    g.add_node("Judge2", judge_r2)
    g.add_node("B2", do_B2)
    g.add_node("A3", do_A3)
    g.add_node("Judge3", judge_r3)
    g.add_node("B3", do_B3)
    g.add_node("A4", do_A4)
    g.add_node("Judge4", judge_r4)
    g.add_node("B4", do_B4)
    g.add_node("A5", do_A5)
    g.add_node("Judge5", judge_r5)
    g.add_node("B5", do_B5)
    g.add_node("A6", do_A6)
    g.add_node("Judge6", judge_r6)
    g.add_node("B6", do_B6)
    g.add_node("FinalJudge", final_judge)

    # Set up the graph flow
    g.set_entry_point("A1")
    g.add_edge("A1", "B1")
    g.add_edge("B1", "Judge1")
    g.add_edge("Judge1", "A2")
    g.add_edge("A2", "B2")
    g.add_edge("B2", "Judge2")
    g.add_edge("Judge2", "A3")
    g.add_edge("A3", "B3")
    g.add_edge("B3", "Judge3")
    g.add_edge("Judge3", "A4")
    g.add_edge("A4", "B4")
    g.add_edge("B4", "Judge4")
    g.add_edge("Judge4", "A5")
    g.add_edge("A5", "B5")
    g.add_edge("B5", "Judge5")
    g.add_edge("Judge5", "A6")
    g.add_edge("A6", "B6")
    g.add_edge("B6", "Judge6")
    g.add_edge("Judge6", "FinalJudge")
    g.add_edge("FinalJudge", END)
    
    return g
