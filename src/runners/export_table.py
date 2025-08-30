"""
Export benchmark results to LaTeX tables
"""
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load benchmark results"""
    with open(results_path, 'r') as f:
        return json.load(f)

def calculate_accuracy(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Calculate accuracy metrics from results"""
    data = []
    
    for result in results:
        if 'error' in result:
            continue
            
        pairing = result['pairing']
        example_id = result['example_id']
        answer = result.get('answer', '')
        
        # Extract final probabilities from debate state
        debate_state = result.get('debate_state', {})
        if 'A' in debate_state and 'r6' in debate_state['A']:
            final_probs = debate_state['A']['r6'].get('output', {})
        elif 'B' in debate_state and 'r6' in debate_state['B']:
            final_probs = debate_state['B']['r6'].get('output', {})
        else:
            continue
        
        # Find predicted answer
        if final_probs:
            predicted = max(final_probs, key=final_probs.get)
            correct = 1 if predicted == answer else 0
        else:
            continue
        
        data.append({
            'pairing': pairing,
            'example_id': example_id,
            'correct': correct,
            'answer': answer,
            'predicted': predicted
        })
    
    df = pd.DataFrame(data)
    
    # Calculate accuracy by pairing
    accuracy_df = df.groupby('pairing')['correct'].agg(['mean', 'count']).reset_index()
    accuracy_df.columns = ['pairing', 'accuracy', 'count']
    accuracy_df['accuracy'] = accuracy_df['accuracy'] * 100  # Convert to percentage
    
    return accuracy_df

def calculate_round_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Calculate per-round information-theoretic metrics"""
    data = []
    
    for result in results:
        if 'error' in result:
            continue
            
        pairing = result['pairing']
        debate_state = result.get('debate_state', {})
        
        # Extract round metrics
        round_metrics = debate_state.get('round_metrics', [])
        for round_data in round_metrics:
            round_num = round_data.get('round', 0)
            
            # Extract metrics
            metrics = {
                'pairing': pairing,
                'round': round_num,
                'KLD': round_data.get('KLD', np.nan),
                'JSD': round_data.get('JSD', np.nan),
                'WD': round_data.get('WD', np.nan),
                'MI': round_data.get('MI', np.nan),
                'H(A)': round_data.get('H_A', np.nan),
                'IG(A)': round_data.get('IG_A', np.nan),
                'H(B)': round_data.get('H_B', np.nan),
                'IG(B)': round_data.get('IG_B', np.nan),
                'AvgCRIT': round_data.get('AvgCRIT', np.nan),
            }
            data.append(metrics)
    
    df = pd.DataFrame(data)
    
    # Calculate mean metrics by pairing and round
    if not df.empty:
        metrics_df = df.groupby(['pairing', 'round']).mean().reset_index()
    else:
        metrics_df = pd.DataFrame()
    
    return metrics_df

def export_latex_accuracy(accuracy_df: pd.DataFrame, outfile: str, caption: str, label: str):
    """Export accuracy table to LaTeX"""
    # Pivot table for better formatting
    pivot_df = accuracy_df.pivot(index='pairing', columns=None, values='accuracy')
    
    latex_table = pivot_df.to_latex(
        float_format='%.1f',
        caption=caption,
        label=label,
        index=True,
        escape=False
    )
    
    # Save to file
    with open(outfile, 'w') as f:
        f.write(latex_table)
    
    print(f"Accuracy table saved to: {outfile}")

def export_latex_metrics(metrics_df: pd.DataFrame, outfile: str, caption: str, label: str, 
                        metrics: List[str]):
    """Export round metrics table to LaTeX"""
    if metrics_df.empty:
        print("No metrics data available")
        return
    
    # Filter to requested metrics
    available_metrics = [m for m in metrics if m in metrics_df.columns]
    if not available_metrics:
        print("No requested metrics found in data")
        return
    
    # Pivot table
    pivot_df = metrics_df.pivot(index='pairing', columns='round', values=available_metrics[0])
    
    latex_table = pivot_df.to_latex(
        float_format='%.3f',
        caption=caption,
        label=label,
        index=True,
        escape=False
    )
    
    # Save to file
    with open(outfile, 'w') as f:
        f.write(latex_table)
    
    print(f"Metrics table saved to: {outfile}")

def main():
    parser = argparse.ArgumentParser(description="Export benchmark results to LaTeX tables")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark config")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    
    args = parser.parse_args()
    
    # Load configurations and results
    benchmark_config = load_config(args.benchmark)
    results = load_results(args.results)
    
    # Create output directory
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Export accuracy table
    if benchmark_config['export']['latex_accuracy']['enabled']:
        accuracy_df = calculate_accuracy(results)
        if not accuracy_df.empty:
            export_latex_accuracy(
                accuracy_df,
                benchmark_config['export']['latex_accuracy']['outfile'],
                benchmark_config['export']['latex_accuracy']['caption'],
                benchmark_config['export']['latex_accuracy']['label']
            )
    
    # Export metrics table
    if benchmark_config['export']['latex_metrics']['enabled']:
        metrics_df = calculate_round_metrics(results)
        if not metrics_df.empty:
            export_latex_metrics(
                metrics_df,
                benchmark_config['export']['latex_metrics']['outfile'],
                benchmark_config['export']['latex_metrics']['caption'],
                benchmark_config['export']['latex_metrics']['label'],
                benchmark_config['export']['latex_metrics']['metrics']
            )

if __name__ == "__main__":
    main()
