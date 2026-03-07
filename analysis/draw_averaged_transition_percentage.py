import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


TASKS = [
    "sakura_animal-r-ja-pr", "sakura_emotion-r-ja-pr", 
    "sakura_gender-r-ja-pr", "sakura_language-r-ja-pr", 
    "mmau-test-mini-r-ja-pr", "mmar-r-ja-pr"
]
TASKS_ABBV = {
    "sakura_animal-r-ja-pr": "Sakura Animal", 
    "sakura_emotion-r-ja-pr": "Sakura Emotion", 
    "sakura_gender-r-ja-pr": "Sakura Gender", 
    "sakura_language-r-ja-pr": "Sakura Language", 
    "mmau-test-mini-r-ja-pr": "MMAU", 
    "mmar-r-ja-pr": "MMAR",
}

METHODS = ["aad", "acd"]
MODELS = ["desta", "qwen", "af3"]

MODEL_NAME_MAP = {
    "desta": "DeSTA2.5-Audio",
    "qwen": "Qwen2.5-Omni",
    "af3": "Audio Flamingo 3"
}

STATES_WRONG_KEYS = [
    "W_Hallucinated_No_Audio",
    "W_Reasoning_but_Wrong_Answer", 
    "W_Direct_Assertive_Wrong_Answer", 
    "W_Guessing_Wrong_or_Refusal"
]

STATES_CORRECT_KEYS = [
    "Correct_Response",
]

LABELS_X = ["W_NoAudio", "W_Reason", "W_Direct", "W_Guess", "Correct"]
LABELS_Y = ["W_NoAudio", "W_Reason", "W_Direct", "W_Guess"]

STATE_MAP = {}
STATE_MAP.update({k: v for k, v in zip(STATES_WRONG_KEYS, LABELS_Y)})
for k in STATES_CORRECT_KEYS:
    STATE_MAP[k] = "Correct"

def load_states(filepath):
    states = {}
    if not os.path.exists(filepath):
        return states
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                raw_state = entry.get('state', '')
                if raw_state in STATE_MAP:
                    states[entry['id']] = STATE_MAP[raw_state]
            except: pass
    return states

def format_label(x):
    s = f"{x:.3f}"
    if s == "0.000":
        return "0"
    return s.rstrip("0").rstrip(".")

def plot_transition_matrix(counts_df, output_path, title):
    """繪製單張 4x5 Transition Matrix"""
    total = counts_df.sum().sum()
    if total == 0: return

    prob_df = (counts_df / total) * 100
    
    annot_labels = prob_df.copy().astype(object)
    for col in annot_labels.columns:
        annot_labels[col] = annot_labels[col].apply(format_label)

    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(prob_df, annot=annot_labels.values, fmt='', cmap='YlGnBu', 
                     linewidths=.5, annot_kws={"size": 24}, cbar=True) 
    
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Contrastive State', fontsize=18, fontweight='bold')
    plt.ylabel('Baseline State', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"   Saved plot: {output_path}")

def plot_combined_matrix(model_matrices, output_path):
    """繪製三張水平排列的合併圖 (Paper 用)，每張圖都有自己的 Colorbar"""
    fig, axes = plt.subplots(1, 3, figsize=(32, 8), sharey=True)
    
    model_order = ["qwen", "desta", "af3"]
    
    for i, model_key in enumerate(model_order):
        ax = axes[i]
        counts_df = model_matrices.get(model_key)
        
        if counts_df is None or counts_df.sum().sum() == 0:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=20)
            continue

        total = counts_df.sum().sum()
        prob_df = (counts_df / total) * 100
        
        annot_labels = prob_df.copy().astype(object)
        for col in annot_labels.columns:
            annot_labels[col] = annot_labels[col].apply(format_label)
        
        sns.heatmap(prob_df, annot=annot_labels.values, fmt='', cmap='YlGnBu', 
                    linewidths=.5, annot_kws={"size": 24}, 
                    ax=ax, cbar=True)
        
        ax.set_title(MODEL_NAME_MAP[model_key], fontsize=24, fontweight='bold', pad=20)
        
        ax.set_xlabel('Contrastive State', fontsize=18, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Baseline State', fontsize=18, fontweight='bold')
        else:
            ax.set_ylabel('') 

        ax.tick_params(axis='x', labelsize=16, rotation=45)
        ax.tick_params(axis='y', labelsize=16, rotation=0)
        ax.set_xticklabels(LABELS_X, rotation=45, ha='right')
        ax.set_yticklabels(LABELS_Y, rotation=0)

    plt.tight_layout() 
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   Saved COMBINED plot: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--base_dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_model_matrices = {}

    for model in MODELS:
        print(f"\n=== Processing Model: {model} ({MODEL_NAME_MAP[model]}) ===")
        
        model_total_counts = pd.DataFrame(0, index=LABELS_Y, columns=LABELS_X)
        
        for method in METHODS:
            for task in TASKS:
                path_none = os.path.join(args.base_dir, f"results/{model}/benchmark/{task}/log/output_states_wq.jsonl")
                path_cont = os.path.join(args.base_dir, f"results/{model}-{method}/benchmark/{task}/log/output_states_wq.jsonl")
                
                states_none = load_states(path_none)
                states_cont = load_states(path_cont)
                common_ids = set(states_none.keys()) & set(states_cont.keys())
                
                task_counts = pd.DataFrame(0, index=LABELS_Y, columns=LABELS_X)
                for uid in common_ids:
                    s_none = states_none[uid]
                    s_cont = states_cont[uid]
                    if s_none not in LABELS_Y: continue
                    if s_cont in LABELS_X:
                        task_counts.loc[s_none, s_cont] += 1
                
                model_total_counts += task_counts
                
                if task_counts.sum().sum() > 0:
                    plot_transition_matrix(
                        task_counts, 
                        os.path.join(args.output_dir, f"{model}_{method}_{task}_4x5.png"), 
                        f"{MODEL_NAME_MAP[model]} - {method.upper()} - {TASKS_ABBV[task]}"
                    )

        all_model_matrices[model] = model_total_counts

        if model_total_counts.sum().sum() > 0:
            plot_transition_matrix(
                model_total_counts, 
                os.path.join(args.output_dir, f"{model}_AVERAGED_4x5.png"), 
                MODEL_NAME_MAP[model]
            )
        else:
            print(f"   No data for {model}")

    print("\n=== Generating Combined Figure ===")
    plot_combined_matrix(
        all_model_matrices, 
        os.path.join(args.output_dir, "combined_models_transition.png")
    )

if __name__ == "__main__":
    main()