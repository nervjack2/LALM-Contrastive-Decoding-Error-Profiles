# LALM Contrastive Decoding Error Profiles

### Setup
- Use Python 3.11
- `pip install -r requirements.txt`
- Install Desta if needed

### Usage

Check available `system_name` in `src/systems/load.py` and `task_name` in `src/tasks/load.py`.

```
python run_benchmark.py -o [output_name] -s [system_name] -t [task_name] --model_config [config_path]
```

Results will be logged under `results/[system_name]/[output_name]/[task_name]/`.

#### Available Systems

| Model | System Names |
|---|---|
| Qwen2.5-Omni | `qwen`, `qwen-aad`, `qwen-acd`, `qwen-amti`, `qwen-dola` |
| Desta2.5 | `desta`, `desta-official`, `desta-aad`, `desta-acd`, `desta-amti`, `desta-dola` |
| Audio Flamingo 3 | `af3`, `af3-aad`, `af3-acd`, `af3-amti`, `af3-dola` |

#### Available Tasks

| Task Name Pattern | Dataset |
|---|---|
| `sakura_[subject]` | SAKURA (subjects: `animal`, `emotion`, `gender`, `language`) |
| `mmau-test-mini` | MMAU Mini |
| `mmar` | MMAR |

Task name suffixes:
- `-ja`: Use API-based LLM judge (GPT-4o)
- `-jl`: Use local LLM judge (Phi-3.5-mini)
- `-m`: Multi-hop mode (SAKURA only)

#### Examples

```bash
# Qwen baseline on SAKURA animal with API judge
python run_benchmark.py -o test -s qwen -t sakura_animal-ja

# Desta with AAD decoding on SAKURA animal with API judge
python run_benchmark.py -o test -s desta-aad -t sakura_animal-ja --model_config config/aad.yaml

# Audio Flamingo 3 with AMTI decoding
python run_benchmark.py -o test -s af3-amti -t sakura_animal-ja --model_config config/amti.yaml

# Qwen with DoLA on MMAU
python run_benchmark.py -o test -s qwen-dola -t mmau-test-mini-ja --model_config config/dola.yaml
```

#### Config Files

Decoding method configs are under `config/`:
- `aad.yaml` — AAD 
- `acd.yaml` — ACD
- `amti.yaml` — AMTI
- `dola.yaml` — DoLA

### Analysis

```bash
python analysis/analyze_wrong_state_w_question.py <input_result.txt> <output_states.jsonl> <task_name> <mode>
```
- `mode`: `api` (GPT-4o) or `local` (Llama-3.1-8B)