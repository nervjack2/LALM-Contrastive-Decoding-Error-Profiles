# Test Time LALM

### Setup
- Use python 3.11
- pip install -r requirements.txt
- Install desta if you need

### Usage

Check available ```systems_name``` and ```task_name``` in ```src/systems/load.py``` and ```src/tasks/load.py```.
```
python run_benchmark.py -o [output_name] -s [system_name] -t [task_name]
```
And the results will be logged under ```results/[system_name]/[output_name]/[task_name]/```.

Some systems might require configuration for more precise control. Examples:
```
python run_benchmark.py -o benchmark -s none -t sakura_animal  # Qwen2.5-Omni-3B
python run_benchmark.py -o benchmark -s none -t sakura_animal-r  # Qwen2.5-Omni-3B, reasoning
python run_benchmark.py -o r=5-emb -s repeat -t sakura_animal --model_config config/model/repeat-emb.yaml  # repeat audio embedding x 5
python run_benchmark.py -o official -s pma -t sakura_language --model_config config/model/pma.yaml  # pay more attention paper with their hyperparameters
```

### Add New System / Task
Each system should at least implement the ```inference``` method.

Each task should be a torch dataset returning a dictionary with 4 keys, for example
```
res = {
    "id": str,
    "audio_input": np.ndarray,
    "text_input": str,
    "output": str
}
```
And need to implement class method ```eval(self, pred: str, gt: str) -> float``` for ```run_benchmark.py``` to calculate the score for each task.

### Other

To access more information, ```inference()``` can return a dictionary for advanced usage. Everything will be dumped into ```result/results.pkl```, and default only the text prediction will be logged into a text file.
