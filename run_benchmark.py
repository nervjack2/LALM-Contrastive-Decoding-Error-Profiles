import os
import argparse
from transformers import set_seed
import yaml
import pickle
from tqdm import tqdm
import json

from src.systems.base import System
from src.systems.load import load_system
from src.tasks.load import get_test_task
from src.utils.tool import batchify


def create_config(args) -> dict:
    """ Create a dictionary for full configuration """
    res = {"system_name": args.system_name, "task_name": None}
    if args.exp_name is not None and args.checkpoint is not None:
        res["checkpoint"] = f"{args.exp_name}/{args.checkpoint}"
        raise NotImplementedError  # currently no training yet
    res["model_config"] = {}
    for path in args.model_config:
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["model_config"].update(config)

    return res


def prepare_system(config) -> System:
    system = load_system(config["system_name"], system_config=config, checkpoint=config.get("checkpoint", None))
    system.eval()

    return system


def run_single_task(system: System, output_dir: str, tname: str, batched: bool=False):
    os.makedirs(output_dir, exist_ok=True)
    system.eval()
    system.cuda()

    ds = get_test_task(tname)

    basenames = []
    predictions = []
    golds = []
    scores = []
    res_list = []
    if not batched:
        for sample in tqdm(ds, total=len(ds)):
            res = system.inference([sample["audio_input"]], [sample["text_input"]], [sample["audio_path"]])
            res_list.append(res)

            prediction = res["prediction"]
            predictions.append(prediction)
            basenames.append(sample["id"])
            golds.append(sample["output"])
            print(prediction)
            score = ds.eval(prediction, sample["output"], sample["text_input"])
            scores.append(score)
            print(score)
    else:  # vllm, currently not supported
        bs = 64
        for batch in tqdm(batchify(ds, batch_size=bs),
            total=(len(ds) + bs - 1) // bs
        ):
            audio_inputs = [sample["audio_input"] for sample in batch]
            text_inputs = [sample["text_input"] for sample in batch]
            res = system.batch_inference(audio_inputs, text_inputs)
            res_list.extend(res)

            for x, sample in zip(res, batch):
                predictions.append(x["prediction"])
                basenames.append(sample["id"])
                golds.append(sample["output"])
                score = ds.eval(x["prediction"], sample["output"])
                scores.append(score)

    # Wrap results with metadata
    payload = {
        "model": system.__class__.__name__,
        "task_name": tname,
        "results": {
            "scores": scores,
            "basenames": basenames,
            "predictions": predictions,
            "golds": golds,
            "all": res_list
        }
    }

    log_results(payload, f"{output_dir}/{tname}")


def log_results(payload, output_dir: str):
    log_dir = f"{output_dir}/log"
    result_dir = f"{output_dir}/result"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # log
    with open(f'{result_dir}/results.pkl', "wb") as f:
        pickle.dump(payload, f)

    results = payload["results"]
    score = sum(results["scores"]) / len(results["scores"])
    print(f"Score: {score * 100:.2f}%")

    results = payload["results"]
    with open(f'{output_dir}/results.txt', "w") as f:
        f.write(f"Score: {score * 100:.2f}%\n")
    with open(f'{log_dir}/predictions.txt', "w") as f:
        for orig, pred, score, basename in zip(results["golds"], results["predictions"], results["scores"], results["basenames"]):
            f.write(f"{score:.2f}|{basename}|{orig}|{pred}\n")


def main(args):
    args.output_dir = f"results/{args.system_name}/{args.output_dir}"
    config = create_config(args)
    system = prepare_system(config)
    print("========================== Start! ==========================")
    print(f"Output Dir: {args.output_dir}")
    print("System name: ", args.system_name)
    print("Task name: ", args.task_names)
    print("Checkpoint Path: ", args.checkpoint)

    for tname in args.task_names:
        os.makedirs(f'{args.output_dir}/{tname}', exist_ok=True)
        config["task_name"] = tname
        with open(f'{args.output_dir}/{tname}/config.yaml', "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)
        run_single_task(system, args.output_dir, tname=tname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR")
    parser.add_argument('-o', '--output_dir', type=str, help="path for evaluated results")
    parser.add_argument('-s', '--system_name', type=str, help="system identifier")
    parser.add_argument('-t', '--task_names', nargs='+', help="list of task names to be evaluate")
    parser.add_argument('-n', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--model_config', nargs='+', default=[])
    parser.add_argument('--debug', action="store_true", default=False)

    set_seed(666)
    args = parser.parse_args()
    main(args)
