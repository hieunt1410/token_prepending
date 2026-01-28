import os
import sys
import json
import logging
import argparse

import mteb
import yaml
import textwrap
from colorama import Fore, Style

from encoder_model import TokenPrependingRetrievalModel

# Import custom LEMB tasks
from retrieval_tasks.LEMBNeedleRetrieval import LEMBNeedleRetrieval
from retrieval_tasks.LEMBPasskeyRetrieval import LEMBPasskeyRetrieval
from retrieval_tasks.LEMBNarrativeQARetrieval import LEMBNarrativeQARetrieval
from retrieval_tasks.LEMBQMSumRetrieval import LEMBQMSumRetrieval
from retrieval_tasks.LEMBSummScreenFDRetrieval import LEMBSummScreenFDRetrieval
from retrieval_tasks.LEMBWikimQARetrieval import LEMBWikimQARetrieval
from retrieval_tasks.coliee_task1 import ColieeTask1

CUSTOM_TASKS = {
    "LEMBNeedleRetrieval": LEMBNeedleRetrieval,
    "LEMBPasskeyRetrieval": LEMBPasskeyRetrieval,
    "LEMBNarrativeQARetrieval": LEMBNarrativeQARetrieval,
    "LEMBQMSumRetrieval": LEMBQMSumRetrieval,
    "LEMBSummScreenFDRetrieval": LEMBSummScreenFDRetrieval,
    "LEMBWikimQARetrieval": LEMBWikimQARetrieval,
    "coliee_task1": ColieeTask1,
}

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config_from_yaml(config_file="config.yaml", config_name=None):
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Config error: {e}")
        return None

    if config_name is None:
        config_name = yaml_config.get("default_config", "llama-2-7b")

    if config_name not in yaml_config.get("models", {}):
        available = list(yaml_config.get("models", {}).keys())
        print(f"Config '{config_name}' not found. Available: {available}")
        return None

    config = yaml_config["models"][config_name].copy()
    if "gpu_config" in yaml_config:
        config["gpu_config"] = yaml_config["gpu_config"]
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--use_which_plan", type=str, choices=["tp", "vanilla"], default="tp"
    )
    parser.add_argument("--output_layer", type=int, default=-1)
    parser.add_argument("--tp_starting_index", type=int, default=1)
    parser.add_argument("--tp_exiting_index", type=int, default=99)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--prompt_method",
        type=str,
        default="prompteol",
        choices=["prompteol", "cot", "ke"],
    )
    parser.add_argument("--encode_max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./retrieval_results")
    parser.add_argument(
        "--task_list",
        type=str,
        nargs="+",
        default=[
            "LEMBSummScreenFDRetrieval",
            "LEMBQMSumRetrieval",
            "LEMBWikimQARetrieval",
            "LEMBNarrativeQARetrieval",
            "LEMBNeedleRetrieval",
            "LEMBPasskeyRetrieval",
            "coliee_task1",
        ],
    )

    args = parser.parse_args()

    # Load from YAML config if specified
    if args.config:
        config = load_config_from_yaml(args.config_file, args.config)
        if config is None:
            sys.exit(1)
        for key in [
            "model_name_or_path",
            "use_which_plan",
            "output_layer",
            "tp_starting_index",
            "tp_exiting_index",
            "batch_size",
            "prompt_method",
            "encode_max_length",
            "output_dir",
            "task_list",
        ]:
            if key in config:
                setattr(args, key, config[key])
        if "gpu_config" in config and "cuda_visible_devices" in config["gpu_config"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_config"][
                "cuda_visible_devices"
            ]

    if not args.model_name_or_path:
        print("Error: model path not specified")
        sys.exit(1)

    print(
        textwrap.dedent(f"""
        {Fore.CYAN}Retrieval Evaluation Configuration:{Style.RESET_ALL}
        {Fore.YELLOW}-----------------------------------{Style.RESET_ALL}
        {Fore.GREEN}Backbone                :{Style.RESET_ALL} {args.model_name_or_path.split("/")[-1]}
        {Fore.GREEN}Plan                    :{Style.RESET_ALL} {args.use_which_plan}
        {Fore.GREEN}Prompt Method           :{Style.RESET_ALL} {args.prompt_method}
        {Fore.GREEN}Output Layer            :{Style.RESET_ALL} {args.output_layer}
        {Fore.GREEN}TP Starting Index       :{Style.RESET_ALL} {args.tp_starting_index}
        {Fore.GREEN}TP Exiting Index        :{Style.RESET_ALL} {args.tp_exiting_index}
        {Fore.GREEN}Batch Size              :{Style.RESET_ALL} {args.batch_size}
        {Fore.GREEN}Encode Max Length        :{Style.RESET_ALL} {args.encode_max_length}
        {Fore.GREEN}Tasks                   :{Style.RESET_ALL} {args.task_list}
    """)
    )

    # Build model
    model = TokenPrependingRetrievalModel(args)

    # Setup output directory
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    mteb_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(mteb_output_dir, exist_ok=True)

    # Split tasks into retrieval vs needle/passkey
    retrieval_task_list = []
    needle_passkey_task_list = []
    output_dict = {}

    for task in [
        "LEMBSummScreenFDRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBWikimQARetrieval",
        "LEMBNarrativeQARetrieval",
        "coliee_task1",
    ]:
        if task in args.task_list:
            retrieval_task_list.append(task)

    for task in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        if task in args.task_list:
            needle_passkey_task_list.append(task)

    # Evaluate needle/passkey tasks
    if needle_passkey_task_list:
        context_length = args.encode_max_length
        tasks = [
            CUSTOM_TASKS[name](context_length=context_length)
            for name in needle_passkey_task_list
        ]
        results = mteb.evaluate(
            model,
            tasks,
            prediction_folder=mteb_output_dir,
            overwrite_strategy="only-missing",
            encode_kwargs={"batch_size": args.batch_size},
        )
        for task_result in results.task_results:
            split = "test"
            if split in task_result.scores and task_result.scores[split]:
                scores = task_result.scores[split][0]
                output_dict[task_result.task_name] = {
                    "ndcg@1": scores.get("ndcg_at_1"),
                    "ndcg@10": scores.get("ndcg_at_10"),
                }

    # Evaluate retrieval tasks
    if retrieval_task_list:
        tasks = [CUSTOM_TASKS[name]() for name in retrieval_task_list]
        results = mteb.evaluate(
            model,
            tasks,
            prediction_folder=mteb_output_dir,
            overwrite_strategy="only-missing",
            encode_kwargs={"batch_size": args.batch_size},
        )
        for task_result in results.task_results:
            split = "test" if "test" in task_result.scores else "validation"
            if split in task_result.scores and task_result.scores[split]:
                scores = task_result.scores[split][0]
                output_dict[task_result.task_name] = {
                    "ndcg@1": scores.get("ndcg_at_1"),
                    "ndcg@10": scores.get("ndcg_at_10"),
                }

    # Print and save results
    logger.info(f"Results: {output_dict}")
    with open(os.path.join(mteb_output_dir, "overall_results.json"), "w") as f:
        json.dump(output_dict, f, indent=4)
    print(f"\nResults saved to {os.path.join(mteb_output_dir, 'overall_results.json')}")


if __name__ == "__main__":
    main()
