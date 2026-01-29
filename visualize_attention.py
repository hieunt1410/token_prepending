import os
import sys
import math
import argparse
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from senllm import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM


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


def build_prompt(text, prompt_method, use_tp):
    """Build prompt-wrapped text (mirrors evaluate.py batcher logic)."""
    if len(text) > 0 and text[-1] not in '.?"\'':
        text += "."
    text = text.replace('"', "'")
    if len(text) > 0 and text[-1] == "?":
        text = text[:-1] + "."

    if prompt_method == "prompteol":
        if use_tp:
            return f'This sentence : <PST> "{text}" means in one word:"'
        else:
            return f'This sentence : "{text}" means in one word:"'
    elif prompt_method == "cot":
        if use_tp:
            return f'After thinking step by step , this sentence : <PST> "{text}" means in one word:"'
        else:
            return f'After thinking step by step , this sentence : "{text}" means in one word:"'
    elif prompt_method == "ke":
        if use_tp:
            return f'The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : <PST> "{text}" means in one word:"'
        else:
            return f'The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : "{text}" means in one word:"'
    else:
        raise ValueError(f"Unknown prompt_method: {prompt_method}")


def plot_head_averaged_heatmap(attn_weights, tokens, layer_idx, output_dir):
    """Plot head-averaged attention heatmap for a single layer."""
    # attn_weights: (num_heads, seq_len, seq_len)
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(6, len(tokens) * 0.4)))
    sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis",
                vmin=0, vmax=avg_attn.max(), ax=ax)
    ax.set_title(f"Layer {layer_idx} — Head-Averaged Attention")
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_avg.png"), dpi=150)
    plt.close()


def plot_per_head_heatmaps(attn_weights, tokens, layer_idx, output_dir):
    """Plot per-head attention heatmaps in a grid for a single layer."""
    # attn_weights: (num_heads, seq_len, seq_len)
    num_heads = attn_weights.shape[0]
    cols = min(4, num_heads)
    rows = math.ceil(num_heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for h in range(num_heads):
        r, c = divmod(h, cols)
        head_attn = attn_weights[h].cpu().numpy()
        sns.heatmap(head_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis",
                    vmin=0, vmax=head_attn.max(), ax=axes[r, c], cbar=False)
        axes[r, c].set_title(f"Head {h}", fontsize=8)
        axes[r, c].tick_params(labelsize=5)

    # Hide unused subplots
    for h in range(num_heads, rows * cols):
        r, c = divmod(h, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Layer {layer_idx} — Per-Head Attention", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_heads.png"), dpi=150)
    plt.close()


def plot_pst_attention_across_layers(all_attentions, tokens, pst_idx, output_dir):
    """Plot how the <PST> token's attention distribution evolves across layers."""
    num_layers = len(all_attentions)

    # 1. Heatmap: each row is a layer, columns are tokens the PST attends to
    pst_attn_matrix = []
    for layer_attn in all_attentions:
        # layer_attn: (1, num_heads, seq_len, seq_len) -> average over heads
        avg = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)
        pst_row = avg[pst_idx].cpu().numpy()  # attention from PST to all tokens
        pst_attn_matrix.append(pst_row)

    pst_attn_matrix = np.array(pst_attn_matrix)  # (num_layers, seq_len)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(4, num_layers * 0.25)))
    sns.heatmap(pst_attn_matrix, xticklabels=tokens,
                yticklabels=[f"L{i}" for i in range(num_layers)],
                cmap="magma", ax=ax)
    ax.set_title("<PST> Token Attention Across Layers")
    ax.set_xlabel("Attended Token")
    ax.set_ylabel("Layer")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pst_attention_across_layers.png"), dpi=150)
    plt.close()

    # 2. Also plot how much OTHER tokens attend TO the PST token across layers
    attn_to_pst = []
    for layer_attn in all_attentions:
        avg = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)
        col = avg[:, pst_idx].cpu().numpy()  # all tokens' attention to PST
        attn_to_pst.append(col)

    attn_to_pst = np.array(attn_to_pst)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(4, num_layers * 0.25)))
    sns.heatmap(attn_to_pst, xticklabels=tokens,
                yticklabels=[f"L{i}" for i in range(num_layers)],
                cmap="magma", ax=ax)
    ax.set_title("Attention TO <PST> Token Across Layers")
    ax.set_xlabel("Query Token")
    ax.set_ylabel("Layer")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_to_pst_across_layers.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize attention scores")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_which_plan", type=str, choices=["tp", "vanilla"], default="tp")
    parser.add_argument("--output_layer", type=int, default=-1)
    parser.add_argument("--tp_starting_index", type=int, default=1)
    parser.add_argument("--tp_exiting_index", type=int, default=99)
    parser.add_argument("--prompt_method", type=str, default="prompteol",
                        choices=["prompteol", "cot", "ke"])
    parser.add_argument("--text", type=str, required=True, help="Input text to visualize")
    parser.add_argument("--output_dir", type=str, default="./attention_viz")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Specific layers to plot (default: all)")
    args = parser.parse_args()

    # Load YAML config
    if args.config:
        config = load_config_from_yaml(args.config_file, args.config)
        if config is None:
            sys.exit(1)
        for key in ["model_name_or_path", "use_which_plan", "output_layer",
                     "tp_starting_index", "tp_exiting_index", "prompt_method"]:
            if key in config:
                setattr(args, key, config[key])
        if "gpu_config" in config and "cuda_visible_devices" in config["gpu_config"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_config"]["cuda_visible_devices"]

    if not args.model_name_or_path:
        print("Error: model path not specified")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    use_tp = args.use_which_plan == "tp"

    # Load model with eager attention
    model_path_lower = args.model_name_or_path.lower()
    load_kwargs = dict(
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if "llama" in model_path_lower:
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    elif "qwen2" in model_path_lower:
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    elif "gemma" in model_path_lower:
        model = Gemma2ForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    else:
        raise ValueError(f"Unsupported model: {args.model_name_or_path}")

    model.model.plan = args.use_which_plan
    model.model.tp_starting_index = args.tp_starting_index
    model.model.tp_exiting_index = args.tp_exiting_index
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if use_tp:
        placeholder_token = "<PST>"
        tokenizer.add_tokens([placeholder_token])
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        model.resize_token_embeddings(len(tokenizer))
        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.requires_grad_(False)
        num_dim = embedding_layer.weight.shape[1]
        device = embedding_layer.weight.device
        with torch.no_grad():
            embedding_layer.weight[placeholder_token_id] = torch.randn(num_dim, device=device)
        embedding_layer.weight.requires_grad_(True)

    # Build prompt and tokenize
    prompted_text = build_prompt(args.text, args.prompt_method, use_tp)
    print(f"Prompted text: {prompted_text}")

    inputs = tokenizer(prompted_text, return_tensors="pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k in inputs:
        if inputs[k] is not None:
            inputs[k] = inputs[k].to(device)

    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]).strip() or f"[{tid}]" for tid in token_ids]

    # Find PST token index
    pst_idx = None
    if use_tp:
        for i, tid in enumerate(token_ids):
            if tid == placeholder_token_id:
                pst_idx = i
                break

    # Forward pass
    with torch.no_grad():
        outputs = model(output_attentions=True, output_hidden_states=True,
                        return_dict=True, **inputs)

    all_attentions = outputs.attentions  # tuple of (1, num_heads, seq_len, seq_len)
    num_layers = len(all_attentions)
    print(f"Model has {num_layers} layers, sequence length = {len(tokens)}")

    # Determine which layers to plot
    layers_to_plot = args.layers if args.layers is not None else list(range(num_layers))
    layers_to_plot = [l for l in layers_to_plot if 0 <= l < num_layers]

    # Plot per-layer visualizations
    for layer_idx in layers_to_plot:
        attn = all_attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
        print(f"Plotting layer {layer_idx}...")
        plot_head_averaged_heatmap(attn, tokens, layer_idx, args.output_dir)
        plot_per_head_heatmaps(attn, tokens, layer_idx, args.output_dir)

    # Plot PST attention across layers
    if use_tp and pst_idx is not None:
        print(f"Plotting <PST> attention (token index {pst_idx})...")
        plot_pst_attention_across_layers(all_attentions, tokens, pst_idx, args.output_dir)
    elif use_tp:
        print("Warning: <PST> token not found in input, skipping PST visualizations")

    print(f"Done. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
