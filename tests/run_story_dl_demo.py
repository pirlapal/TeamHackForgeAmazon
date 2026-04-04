"""
Zeno - Compact Story Demo for Deep Learning
===========================================

A smaller, more coherent DistilBERT demo centered on a real-world story:

  Generic sentiment knowledge -> financial sentiment classification

Why this story is stronger:
  1. the target problem is real-world and economically relevant,
  2. the transfer source is plausibly related to the target,
  3. the demo compares scratch, full transfer, LoRA, and LoRA+,
  4. it can optionally contrast a mismatched source to show why
     transfer-source choice matters.

Usage:
    python -m tests.run_story_dl_demo
    python -m tests.run_story_dl_demo --include-bad-source
    python -m tests.run_story_dl_demo --seeds 3 --max-source-samples 1200 --max-target-samples 512
"""

import argparse
import copy
import os
import sys
import time
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libraries.metrics import set_seed
from libraries.dl.lora import LoRAInjector
from libraries.dl.negative_transfer import compute_cka
from libraries.dl.carbon import GPUCarbonTracker
from libraries.dl.train import train_epoch, evaluate


MODEL_NAME = "distilbert-base-uncased"

DATASET_SPECS = {
    "sst2": {
        "path": "stanfordnlp/sst2",
        "text_key": "sentence",
        "label_key": "label",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "num_labels": 2,
        "story_name": "General sentiment (SST-2)",
    },
    "financial_phrasebank": {
        "path": "atrost/financial_phrasebank",
        "text_key": "sentence",
        "label_key": "label",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "num_labels": 3,
        "story_name": "Financial sentiment (Financial Phrasebank)",
    },
    "ag_news": {
        "path": "fancyzhx/ag_news",
        "text_key": "text",
        "label_key": "label",
        "train_split": "train",
        "val_split": None,
        "test_split": "test",
        "num_labels": 4,
        "story_name": "News topic classification (AG News)",
    },
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def detect_power():
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            return power_mw / 1000.0, name
        except Exception:
            name = torch.cuda.get_device_name(0)
            tdp_map = {
                "T4": 70,
                "L4": 72,
                "A10": 150,
                "V100": 300,
                "A100": 400,
                "RTX 4090": 450,
                "RTX 3090": 350,
            }
            for key, watts in tdp_map.items():
                if key in name:
                    return float(watts), name
            return 70.0, name
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 30.0, "Apple MPS"
    return 30.0, "CPU"


GPU_POWER_WATTS, GPU_NAME = detect_power()


class LLMClassifier(nn.Module):
    """DistilBERT wrapper with reusable CLS features."""

    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import AutoModel
        sys.stdout.flush()
        sys.stderr.flush()
        old_out = os.dup(1)
        old_err = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            self.transformer = AutoModel.from_pretrained(model_name)
        finally:
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(devnull)
            os.close(old_out)
            os.close(old_err)

        hidden = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def encode(self, x):
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, x):
        return self.classifier(self.encode(x))



def get_backbone_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("classifier.")}



def load_backbone_state_dict(model, backbone_sd):
    current = model.state_dict()
    for key, value in backbone_sd.items():
        if key in current and current[key].shape == value.shape:
            current[key] = value
    model.load_state_dict(current)



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def count_total_params(model):
    return sum(p.numel() for p in model.parameters())



def count_peft_trainable(model):
    return sum(p.numel() for p in (
        LoRAInjector.get_lora_parameters(model)
        + LoRAInjector.get_non_lora_trainable_parameters(model)
    ))



def distilbert_lora_targets(scope):
    if scope == "attention":
        return ["q_lin", "v_lin"]
    return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]



def make_tracker(label):
    return GPUCarbonTracker(label, power_watts=GPU_POWER_WATTS)



def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, ci95



def fmt_stats(stats, pct=False):
    mean, std, ci95 = stats
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return f"{mean * scale:.2f}{suffix} +/- {std * scale:.2f}{suffix} (95% CI +/- {ci95 * scale:.2f}{suffix})"



def _get_split(ds_dict, requested):
    if requested is None:
        return None
    if requested in ds_dict:
        return ds_dict[requested]
    return None



def _subsample(ds, limit, seed):
    if ds is None or limit is None or len(ds) <= limit:
        return ds
    return ds.shuffle(seed=seed).select(range(limit))



def _train_val_test_from_train(train_ds, seed):
    temp = train_ds.train_test_split(test_size=0.2, seed=seed)
    train_part = temp["train"]
    rest = temp["test"].train_test_split(test_size=0.5, seed=seed + 1)
    return train_part, rest["train"], rest["test"]



def load_dataset_triplet(name, tokenizer, max_train_samples, max_eval_samples,
                         max_length, seed):
    from datasets import load_dataset

    spec = DATASET_SPECS[name]
    ds = load_dataset(spec["path"])

    train_ds = _get_split(ds, spec["train_split"])
    val_ds = _get_split(ds, spec["val_split"])
    test_ds = _get_split(ds, spec["test_split"])

    if val_ds is None or test_ds is None:
        train_ds, val_ds, test_ds = _train_val_test_from_train(train_ds, seed)

    train_ds = _subsample(train_ds, max_train_samples, seed)
    val_ds = _subsample(val_ds, max_eval_samples, seed + 1)
    test_ds = _subsample(test_ds, max_eval_samples, seed + 2)

    def encode(split_ds):
        texts = split_ds[spec["text_key"]]
        labels = split_ds[spec["label_key"]]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        packed = torch.stack([
            encoded["input_ids"].float(),
            encoded["attention_mask"].float(),
        ], dim=1)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return packed, labels_t

    train_x, train_y = encode(train_ds)
    val_x, val_y = encode(val_ds)
    test_x, test_y = encode(test_ds)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_labels": spec["num_labels"],
        "story_name": spec["story_name"],
        "n_train": len(train_x),
        "n_val": len(val_x),
        "n_test": len(test_x),
    }



def train_with_best(model, train_loader, val_loader, criterion, optimizer, epochs,
                    device, tracker=None, quiet=False, label=None):
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    history = []

    if tracker is not None:
        tracker.start()

    start = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device=str(device))
        val_result = evaluate(model, val_loader, criterion, device=str(device))
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_acc": val_result["accuracy"],
        })
        if (val_result["accuracy"] > best_val_acc or
                (abs(val_result["accuracy"] - best_val_acc) < 1e-9 and val_result["loss"] < best_val_loss)):
            best_val_acc = val_result["accuracy"]
            best_val_loss = val_result["loss"]
            best_state = copy.deepcopy(model.state_dict())
        if not quiet:
            prefix = f"[{label}] " if label else ""
            print(f"    {prefix}epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_result['loss']:.4f}  val_acc={val_result['accuracy']:.1%}")

    elapsed = time.time() - start
    carbon = tracker.stop() if tracker is not None else None
    model.load_state_dict(best_state)
    best_val = evaluate(model, val_loader, criterion, device=str(device))

    return {
        "best_val": best_val,
        "history": history,
        "time_s": elapsed,
        "carbon": carbon,
    }



def collect_embeddings(model, dataloader, device, max_batches=4):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            x = batch[0].to(device)
            outputs.append(model.encode(x).cpu())
    return torch.cat(outputs, dim=0)



def train_source_model(args, tokenizer, dataset_name, seed, quiet):
    data = load_dataset_triplet(
        dataset_name,
        tokenizer,
        max_train_samples=args.max_source_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        seed=seed,
    )
    model = LLMClassifier(MODEL_NAME, num_labels=data["num_labels"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tracker = make_tracker(f"source_{dataset_name}")
    train_info = train_with_best(
        model,
        data["train"],
        data["val"],
        criterion,
        optimizer,
        epochs=args.source_epochs,
        device=DEVICE,
        tracker=tracker,
        quiet=quiet,
        label=f"source:{dataset_name}",
    )
    test_result = evaluate(model, data["test"], criterion, device=str(DEVICE))
    return model, data, train_info, test_result



def run_target_scratch(args, tokenizer, target_name, seed, quiet):
    data = load_dataset_triplet(
        target_name,
        tokenizer,
        max_train_samples=args.max_target_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        seed=seed,
    )
    model = LLMClassifier(MODEL_NAME, num_labels=data["num_labels"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tracker = make_tracker("target_scratch")
    train_info = train_with_best(
        model, data["train"], data["val"], criterion, optimizer,
        epochs=args.target_epochs, device=DEVICE, tracker=tracker,
        quiet=quiet, label="scratch",
    )
    test_result = evaluate(model, data["test"], criterion, device=str(DEVICE))
    return {
        "model": model,
        "data": data,
        "train": train_info,
        "test": test_result,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
    }



def run_target_transfer(args, source_model, target_data, quiet, label="transfer"):
    model = LLMClassifier(MODEL_NAME, num_labels=target_data["num_labels"]).to(DEVICE)
    load_backbone_state_dict(model, get_backbone_state_dict(source_model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tracker = make_tracker(f"{label}_full")
    train_info = train_with_best(
        model, target_data["train"], target_data["val"], criterion, optimizer,
        epochs=args.target_epochs, device=DEVICE, tracker=tracker,
        quiet=quiet, label=label,
    )
    test_result = evaluate(model, target_data["test"], criterion, device=str(DEVICE))
    return {
        "model": model,
        "train": train_info,
        "test": test_result,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
    }



def run_target_lora(args, source_model, target_data, quiet, use_lora_plus=False):
    model = LLMClassifier(MODEL_NAME, num_labels=target_data["num_labels"]).to(DEVICE)
    load_backbone_state_dict(model, get_backbone_state_dict(source_model))
    LoRAInjector.inject(
        model,
        target_modules=distilbert_lora_targets(args.lora_scope),
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
    )
    LoRAInjector.freeze_non_lora(model)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    if use_lora_plus:
        param_groups = LoRAInjector.get_lora_plus_param_groups(
            model,
            base_lr=args.lora_lr,
            lr_ratio=args.lora_plus_ratio,
            weight_decay=args.weight_decay,
            other_lr=args.lora_lr,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        tracker = make_tracker("target_lora_plus")
        label = "lora+"
    else:
        params = (
            LoRAInjector.get_lora_parameters(model)
            + LoRAInjector.get_non_lora_trainable_parameters(model)
        )
        optimizer = torch.optim.AdamW(params, lr=args.lora_lr, weight_decay=args.weight_decay)
        tracker = make_tracker("target_lora")
        label = "lora"

    train_info = train_with_best(
        model, target_data["train"], target_data["val"], criterion, optimizer,
        epochs=args.target_epochs, device=DEVICE, tracker=tracker,
        quiet=quiet, label=label,
    )
    test_result = evaluate(model, target_data["test"], criterion, device=str(DEVICE))
    return {
        "model": model,
        "train": train_info,
        "test": test_result,
        "trainable_params": count_peft_trainable(model),
        "total_params": count_total_params(model),
    }



def print_method(name, result, pct=True):
    acc = result["test"]["accuracy"]
    val_acc = result["train"]["best_val"]["accuracy"]
    co2 = result["train"]["carbon"]["co2_kg"] if result["train"]["carbon"] else 0.0
    time_s = result["train"]["time_s"]
    trainable = result["trainable_params"]
    total = result["total_params"]
    print(f"  {name:<16} test={acc:.1%}  best_val={val_acc:.1%}  "
          f"trainable={trainable:,}/{total:,}  time={time_s:.1f}s  CO2={co2:.2e} kg")



def summarize_across_runs(runs, key):
    acc = summarize([r[key]["test"]["accuracy"] for r in runs])
    co2 = summarize([r[key]["train"]["carbon"]["co2_kg"] for r in runs])
    time_s = summarize([r[key]["train"]["time_s"] for r in runs])
    return {"accuracy": acc, "co2": co2, "time": time_s}



def main():
    parser = argparse.ArgumentParser(description="Compact DL transfer-learning story demo")
    parser.add_argument("--source-dataset", default="sst2", choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--target-dataset", default="financial_phrasebank", choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--bad-source-dataset", default="ag_news", choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--include-bad-source", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--source-epochs", type=int, default=2)
    parser.add_argument("--target-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-plus-ratio", type=float, default=16.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-scope", choices=["attention", "all"], default="all")
    parser.add_argument("--max-source-samples", type=int, default=600)
    parser.add_argument("--max-target-samples", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    print("=" * 84)
    print("  Zeno Story Demo - Deep Learning")
    print("  \"Transfer generic language skill into a small real-world finance task\"")
    print("=" * 84)
    print(f"  model={MODEL_NAME}  device={DEVICE}  gpu={GPU_NAME}  power={GPU_POWER_WATTS:.0f}W")
    print(f"  source={args.source_dataset}  target={args.target_dataset}  seeds={args.seeds}")
    print(f"  source_epochs={args.source_epochs}  target_epochs={args.target_epochs}  "
          f"lora_rank={args.lora_rank}  lora_scope={args.lora_scope}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_runs = []
    for run_idx in range(args.seeds):
        seed = args.seed + run_idx
        set_seed(seed)
        print("\n" + "-" * 84)
        print(f"  Run {run_idx + 1}/{args.seeds}  (seed={seed})")
        print("-" * 84)

        source_model, source_data, source_train, source_test = train_source_model(
            args, tokenizer, args.source_dataset, seed, args.quiet
        )
        print(f"  Source model: {DATASET_SPECS[args.source_dataset]['story_name']}")
        print(f"    train/val/test = {source_data['n_train']}/{source_data['n_val']}/{source_data['n_test']}")
        print(f"    source test accuracy = {source_test['accuracy']:.1%}")

        target_data = load_dataset_triplet(
            args.target_dataset,
            tokenizer,
            max_train_samples=args.max_target_samples,
            max_eval_samples=args.max_eval_samples,
            max_length=args.max_length,
            seed=seed,
        )
        print(f"  Target task: {DATASET_SPECS[args.target_dataset]['story_name']}")
        print(f"    train/val/test = {target_data['n_train']}/{target_data['n_val']}/{target_data['n_test']}")

        scratch = run_target_scratch(args, tokenizer, args.target_dataset, seed, args.quiet)
        transfer = run_target_transfer(args, source_model, target_data, args.quiet, label="transfer")
        lora = run_target_lora(args, source_model, target_data, args.quiet, use_lora_plus=False)
        lora_plus = run_target_lora(args, source_model, target_data, args.quiet, use_lora_plus=True)

        base_model = LLMClassifier(MODEL_NAME, num_labels=target_data["num_labels"]).to(DEVICE)
        base_feats = collect_embeddings(base_model, target_data["val"], DEVICE)
        source_feats = collect_embeddings(transfer["model"], target_data["val"], DEVICE)
        source_cka = compute_cka(base_feats, source_feats)

        bad_source_result = None
        bad_source_cka = None
        if args.include_bad_source:
            bad_source_model, _, _, _ = train_source_model(
                args, tokenizer, args.bad_source_dataset, seed + 1000, True
            )
            bad_source_result = run_target_transfer(
                args, bad_source_model, target_data, True, label="bad-source"
            )
            bad_source_feats = collect_embeddings(bad_source_result["model"], target_data["val"], DEVICE)
            bad_source_cka = compute_cka(base_feats, bad_source_feats)

        print("\n  Target-side results")
        print_method("Scratch", scratch)
        print_method("Transfer FT", transfer)
        print_method("Transfer + LoRA", lora)
        print_method("Transfer + LoRA+", lora_plus)
        print(f"  Alignment clue     CKA(base, transferred target features) = {source_cka:.4f}")
        if bad_source_result is not None:
            print_method("Bad-source FT", bad_source_result)
            print(f"  Bad-source CKA     CKA(base, bad-source target features) = {bad_source_cka:.4f}")

        all_runs.append({
            "seed": seed,
            "scratch": scratch,
            "transfer": transfer,
            "lora": lora,
            "lora_plus": lora_plus,
            "source_test": source_test,
            "source_cka": source_cka,
            "bad_source": bad_source_result,
            "bad_source_cka": bad_source_cka,
        })

    print("\n" + "=" * 84)
    print("  Cross-run summary")
    print("=" * 84)
    for key, label in [
        ("scratch", "Scratch"),
        ("transfer", "Transfer FT"),
        ("lora", "Transfer + LoRA"),
        ("lora_plus", "Transfer + LoRA+"),
    ]:
        summary = summarize_across_runs(all_runs, key)
        print(f"  {label:<16} accuracy={fmt_stats(summary['accuracy'], pct=True)}")
        print(f"{'':<18} CO2={summary['co2'][0]:.2e} kg +/- {summary['co2'][1]:.2e}  "
              f"time={summary['time'][0]:.1f}s +/- {summary['time'][1]:.1f}s")

    if args.include_bad_source:
        bad_acc = summarize([r["bad_source"]["test"]["accuracy"] for r in all_runs if r["bad_source"] is not None])
        print(f"  {'Bad-source FT':<16} accuracy={fmt_stats(bad_acc, pct=True)}")

    if all_runs:
        base = all_runs[0]["scratch"]["test"]["accuracy"]
        best_key = max(["scratch", "transfer", "lora", "lora_plus"], key=lambda k: all_runs[0][k]["test"]["accuracy"])
        best = all_runs[0][best_key]["test"]["accuracy"]
        print("\n  Demo story")
        print(f"  - Start from scarce target labels on {DATASET_SPECS[args.target_dataset]['story_name']}.")
        print(f"  - Reuse a source model trained on {DATASET_SPECS[args.source_dataset]['story_name']}.")
        print(f"  - Compare full transfer against adapter-based transfer with LoRA and LoRA+.")
        print(f"  - In the first run, the best target result was {best:.1%} vs {base:.1%} for scratch.")
        print("  - For a final project number, rerun this script with 3-5 seeds and larger sample caps.")

    if args.save_json:
        payload = {
            "args": vars(args),
            "runs": [],
        }
        for run in all_runs:
            payload["runs"].append({
                "seed": run["seed"],
                "source_test_accuracy": run["source_test"]["accuracy"],
                "source_cka": run["source_cka"],
                "scratch_test_accuracy": run["scratch"]["test"]["accuracy"],
                "transfer_test_accuracy": run["transfer"]["test"]["accuracy"],
                "lora_test_accuracy": run["lora"]["test"]["accuracy"],
                "lora_plus_test_accuracy": run["lora_plus"]["test"]["accuracy"],
                "bad_source_test_accuracy": (
                    run["bad_source"]["test"]["accuracy"] if run["bad_source"] is not None else None
                ),
                "bad_source_cka": run["bad_source_cka"],
            })
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved results to {args.save_json}")


if __name__ == "__main__":
    main()
