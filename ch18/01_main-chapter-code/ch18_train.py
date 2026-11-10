"""Chapter 18: LoRA / QLoRA fine-tuning of Qwen3 for ShopFlow ticket triage.

Runs in the chapter's CUDA venv (see requirements-cuda.txt), on one GPU:

    cd ch18/01_main-chapter-code
    .venv-cuda/bin/python ch18_train.py --model Qwen/Qwen3-1.7B --name q17-lora
    .venv-cuda/bin/python ch18_train.py --model Qwen/Qwen3-1.7B --name q17-qlora --4bit
    .venv-cuda/bin/python ch18_train.py --model Qwen/Qwen3-1.7B --name q17-lora-peft --backend peft

``--backend unsloth`` (default) uses Unsloth's patched model and kernels; ``--backend peft`` is
plain Hugging Face transformers + PEFT + TRL with the same data and hyperparameters, as the
baseline. Both write the adapter to ``adapters/<name>/`` and the measurements (time, peak VRAM,
loss curves) to ``results/train__<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from ch18_data import DATA, HERE, prompt_completion, read_jsonl

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA fine-tuning for ticket triage")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--name", required=True, help="adapter name, e.g. q17-lora")
    ap.add_argument("--backend", choices=["unsloth", "peft"], default="unsloth")
    ap.add_argument("--4bit", dest="load_in_4bit", action="store_true",
                    help="QLoRA: load the frozen base in 4-bit NF4")
    ap.add_argument("--train", nargs="+", default=["train.jsonl"], help="files in data/")
    ap.add_argument("--val", default="val.jsonl")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    ap.add_argument("--limit", type=int, help="use only the first N examples (smoke tests)")
    return ap.parse_args()


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Returns (model, tokenizer, peft_config). peft_config is None when Unsloth wraps it."""
    import torch

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel  # must be imported before transformers / trl

        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_len, dtype=None,
            load_in_4bit=args.load_in_4bit, use_exact_model_name=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none",
            target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        return model, tok, None

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, quantization_config=quant, attn_implementation="sdpa",
        device_map={"": 0},
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    return model, tok, peft_config


def build_dataset(files: list[str], tok: Any, limit: int | None = None) -> Any:
    from datasets import Dataset

    rows = [r for f in files for r in read_jsonl(DATA / f)][:limit]
    return Dataset.from_list([prompt_completion(r, tok.apply_chat_template, tok.eos_token)
                              for r in rows])


def main() -> None:
    args = parse_args()
    t_load = time.perf_counter()
    model, tok, peft_config = load_model(args)  # Unsloth first: it patches transformers/trl

    import torch
    import transformers
    import trl
    from trl import SFTConfig, SFTTrainer

    if tok.eos_token != "<|im_end|>":
        raise SystemExit(f"unexpected eos token {tok.eos_token!r}: check the chat template")
    train_ds = build_dataset(args.train, tok, args.limit)
    val_ds = build_dataset([args.val], tok, args.limit)
    out_dir = HERE / "adapters" / args.name
    cfg = SFTConfig(
        output_dir=str(HERE / "adapters" / f"_ckpt_{args.name}"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=16,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        max_length=args.max_len,
        completion_only_loss=True,  # loss on the answer only, not on the ticket
        gradient_checkpointing=args.backend == "peft",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, peft_config=peft_config)
    load_s = time.perf_counter() - t_load

    # Sanity check: decode exactly the tokens that carry loss in the first example. TRL marks
    # them with ``completion_mask``; Unsloth pre-builds ``labels`` (-100 = no loss) instead.
    row = trainer.train_dataset[0]
    if row.get("labels") is not None:
        mask = [lab != -100 for lab in row["labels"]]
    else:
        mask = row.get("completion_mask") or row["assistant_masks"]
    trained = tok.decode([t for t, m in zip(row["input_ids"], mask, strict=True) if m])
    print(f"tokens in loss for example 0: {trained!r} (keys: {sorted(row)})")
    if not trained.endswith("<|im_end|>") or "<ticket>" in trained:
        raise SystemExit("loss mask is wrong: it must cover the answer and <|im_end|> only")
    n_tokens = sum(len(r["input_ids"]) for r in trainer.train_dataset)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    stats = trainer.train()
    train_s = time.perf_counter() - t0
    final_eval = trainer.evaluate()
    trained_model = trainer.model  # the PEFT-wrapped model in both backends
    trained_model.save_pretrained(out_dir)  # adapter only (tens of MB), not the base
    tok.save_pretrained(out_dir)

    trainable = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trained_model.parameters())
    hist = trainer.state.log_history
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "trl": trl.__version__}
    if args.backend == "unsloth":
        import unsloth

        versions["unsloth"] = unsloth.__version__
    else:
        import peft

        versions["peft"] = peft.__version__
    result = {
        "name": args.name, "backend": args.backend, "model": args.model,
        "quant": "4bit-nf4" if args.load_in_4bit else "bf16",
        "rank": args.rank, "alpha": args.alpha, "epochs": args.epochs, "lr": args.lr,
        "effective_batch": args.batch * args.grad_accum, "train_files": args.train,
        "n_train": len(train_ds), "n_val": len(val_ds), "tokens_per_epoch": n_tokens,
        "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
        "steps": stats.global_step, "final_train_loss": round(stats.training_loss, 4),
        "final_eval_loss": round(final_eval["eval_loss"], 4),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "trainable_params": trainable, "total_params": total,
        "loss": [(h["step"], h["loss"]) for h in hist if "loss" in h],
        "eval_loss": [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h],
        "gpu": torch.cuda.get_device_name(0), "versions": versions,
        "adapter_mb": round(sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 2**20,
                            1),
        "example_0": {"prompt": train_ds[0]["prompt"], "tokens_in_loss": trained},
    }
    out = HERE / "results" / f"train__{args.name}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("loss", "eval_loss")},
                     indent=1))


if __name__ == "__main__":
    main()
