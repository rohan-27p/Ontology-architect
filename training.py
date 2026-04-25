"""Training helpers for staged Hugging Face workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .curriculum import read_jsonl
    from .models import OntologyArchitectAction
    from .server.ontology_architect_environment import OntologyArchitectEnvironment
except ImportError:  # pragma: no cover
    from curriculum import read_jsonl
    from models import OntologyArchitectAction
    from server.ontology_architect_environment import OntologyArchitectEnvironment


def write_sft_manifest(
    model_id: str,
    data_path: str | Path,
    output_dir: str | Path,
    dry_run: bool,
) -> dict:
    examples = read_jsonl(data_path)
    manifest = {
        "stage": "sft",
        "model_id": model_id,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "examples": len(examples),
        "dry_run": dry_run,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def write_gro_manifest(
    model_id: str,
    output_dir: str | Path,
    group_size: int,
    max_steps: int,
    dry_run: bool,
) -> dict:
    manifest = {
        "stage": "group_reward_optimization",
        "model_id": model_id,
        "output_dir": str(output_dir),
        "group_size": group_size,
        "max_steps": max_steps,
        "dry_run": dry_run,
        "algorithm": "sample multiple theory rewrites, score each group, optimize relative discovery rewards",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gro_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def run_transformers_sft(
    model_id: str,
    data_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    batch_size: int,
) -> Any:
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Install training extras with `uv sync --extra train` before full SFT.") from exc

    examples = read_jsonl(data_path)
    if not examples:
        raise ValueError(f"No training examples found in {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    def format_row(row: dict) -> dict:
        text = row["prompt"] + "\n\n# THEORY MODULE\n" + row["completion"]
        tokens = tokenizer(text, truncation=True, max_length=2048)
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    dataset = Dataset.from_list(examples).map(format_row, remove_columns=list(examples[0]))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max(1, max_steps),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def run_group_reward_optimization(
    model_id: str,
    config: Any,
    output_dir: str | Path,
    group_size: int,
    max_steps: int,
    max_new_tokens: int = 768,
    learning_rate: float = 1e-6,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Install training extras with `uv sync --extra train` before GRO.") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reward_log_path = output / "group_reward_samples.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    reward_history = []
    with reward_log_path.open("w", encoding="utf-8") as reward_log:
        for step in range(max_steps):
            prompt_env = OntologyArchitectEnvironment(config)
            prompt = prompt_env.reset().text + "\n\n# THEORY MODULE\n"
            encoded = tokenizer([prompt] * group_size, return_tensors="pt", padding=True).to(device)
            prompt_len = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            candidates = [_extract_completion(text) for text in decoded]
            rewards = []
            for candidate in candidates:
                env = OntologyArchitectEnvironment(config)
                env.reset()
                observation = env.step(
                    OntologyArchitectAction(
                        theory_module=candidate,
                        revision_note=f"group reward optimization step={step}",
                    )
                )
                rewards.append(float(observation.reward or 0.0))

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantages = reward_tensor - reward_tensor.mean()
            logits = model(generated).logits[:, :-1, :]
            targets = generated[:, 1:]
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1,
                targets.unsqueeze(-1),
            ).squeeze(-1)
            mask = torch.ones_like(targets, dtype=torch.float32, device=device)
            mask[:, : max(prompt_len - 1, 0)] = 0.0
            sequence_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            loss = -(advantages.detach() * sequence_log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            row = {
                "step": step,
                "rewards": rewards,
                "mean_reward": float(reward_tensor.mean().detach().cpu()),
                "loss": float(loss.detach().cpu()),
            }
            reward_log.write(json.dumps(row, sort_keys=True) + "\n")
            reward_log.flush()
            reward_history.append(row)

    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    return {
        "stage": "group_reward_optimization",
        "model_id": model_id,
        "output_dir": str(output),
        "reward_log_path": str(reward_log_path),
        "steps": max_steps,
        "last_mean_reward": reward_history[-1]["mean_reward"] if reward_history else 0.0,
    }


def require_trl_gro() -> None:
    try:
        import trl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Install training extras with `uv sync --extra train` before TRL trainers.") from exc


def _extract_completion(decoded_text: str) -> str:
    marker = "# THEORY MODULE"
    if marker in decoded_text:
        return decoded_text.split(marker, 1)[1].strip()
    return decoded_text.strip()
