"""Training helpers for staged Hugging Face workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    from .baselines import DUAL_FLUID_DSL_ORACLE
    from .curriculum import read_jsonl
    from .models import OntologyArchitectAction
    from .server.ontology_architect_environment import OntologyArchitectEnvironment
except ImportError:  # pragma: no cover
    from baselines import DUAL_FLUID_DSL_ORACLE
    from curriculum import read_jsonl
    from models import OntologyArchitectAction
    from server.ontology_architect_environment import OntologyArchitectEnvironment

# Few-shot oracle prefix injected into every GRPO prompt to seed DSL format.
# Truncated to ~400 chars so it fits within the context budget.
_FEW_SHOT_PREFIX = (
    "EXAMPLE VALID THEORY (JSON DSL format):\n"
    + DUAL_FLUID_DSL_ORACLE[:400]
    + "\n...\n\n"
)


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
    checkpoint_steps: int | None = None,
) -> dict:
    manifest = {
        "stage": "group_reward_optimization",
        "model_id": model_id,
        "output_dir": str(output_dir),
        "group_size": group_size,
        "max_steps": max_steps,
        "dry_run": dry_run,
        "checkpoint_steps": checkpoint_steps,
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
    use_wandb: bool = True,
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
        tokens = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
        tokens["labels"] = [
            t if t != tokenizer.pad_token_id else -100 
            for t in tokens["input_ids"]
        ]
        return tokens

    dataset = Dataset.from_list(examples).map(format_row, remove_columns=list(examples[0]))

    # Experiment tracking: wandb if available, else none
    report_to = []
    if use_wandb:
        try:
            import wandb  # noqa: F401
            report_to = ["wandb"]
            import os
            os.environ.setdefault("WANDB_PROJECT", "ontology-architect")
            os.environ.setdefault("WANDB_RUN_NAME", "sft-theory-dsl")
        except ImportError:
            pass

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max(1, max_steps),
        report_to=report_to,
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
    max_new_tokens: int = 384,
    learning_rate: float = 1e-6,
    checkpoint_steps: int | None = None,
    use_wandb: bool = True,
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
    tokenizer.padding_side = "left"  # Crucial for causal LM batched generation
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint_steps = checkpoint_steps if checkpoint_steps is not None else getattr(config.training, "checkpoint_steps", 0)
    checkpoint_steps = max(0, int(checkpoint_steps))

    # Experiment tracking
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="ontology-architect",
                name="grpo-theory-discovery",
                config={"model_id": model_id, "group_size": group_size,
                        "max_steps": max_steps, "lr": learning_rate,
                        "max_new_tokens": max_new_tokens},
                reinit=True,
            )
        except ImportError:
            pass

    reward_history = []
    last_checkpoint = ""
    with reward_log_path.open("w", encoding="utf-8") as reward_log:
        for step in range(max_steps):
            prompt_env = OntologyArchitectEnvironment(config)
            # Inject few-shot DSL oracle prefix + DSL opening to force JSON format
            # from the very first generated token. This eliminates the "prose collapse"
            # where the model generates English text instead of valid DSL JSON.
            prompt = (
                _FEW_SHOT_PREFIX
                + prompt_env.reset().text
                + '\n\n# THEORY MODULE\n{"dsl_version": 1, "name": "'
            )
            encoded = tokenizer([prompt] * group_size, return_tensors="pt", padding=True).to(device)
            prompt_len = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=1.2,  # Higher temperature for diversity — GRPO needs varied samples to rank
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

            # Zero-gradient guard: when all rewards are identical (e.g. all -25.0
            # on an all-invalid step), advantages collapse to zero and no gradient
            # flows. Skip the backward pass entirely — there is no ranking signal.
            if reward_tensor.std() < 1e-6:
                valid_rewards = [r for r in rewards if r > -20]
                row = {
                    "step": step,
                    "rewards": rewards,
                    "mean_reward": float(reward_tensor.mean()),
                    "loss": 0.0,
                    "best_reward": max(valid_rewards) if valid_rewards else -25.0,
                    "valid_rate": len(valid_rewards) / len(rewards),
                    "skipped": True,
                }
                reward_log.write(json.dumps(row, sort_keys=True) + "\n")
                reward_log.flush()
                reward_history.append(row)
                print(f"[Step {step:3d}/{max_steps}] SKIPPED (all rewards identical, no ranking signal) valid={row['valid_rate']:.0%}")
                if wandb_run:
                    wandb_run.log({"grpo/skipped": 1, "grpo/valid_rate": row["valid_rate"]}, step=step)
                continue

            # Normalize advantages for stable gradient scale
            advantages = reward_tensor - reward_tensor.mean()
            advantages = advantages / (advantages.std() + 1e-8)

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

            valid_rewards = [r for r in rewards if r > -20]
            row = {
                "step": step,
                "rewards": rewards,
                "mean_reward": float(reward_tensor.mean().detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "best_reward": max(valid_rewards) if valid_rewards else -25.0,  # max = highest reward (least negative)
                "valid_rate": len(valid_rewards) / len(rewards),
                "skipped": False,
            }
            reward_log.write(json.dumps(row, sort_keys=True) + "\n")
            reward_log.flush()
            reward_history.append(row)

            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    "grpo/loss": row["loss"],
                    "grpo/mean_reward": row["mean_reward"],
                    "grpo/best_reward": row["best_reward"],
                    "grpo/valid_rate": row["valid_rate"],
                    "grpo/mean_valid_reward": sum(valid_rewards) / len(valid_rewards) if valid_rewards else -25.0,
                }, step=step)

            # Print progress
            print(f"[Step {step:3d}/{max_steps}] loss={row['loss']:.4f} mean_reward={row['mean_reward']:.2f} best={row['best_reward']:.2f} valid={row['valid_rate']:.0%}")
            if checkpoint_steps and (step + 1) % checkpoint_steps == 0:
                checkpoint_dir = output / f"checkpoint-{step + 1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(checkpoint_dir))
                tokenizer.save_pretrained(str(checkpoint_dir))
                torch.save(
                    {
                        "step": step + 1,
                        "optimizer": optimizer.state_dict(),
                        "last_row": row,
                    },
                    checkpoint_dir / "trainer_state.pt",
                )
                last_checkpoint = str(checkpoint_dir)

    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Finalize experiment tracking
    if wandb_run:
        wandb_run.finish()

    return {
        "stage": "group_reward_optimization",
        "model_id": model_id,
        "output_dir": str(output),
        "reward_log_path": str(reward_log_path),
        "steps": max_steps,
        "last_mean_reward": reward_history[-1]["mean_reward"] if reward_history else 0.0,
        "last_checkpoint": last_checkpoint,
    }


def require_trl_gro() -> None:
    try:
        import trl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Install training extras with `uv sync --extra train` before TRL trainers.") from exc


def _extract_completion(decoded_text: str) -> str:
    """Extract the theory module text that follows the # THEORY MODULE marker.

    Because we inject a partial DSL opening ('{"dsl_version": 1, "name": "') into
    the prompt, the extracted text already begins mid-JSON. The sandbox's DSL parser
    will handle this correctly as long as the JSON is complete after the marker.
    """
    marker = "# THEORY MODULE"
    if marker in decoded_text:
        return decoded_text.split(marker, 1)[1].strip()
    # Fallback: if marker was never generated, try to find any JSON object
    stripped = decoded_text.strip()
    brace = stripped.find("{")
    if brace != -1:
        return stripped[brace:]
    return stripped
