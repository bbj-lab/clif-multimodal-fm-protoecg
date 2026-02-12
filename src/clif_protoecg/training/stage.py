"""Training stage orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clif_protoecg.core.stage import BaseStage, StageResult
from clif_protoecg.tokenization.vocabulary import Vocabulary
from clif_protoecg.training.model import create_model, save_model, SIZE_PRESETS
from clif_protoecg.training.data.dataset import MedicalEventDataset
from clif_protoecg.training.data.collator import (
    DynamicBatchCollator,
    DynamicBatchSampler,
    create_dynamic_batches,
)
from clif_protoecg.training.trainer import (
    TrainerConfig,
    train_step,
    train_epoch,
    evaluate,
    create_optimizer_and_scheduler,
)
from clif_protoecg.utils.logging import get_logger

logger = get_logger("training")

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class TrainingStage(BaseStage):
    @property
    def name(self) -> str:
        return "training"

    def run(
        self,
        input_path: Path,
        output_path: Path,
        vocab: Vocabulary | None = None,
        train_sequences: list[list[int]] | None = None,
        val_sequences: list[list[int]] | None = None,
        config: TrainerConfig | None = None,
        size_name: str = "tiny",
        label_token_ids: set[int] | None = None,
        **kwargs: Any,
    ) -> StageResult:
        wall_start = time.time()
        config = config or TrainerConfig()

        if vocab is None:
            vocab = Vocabulary.load(input_path)

        # Resolve max_seq_len from model preset if not explicitly set
        max_seq_len = config.max_seq_len
        if max_seq_len is None:
            max_seq_len = SIZE_PRESETS[size_name]["max_pos"]
        logger.info(f"Using max_seq_len={max_seq_len} (model {size_name} max_pos={SIZE_PRESETS[size_name]['max_pos']})")

        # Create datasets with chunking for long sequences
        train_ds = MedicalEventDataset(
            train_sequences or [],
            max_seq_len=max_seq_len,
            chunk_overlap=config.chunk_overlap,
            label_token_ids=label_token_ids,
            cont_token_id=vocab.cont_id,
        )
        val_ds = MedicalEventDataset(
            val_sequences or [],
            max_seq_len=max_seq_len,
            chunk_overlap=config.chunk_overlap,
            label_token_ids=label_token_ids,
            cont_token_id=vocab.cont_id,
        )
        logger.info(
            f"Chunked sequences: max_seq_len={max_seq_len}, "
            f"overlap={config.chunk_overlap}, "
            f"train chunks={len(train_ds)}, val chunks={len(val_ds)}"
        )

        # Dynamic batching with token budget using chunk lengths
        collator = DynamicBatchCollator(pad_id=vocab.pad_id)
        train_batches = create_dynamic_batches(train_ds.chunk_lengths(), config.max_tokens_per_batch)
        train_sampler = DynamicBatchSampler(train_batches, shuffle=True)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collator)
        logger.info(f"Train: {len(train_ds)} chunks in {len(train_batches)} batches (budget={config.max_tokens_per_batch})")

        if len(val_ds) > 0:
            val_batches = create_dynamic_batches(val_ds.chunk_lengths(), config.max_tokens_per_batch)
            val_sampler = DynamicBatchSampler(val_batches, shuffle=False)
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collator)
        else:
            val_loader = None

        # Create model
        model = create_model(vocab.size, size_name)
        model.to(config.device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {size_name} ({n_params:,} params) on {config.device}")

        total_steps = len(train_loader) * config.num_epochs
        optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)
        logger.info(
            f"Training: {config.num_epochs} epoch(s), {total_steps} steps, "
            f"eval every {config.eval_steps} steps, lr={config.learning_rate}"
        )

        # Initialize wandb
        use_wandb = _WANDB_AVAILABLE and config.wandb_project is not None
        if use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"{size_name}_{output_path.parent.name}",
                config={
                    "model_size": size_name,
                    "n_params": n_params,
                    "vocab_size": vocab.size,
                    "max_seq_len": max_seq_len,
                    "chunk_overlap": config.chunk_overlap,
                    "max_tokens_per_batch": config.max_tokens_per_batch,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "warmup_steps": config.warmup_steps,
                    "num_epochs": config.num_epochs,
                    "label_loss_weight": config.label_loss_weight,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "bf16": config.bf16,
                    "train_chunks": len(train_ds),
                    "val_chunks": len(val_ds),
                    "train_batches": len(train_batches),
                    "total_steps": total_steps,
                },
            )
            wandb.watch(model, log="gradients", log_freq=config.log_steps)
            logger.info(f"wandb: run={wandb.run.name}, project={config.wandb_project}")
        elif config.wandb_project is not None and not _WANDB_AVAILABLE:
            logger.warning("wandb not installed, skipping logging (pip install wandb)")

        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0
        total_loss = 0.0
        stopped_early = False
        running_loss = 0.0
        running_steps = 0

        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                unit="batch",
                leave=True,
            )

            for step, batch in pbar:
                loss = train_step(model, batch, optimizer, scheduler, config, step)
                epoch_loss += loss
                epoch_steps += 1
                global_step += 1
                running_loss += loss
                running_steps += 1

                # Update progress bar with running loss and LR
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / running_steps
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr:.2e}",
                    step=global_step,
                )

                # Periodic log to file + wandb
                if config.log_steps and global_step % config.log_steps == 0:
                    logger.info(
                        f"Step {global_step}/{total_steps}: "
                        f"loss={avg_loss:.4f}, lr={lr:.2e}"
                    )
                    if use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + epoch_steps / len(train_loader),
                        }, step=global_step)
                    running_loss = 0.0
                    running_steps = 0

                # Step-based evaluation
                if (
                    config.eval_steps
                    and global_step % config.eval_steps == 0
                    and val_loader is not None
                ):
                    pbar.set_description(f"Epoch {epoch + 1} [evaluating]")
                    val_loss = evaluate(model, val_loader, config)
                    train_avg = epoch_loss / epoch_steps
                    elapsed = time.time() - wall_start
                    logger.info(
                        f"Step {global_step}/{total_steps}: "
                        f"train_loss={train_avg:.4f}, val_loss={val_loss:.4f}, "
                        f"best_val={best_val_loss:.4f}, elapsed={elapsed:.0f}s"
                    )
                    if use_wandb:
                        wandb.log({
                            "val/loss": val_loss,
                            "train/loss_avg": train_avg,
                            "val/best_loss": min(best_val_loss, val_loss),
                        }, step=global_step)
                    pbar.set_description(f"Epoch {epoch + 1}/{config.num_epochs}")

                    if config.early_stopping:
                        if val_loss < best_val_loss - config.early_stopping_min_delta:
                            best_val_loss = val_loss
                            patience_counter = 0
                            save_model(model, output_path / "best")
                            logger.info(f"  New best val_loss={val_loss:.4f}, saved checkpoint")
                        else:
                            patience_counter += 1
                            logger.info(
                                f"  No improvement ({patience_counter}/{config.early_stopping_patience})"
                            )
                            if patience_counter >= config.early_stopping_patience:
                                logger.info(f"Early stopping at step {global_step}")
                                stopped_early = True
                                break
                    model.train()

            pbar.close()
            train_loss = epoch_loss / max(epoch_steps, 1)
            total_loss = train_loss
            elapsed = time.time() - wall_start
            logger.info(
                f"Epoch {epoch + 1} complete: train_loss={train_loss:.4f}, "
                f"elapsed={elapsed:.0f}s"
            )

            if stopped_early:
                break

            # End-of-epoch eval (if no step-based eval, or final check)
            if val_loader is not None and not config.eval_steps:
                val_loss = evaluate(model, val_loader, config)
                logger.info(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "train/loss_epoch": train_loss,
                    }, step=global_step)
                if config.early_stopping:
                    if val_loss < best_val_loss - config.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_model(model, output_path / "best")
                        logger.info(f"  New best val_loss={val_loss:.4f}, saved checkpoint")
                    else:
                        patience_counter += 1
                        logger.info(
                            f"  No improvement ({patience_counter}/{config.early_stopping_patience})"
                        )
                        if patience_counter >= config.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                            break

        # Final validation (always run to ensure we have a val_loss)
        if val_loader is not None and best_val_loss == float("inf"):
            final_val = evaluate(model, val_loader, config)
            best_val_loss = final_val
            save_model(model, output_path / "best")
            logger.info(f"Final validation: val_loss={final_val:.4f} (saved as best)")
            if use_wandb:
                wandb.log({"val/loss": final_val}, step=global_step)

        # Save final
        save_model(model, output_path / "final")
        elapsed = time.time() - wall_start
        logger.info(
            f"Training complete: {global_step} steps in {elapsed:.0f}s, "
            f"final train_loss={total_loss:.4f}, best_val_loss={best_val_loss:.4f}"
        )

        if use_wandb:
            wandb.summary["final_train_loss"] = total_loss
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["total_steps"] = global_step
            wandb.summary["elapsed_seconds"] = elapsed
            wandb.summary["stopped_early"] = stopped_early
            wandb.finish()
            logger.info("wandb: run finished")

        return StageResult(
            success=True,
            output_path=output_path,
            duration_seconds=elapsed,
            metrics={"train_loss": total_loss, "best_val_loss": best_val_loss},
        )
