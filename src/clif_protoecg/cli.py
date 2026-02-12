"""Typer CLI for the ProtoECG-FM pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from clif_protoecg.utils.logging import get_logger

app = typer.Typer(
    name="protoecg-pipeline",
    help="ProtoECG Foundation Model training pipeline (CLIF v2.1.0).",
    no_args_is_help=True,
)
logger = get_logger("cli")


def _load_config(config_path: Path | None = None):
    from clif_protoecg.config import PipelineConfig

    if config_path and config_path.exists():
        return PipelineConfig.from_yaml(config_path)
    return PipelineConfig()


# ──────────────────────── Step 1-2: Extract ──────────────────────────


@app.command()
def extract(
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to pipeline config YAML")
    ] = None,
    n_patients: Annotated[
        Optional[int], typer.Option("--n-patients", "-n", help="Limit to N patients")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Number of parallel workers for post-processing (0=auto)")
    ] = 0,
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", help="Hospitalizations per chunk (0=use config default)")
    ] = 0,
) -> None:
    """Step 1-2: Extract CLIF data → MEDS sequences + labels."""
    import gc
    import os
    import time

    import polars as pl
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from clif_protoecg.data.loader import CLIFDataLoader
    from clif_protoecg.data.split import create_patient_splits
    from clif_protoecg.stages.profile import profile_codes
    from clif_protoecg.stages.bin_edges import compute_bin_edges
    from clif_protoecg.stages.extract_batch import build_all_base_events
    from clif_protoecg.labels.evaluator import LabelEvaluator
    from clif_protoecg.stages.ecg_prototypes import ECGPrototypeTokenizer
    from clif_protoecg.data.id_mapper import IDMapper
    from clif_protoecg.core.modalities import ECGRoute

    t0 = time.time()
    cfg = _load_config(config)
    if n_patients is not None:
        cfg.extraction.n_patients = n_patients
        cfg.data.output_dir = cfg.data.output_dir / f"n{n_patients}"

    effective_chunk_size = chunk_size if chunk_size > 0 else cfg.extraction.chunk_size

    loader = CLIFDataLoader(cfg.data.data_dir)
    logger.info("Loading hospitalization table...")
    hosp_df = loader.load("hospitalization")
    logger.info(f"Loaded {len(hosp_df):,} hospitalizations")

    # ── Split patients ──────────────────────────────────────────────
    logger.info("Splitting patients...")
    splits = create_patient_splits(hosp_df, val_fraction=cfg.split.val_fraction, seed=cfg.split.seed)
    metadata_dir = cfg.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits.save(metadata_dir / "splits.json")
    logger.info(f"Split: train={len(splits.train):,}, val={len(splits.val):,}, test={len(splits.test):,}")

    # Limit patients if requested — sample proportionally from each split
    if n_patients is not None:
        total = len(splits.train) + len(splits.val) + len(splits.test)
        frac_train = len(splits.train) / total
        frac_val = len(splits.val) / total
        n_train = max(1, int(n_patients * frac_train))
        n_val = max(1, int(n_patients * frac_val))
        n_test = n_patients - n_train - n_val
        sampled_train = splits.train[:n_train]
        sampled_val = splits.val[:n_val]
        sampled_test = splits.test[:n_test]
        all_patients = sampled_train + sampled_val + sampled_test
        train_patients = sampled_train
        logger.info(
            f"Sampled {n_patients:,} patients: "
            f"train={len(sampled_train):,}, val={len(sampled_val):,}, test={len(sampled_test):,}"
        )
    else:
        all_patients = splits.train + splits.val + splits.test
        train_patients = splits.train

    logger.info(f"Resolving hospitalization IDs for {len(all_patients):,} patients...")
    if n_patients is None:
        # All patients — just get all hosp IDs without expensive is_in filter
        all_hosp_ids = hosp_df["hospitalization_id"].unique().sort().to_list()
        train_hosp_ids = (
            hosp_df.filter(pl.col("patient_id").is_in(train_patients))
            ["hospitalization_id"].unique().sort().to_list()
        )
    else:
        all_hosp_ids = loader.get_hospitalization_ids(all_patients)
        train_hosp_ids = loader.get_hospitalization_ids(train_patients)
    logger.info(
        f"Patients: {len(all_patients):,}, hospitalizations: {len(all_hosp_ids):,}, "
        f"train hosps: {len(train_hosp_ids):,}"
    )

    # ── Profile codes (no preload needed — uses lazy scan) ──────────
    logger.info("Profiling codes...")
    profile = profile_codes(loader, train_hosp_ids)
    profile.save(metadata_dir / "code_profile.json")

    # ── Bin edges (no preload needed — uses lazy scan) ──────────────
    logger.info("Computing bin edges...")
    bin_edges = compute_bin_edges(loader, train_hosp_ids, bin_mode=cfg.extraction.bin_mode, n_bins=cfg.extraction.n_bins)
    bin_edges.save(metadata_dir / "bin_edges.json")

    # ── Labels ──────────────────────────────────────────────────────
    label_eval = LabelEvaluator(loader, categories=cfg.labels.categories)
    logger.info(f"Labels: {len(label_eval.label_names)} definitions")

    # ── ECG prototypes (small — keep in memory) ────────────────────
    ecg_tokenizer = ECGPrototypeTokenizer(route=ECGRoute.ALL_BRANCHES)
    id_mapper = IDMapper.from_clif_tables(hosp_df)
    ecg_csv = Path(cfg.data.ecg_csv)
    ecg_df = None
    if ecg_csv.exists():
        logger.info("Loading ECG prototypes...")
        ecg_df = ecg_tokenizer.load_prototypes(
            str(ecg_csv),
            id_mapping=id_mapper.hadm_to_hosp,
            hosp_df=hosp_df,
            lookback_days=cfg.ecg.lookback_days,
        )
        ecg_tokenizer.compute_similarity_quantiles(ecg_df, n_bins=cfg.ecg.n_similarity_bins)
        logger.info(f"ECG records: {len(ecg_df):,} (lookback={cfg.ecg.lookback_days}d)")
    else:
        logger.warning(f"ECG CSV not found at {ecg_csv}, skipping ECG tokens")

    # ── Pre-build lookup dicts (small — keep for all hosps) ────────
    hosp_to_patient: dict[str, str] = dict(
        hosp_df.select("hospitalization_id", "patient_id").iter_rows()
    )
    hosp_lookup: dict[str, dict] = {
        row["hospitalization_id"]: row
        for row in hosp_df.filter(
            pl.col("hospitalization_id").is_in(all_hosp_ids)
        ).iter_rows(named=True)
    }
    patient_df = loader.load("patient")
    patient_lookup: dict[str, dict] = {
        row["patient_id"]: row
        for row in patient_df.iter_rows(named=True)
    }

    # ── Determine chunks ───────────────────────────────────────────
    chunks = [
        all_hosp_ids[i : i + effective_chunk_size]
        for i in range(0, len(all_hosp_ids), effective_chunk_size)
    ]
    n_chunks = len(chunks)
    logger.info(
        f"Processing {len(all_hosp_ids):,} hospitalizations in {n_chunks} chunk(s) "
        f"(chunk_size={effective_chunk_size:,})"
    )

    # ── Shared state for postprocessing ────────────────────────────
    from clif_protoecg.stages.postprocess import (
        init_context, process_hospitalization, build_task,
    )

    _LABEL_TABLES = [
        "adt", "vitals", "labs", "medication_admin_continuous",
        "respiratory_support", "patient_assessments",
        "position", "crrt_therapy", "ecmo_mcs",
    ]

    output_dir = cfg.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ecg_stats = {"hosps_with_ecg": 0, "total_ecg_events": 0, "total_ecg_records": 0}
    label_stats = {"hosps_with_labels": 0, "total_labels": 0}
    total_events = 0
    skipped = 0
    total_sequences = 0
    chunk_paths: list[Path] = []

    hosp_df_empty = hosp_df.clear()

    # Init shared context once (only label_eval + ecg_tokenizer — small)
    init_context(label_eval=label_eval, ecg_tokenizer=ecg_tokenizer)

    # ── Process each chunk ─────────────────────────────────────────
    for chunk_idx, chunk_hids in enumerate(chunks):
        t_chunk = time.time()
        logger.info(
            f"── Chunk {chunk_idx + 1}/{n_chunks}: "
            f"{len(chunk_hids):,} hospitalizations ──"
        )

        # 1. Preload tables for this chunk only
        loader.clear_preload()
        logger.info("Pre-loading and partitioning tables...")
        loader.preload(chunk_hids)

        # 2. Build base events (vectorized)
        logger.info("Building base events (vectorized)...")
        t_extract = time.time()
        chunk_sequences = build_all_base_events(
            chunk_hids, loader, hosp_lookup, patient_lookup, hosp_to_patient,
            profile,
        )
        chunk_base_events = sum(len(seq) for seq in chunk_sequences.values())
        logger.info(
            f"Base events: {chunk_base_events:,} in {time.time() - t_extract:.1f}s"
        )

        # 3. Build chunk-specific partitions for postprocessing

        # ECG partition
        ecg_partition: dict[str, pl.DataFrame] = {}
        if ecg_df is not None and "hospitalization_id" in ecg_df.columns:
            ecg_filtered = ecg_df.filter(pl.col("hospitalization_id").is_in(chunk_hids))
            for group_df in ecg_filtered.partition_by("hospitalization_id"):
                if len(group_df) > 0:
                    ecg_partition[group_df["hospitalization_id"][0]] = group_df
            logger.info(f"ECG partition: {len(ecg_partition)} hosps ({len(ecg_filtered):,} rows)")

        # hosp_single_row
        hosp_single_row: dict[str, pl.DataFrame] = {}
        for group_df in hosp_df.filter(
            pl.col("hospitalization_id").is_in(chunk_hids)
        ).partition_by("hospitalization_id"):
            if len(group_df) > 0:
                hosp_single_row[group_df["hospitalization_id"][0]] = group_df

        # code_status partition
        cs_partition: dict[str, pl.DataFrame] = {}
        cs_empty: pl.DataFrame | None = None
        try:
            cs_full = loader.load("code_status")
            if "hospitalization_id" in cs_full.columns:
                cs_filtered = cs_full.filter(pl.col("hospitalization_id").is_in(chunk_hids))
                for group_df in cs_filtered.partition_by("hospitalization_id"):
                    if len(group_df) > 0:
                        cs_partition[group_df["hospitalization_id"][0]] = group_df
                cs_empty = cs_filtered.clear()
            else:
                pids = list({hosp_to_patient[h] for h in chunk_hids if h in hosp_to_patient})
                cs_filtered = cs_full.filter(pl.col("patient_id").is_in(pids))
                for group_df in cs_filtered.partition_by("patient_id"):
                    if len(group_df) > 0:
                        cs_partition[group_df["patient_id"][0]] = group_df
                cs_empty = cs_filtered.clear()
        except FileNotFoundError:
            pass

        # Label partitions (from preloaded data)
        label_partitions: dict[str, tuple[dict[str, pl.DataFrame], pl.DataFrame]] = {}
        for tbl_name in _LABEL_TABLES:
            if tbl_name in loader._partitions:
                partition, empty = loader._partitions[tbl_name]
                label_partitions[f"clif_{tbl_name}"] = (partition, empty)

        # 4. Build task iterator for postprocessing
        cs_by_hosp = cs_empty is not None and "hospitalization_id" in cs_empty.columns

        def _task_iter():
            for hid in chunk_hids:
                yield build_task(
                    hid,
                    all_sequences=chunk_sequences,
                    hosp_to_patient=hosp_to_patient,
                    hosp_lookup=hosp_lookup,
                    hosp_single_row=hosp_single_row,
                    label_partitions=label_partitions,
                    cs_partition=cs_partition,
                    cs_empty=cs_empty,
                    cs_by_hosp=cs_by_hosp,
                    ecg_partition=ecg_partition,
                    hosp_df_empty=hosp_df_empty,
                )

        chunk_events = 0

        def _collect_chunk_result(result: dict | None) -> None:
            nonlocal total_events, skipped, chunk_events
            if result is None:
                skipped += 1
                return
            hid = result["hid"]
            chunk_sequences[hid] = result["events"]
            n_evt = len(result["events"])
            total_events += n_evt
            chunk_events += n_evt
            if result["n_labels"] > 0:
                label_stats["hosps_with_labels"] += 1
                label_stats["total_labels"] += result["n_labels"]
            if result["n_ecg_records"] > 0:
                ecg_stats["hosps_with_ecg"] += 1
                ecg_stats["total_ecg_records"] += result["n_ecg_records"]
                ecg_stats["total_ecg_events"] += result["n_ecg_events"]

        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Chunk {chunk_idx + 1}/{n_chunks}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[dim]|"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            TextColumn("[dim]|"),
            TextColumn("[green]{task.fields[events]:,} events"),
        )

        if workers != 1:
            import multiprocessing as mp
            from clif_protoecg.stages.postprocess import save_context, _init_worker

            logger.info("Serializing shared context...")
            ctx_path = save_context()
            ctx_size_mb = os.path.getsize(ctx_path) / (1024 * 1024)

            MAX_WORKERS = 8
            if workers == 0:
                n_cpus = os.cpu_count() or 1
                try:
                    import psutil
                    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
                except ImportError:
                    avail_gb = 128.0
                # Context is now tiny (just label_eval + ecg_tokenizer).
                # Each worker uses ~200 MB base + task data in flight.
                worker_base_gb = 0.2
                max_by_ram = max(1, int((avail_gb * 0.75) / worker_base_gb) - 1)
                actual_workers = max(1, min(n_cpus, max_by_ram, MAX_WORKERS))
            else:
                actual_workers = min(workers, MAX_WORKERS)

            logger.info(
                f"Shared context: {ctx_size_mb:.1f} MB, "
                f"workers: {actual_workers} (cpus={os.cpu_count()})"
            )
            spawn_ctx = mp.get_context("spawn")
            with progress:
                prog_task = progress.add_task(
                    "postprocess", total=len(chunk_hids), events=0
                )
                with spawn_ctx.Pool(
                    processes=actual_workers,
                    initializer=_init_worker,
                    initargs=(ctx_path,),
                ) as pool:
                    for result in pool.imap_unordered(
                        process_hospitalization, _task_iter(), chunksize=64
                    ):
                        _collect_chunk_result(result)
                        progress.update(prog_task, advance=1, events=chunk_events)

            os.unlink(ctx_path)
        else:
            logger.info("Post-processing (single-threaded)...")
            with progress:
                prog_task = progress.add_task(
                    "postprocess", total=len(chunk_hids), events=0
                )
                for task in _task_iter():
                    result = process_hospitalization(task)
                    _collect_chunk_result(result)
                    progress.update(prog_task, advance=1, events=chunk_events)

        # 5. Save chunk sequences
        from clif_protoecg.data.sequences_io import save_sequences

        chunk_path = output_dir / f"_sequences_chunk_{chunk_idx}.parquet"
        save_sequences(chunk_sequences, chunk_path)
        total_sequences += len(chunk_sequences)
        chunk_paths.append(chunk_path)

        logger.info(
            f"Chunk {chunk_idx + 1} done: {chunk_events:,} events, "
            f"{len(chunk_sequences):,} sequences in {time.time() - t_chunk:.1f}s"
        )

        # 6. Free memory
        del chunk_sequences, ecg_partition, hosp_single_row
        del cs_partition, label_partitions
        loader.clear_preload()
        gc.collect()

    # ── Merge chunks ───────────────────────────────────────────────
    seq_path = output_dir / "sequences.parquet"
    if len(chunk_paths) == 1:
        # Single chunk — just rename
        chunk_paths[0].rename(seq_path)
        logger.info(f"Saved {total_sequences} sequences -> {seq_path}")
    else:
        from clif_protoecg.data.sequences_io import merge_sequence_files

        logger.info(f"Merging {len(chunk_paths)} chunk files...")
        n_rows = merge_sequence_files(chunk_paths, seq_path)
        logger.info(f"Merged {n_rows:,} rows -> {seq_path}")
        for p in chunk_paths:
            p.unlink()

    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────
    logger.info(f"Saved {total_sequences} sequences -> {seq_path}")
    avg_events = total_events / max(total_sequences, 1)
    logger.info(
        f"Events: {total_events:,} total, {avg_events:,.0f} avg/hosp"
    )
    if label_stats["total_labels"] > 0:
        logger.info(
            f"Labels: {label_stats['hosps_with_labels']}/{total_sequences} hosps, "
            f"{label_stats['total_labels']} label events"
        )
    if ecg_df is not None:
        logger.info(
            f"ECG: {ecg_stats['hosps_with_ecg']}/{total_sequences} hosps with ECGs, "
            f"{ecg_stats['total_ecg_records']} ECG records -> {ecg_stats['total_ecg_events']} tokens"
        )
    if skipped:
        logger.warning(f"Skipped {skipped} hospitalizations (missing patient data)")
    logger.info(f"Metadata -> {metadata_dir}")
    logger.info(f"Completed in {elapsed:.1f}s")


# ──────────────────────── Step 5: Tokenize ──────────────────────────


@app.command()
def tokenize(
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to pipeline config YAML")
    ] = None,
    n_patients: Annotated[
        Optional[int], typer.Option("--n-patients", "-n", help="Use output dir for N-patient test run")
    ] = None,
) -> None:
    """Step 5: Build vocabulary and tokenize event sequences to token IDs."""
    from clif_protoecg.data.split import SplitInfo
    from clif_protoecg.data.sequences_io import load_sequences
    from clif_protoecg.tokenization.stage import build_vocabulary_from_sequences, tokenize_and_write

    cfg = _load_config(config)
    if n_patients is not None:
        cfg.data.output_dir = cfg.data.output_dir / f"n{n_patients}"

    # Load sequences
    seq_path = cfg.processed_dir / "sequences.parquet"
    if not seq_path.exists():
        logger.error(f"No sequences found at {seq_path}. Run 'extract' first.")
        raise typer.Exit(1)

    all_sequences = load_sequences(seq_path)
    logger.info(f"Loaded {len(all_sequences)} sequences")

    # Load splits to build vocab from training data only
    splits_path = cfg.metadata_dir / "splits.json"
    from clif_protoecg.core.artifacts import SplitInfo as SI
    splits = SI.load(splits_path)

    from clif_protoecg.data.loader import CLIFDataLoader
    loader = CLIFDataLoader(cfg.data.data_dir)
    train_hosp_ids = set(loader.get_hospitalization_ids(splits.train))
    train_sequences = [seq for hid, seq in all_sequences.items() if hid in train_hosp_ids]
    logger.info(f"Building vocabulary from {len(train_sequences)} training sequences")

    vocab = build_vocabulary_from_sequences(
        train_sequences,
        n_bins=cfg.extraction.n_bins,
        fuse_bins=cfg.tokenization.fuse_numeric_bins,
        min_count=cfg.tokenization.min_token_count,
    )
    vocab_dir = cfg.tokenized_dir / "vocab"
    vocab.save(vocab_dir)
    logger.info(f"Vocabulary: {vocab.size} tokens -> {vocab_dir}")

    # Tokenize all sequences
    n = tokenize_and_write(
        all_sequences,
        vocab,
        cfg.tokenized_dir / "all_tokens.parquet",
        fuse_bins=cfg.tokenization.fuse_numeric_bins,
    )
    logger.info(f"Tokenized {n} sequences -> {cfg.tokenized_dir / 'all_tokens.parquet'}")


# ──────────────────────── Step 6: Train ──────────────────────────


@app.command()
def train(
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to pipeline config YAML")
    ] = None,
    size: Annotated[
        str, typer.Option("--size", "-s", help="Model size: tiny, small, medium, large")
    ] = "tiny",
    route: Annotated[
        str, typer.Option("--route", "-r", help="ECG route: no_ecg, fusion_class, all_branches")
    ] = "all_branches",
    n_patients: Annotated[
        Optional[int], typer.Option("--n-patients", "-n", help="Use output dir for N-patient test run")
    ] = None,
    wandb_project: Annotated[
        Optional[str], typer.Option("--wandb-project", help="wandb project name (None to disable)")
    ] = "protoecg-fm",
    no_wandb: Annotated[
        bool, typer.Option("--no-wandb", help="Disable wandb logging")
    ] = False,
) -> None:
    """Step 6: Train the foundation model."""
    import polars as pl

    from clif_protoecg.data.loader import CLIFDataLoader
    from clif_protoecg.core.artifacts import SplitInfo as SI
    from clif_protoecg.tokenization.vocabulary import Vocabulary
    from clif_protoecg.training.data.token_filter import ECGTokenFilter
    from clif_protoecg.training.stage import TrainingStage
    from clif_protoecg.training.trainer import TrainerConfig

    cfg = _load_config(config)
    if n_patients is not None:
        cfg.data.output_dir = cfg.data.output_dir / f"n{n_patients}"
    cfg.training.model_size = size
    cfg.ecg.route = route

    # Load vocab + tokenized data
    vocab_dir = cfg.tokenized_dir / "vocab"
    vocab = Vocabulary.load(vocab_dir)
    logger.info(f"Vocabulary: {vocab.size} tokens")

    tokens_path = cfg.tokenized_dir / "all_tokens.parquet"
    df = pl.read_parquet(tokens_path)
    logger.info(f"Loaded {len(df)} tokenized sequences")

    # Split by hospitalization
    splits = SI.load(cfg.metadata_dir / "splits.json")
    loader = CLIFDataLoader(cfg.data.data_dir)
    train_hosp_ids = set(loader.get_hospitalization_ids(splits.train))
    val_hosp_ids = set(loader.get_hospitalization_ids(splits.val))

    train_df = df.filter(pl.col("hospitalization_id").is_in(list(train_hosp_ids)))
    val_df = df.filter(pl.col("hospitalization_id").is_in(list(val_hosp_ids)))

    train_seqs = train_df["token_ids"].to_list()
    val_seqs = val_df["token_ids"].to_list()

    # Apply ECG token filter
    ecg_filter = ECGTokenFilter(vocab.token_to_id, route)
    train_seqs = [ecg_filter.filter(s) for s in train_seqs]
    val_seqs = [ecg_filter.filter(s) for s in val_seqs]
    logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Route: {route}")

    # Identify label token IDs
    label_token_ids = {tid for tok, tid in vocab.token_to_id.items() if tok.startswith("LABEL//")}

    # Build TrainerConfig
    effective_wandb = None if no_wandb else (wandb_project or cfg.training.wandb_project)
    trainer_config = TrainerConfig(
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        beta1=cfg.training.beta1,
        beta2=cfg.training.beta2,
        warmup_steps=cfg.training.warmup_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        num_epochs=cfg.training.num_epochs,
        label_loss_weight=cfg.training.label_loss_weight,
        early_stopping=cfg.training.early_stopping,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        eval_steps=getattr(cfg.training, "eval_steps", 500),
        bf16=cfg.training.bf16,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_seq_len=cfg.training.max_seq_len,
        chunk_overlap=getattr(cfg.training, "chunk_overlap", 256),
        max_tokens_per_batch=cfg.training.max_tokens_per_batch,
        pad_id=vocab.pad_id,
        val_max_batches=cfg.training.val_max_batches,
        wandb_project=effective_wandb,
        wandb_run_name=cfg.training.wandb_run_name or f"{route}_{size}",
    )

    stage = TrainingStage()
    checkpoint_dir = cfg.checkpoints_dir / f"{route}_{size}"
    result = stage.run(
        input_path=vocab_dir,
        output_path=checkpoint_dir,
        vocab=vocab,
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        config=trainer_config,
        size_name=size,
        label_token_ids=label_token_ids,
    )

    logger.info(f"Training done: {result.metrics}")
    logger.info(f"Model saved to {checkpoint_dir}")


# ──────────────────────── Step 7-8: Evaluate ──────────────────────────


@app.command()
def evaluate(
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to pipeline config YAML")
    ] = None,
    model_path: Annotated[
        Optional[Path], typer.Option("--model-path", "-m", help="Path to model checkpoint")
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option("--backend", "-b", help="Inference backend: native, vllm")
    ] = None,
    size: Annotated[
        str, typer.Option("--size", "-s", help="Model size")
    ] = "tiny",
    route: Annotated[
        str, typer.Option("--route", "-r", help="ECG route")
    ] = "all_branches",
    n_patients: Annotated[
        Optional[int], typer.Option("--n-patients", "-n", help="Use output dir for N-patient test run")
    ] = None,
) -> None:
    """Step 7-8: Run inference and evaluation."""
    import json
    from datetime import datetime, timezone

    import polars as pl
    import torch

    from clif_protoecg.core.artifacts import SplitInfo as SI
    from clif_protoecg.data.loader import CLIFDataLoader
    from clif_protoecg.tokenization.vocabulary import Vocabulary
    from clif_protoecg.training.data.token_filter import ECGTokenFilter
    from clif_protoecg.evaluation.stage import EvaluationStage

    cfg = _load_config(config)
    if n_patients is not None:
        cfg.data.output_dir = cfg.data.output_dir / f"n{n_patients}"
    if backend is not None:
        cfg.inference.backend = backend

    if model_path is None:
        model_path = cfg.checkpoints_dir / f"{route}_{size}" / "final"

    # Load vocab
    vocab_dir = cfg.tokenized_dir / "vocab"
    vocab = Vocabulary.load(vocab_dir)

    # Load tokenized data with timestamps
    tokens_path = cfg.tokenized_dir / "all_tokens.parquet"
    df = pl.read_parquet(tokens_path)
    logger.info(f"Loaded {len(df)} tokenized sequences")

    # Get test split
    splits = SI.load(cfg.metadata_dir / "splits.json")
    loader = CLIFDataLoader(cfg.data.data_dir)
    test_hosp_ids = set(loader.get_hospitalization_ids(splits.test))
    test_df = df.filter(pl.col("hospitalization_id").is_in(list(test_hosp_ids)))
    logger.info(f"Test set: {len(test_df)} sequences")

    if len(test_df) == 0:
        logger.warning("No test sequences found, exiting")
        return

    # Apply ECG token filter
    ecg_filter = ECGTokenFilter(vocab.token_to_id, route)

    # Build test data dicts with timestamps
    test_data: list[dict] = []
    for row in test_df.iter_rows(named=True):
        token_ids = ecg_filter.filter(row["token_ids"])
        # Convert microsecond epochs back to datetimes
        if "timestamps_us" in row and row["timestamps_us"] is not None:
            # Filter timestamps in parallel with token filtering
            raw_ts = row["timestamps_us"]
            if ecg_filter.filtered_ids:
                # Rebuild timestamps matching filtered token_ids
                ts_filtered = [
                    ts for tid, ts in zip(row["token_ids"], raw_ts)
                    if tid not in ecg_filter.filtered_ids
                ]
            else:
                ts_filtered = raw_ts
            timestamps = [
                datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)
                for us in ts_filtered
            ]
        else:
            logger.warning("No timestamps found — re-run 'tokenize' to generate them")
            return

        test_data.append({
            "hospitalization_id": row["hospitalization_id"],
            "token_ids": token_ids,
            "timestamps": timestamps,
        })

    # Identify label token IDs (name -> id mapping for evaluation)
    label_token_ids = {
        tok: tid for tok, tid in vocab.token_to_id.items()
        if tok.startswith("LABEL//")
    }
    logger.info(f"Labels: {list(label_token_ids.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Evaluation: model={model_path}, backend={cfg.inference.backend}, "
        f"device={device}, horizons={cfg.evaluation.horizons}, "
        f"n_samples={cfg.inference.n_samples}"
    )

    stage = EvaluationStage()
    eval_dir = cfg.data.output_dir / "evaluation" / f"{route}_{size}"
    result = stage.run(
        input_path=model_path,
        output_path=eval_dir,
        vocab=vocab,
        test_data=test_data,
        label_token_ids=label_token_ids,
        horizons=cfg.evaluation.horizons,
        n_samples=cfg.inference.n_samples,
        max_new_tokens=cfg.inference.max_new_tokens,
        backend=cfg.inference.backend,
        device=device,
    )

    if result.success:
        logger.info(f"Evaluation complete: {eval_dir / 'metrics.json'}")
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        for key, vals in metrics.items():
            auroc = vals.get("auroc")
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            logger.info(f"  {key}: AUROC={auroc_str}, n={vals['n_total']}")
    else:
        logger.error(f"Evaluation failed: {result.metrics}")


# ──────────────────────── Ablation ──────────────────────────


@app.command()
def ablation(
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to pipeline config YAML")
    ] = None,
    routes: Annotated[
        str, typer.Option("--routes", help="Comma-separated ECG routes")
    ] = "no_ecg,fusion_class,all_branches",
    sizes: Annotated[
        str, typer.Option("--sizes", help="Comma-separated model sizes")
    ] = "tiny",
    mode: Annotated[
        str, typer.Option("--mode", help="test or full")
    ] = "test",
    n_patients: Annotated[
        Optional[int], typer.Option("--n-patients", "-n", help="Limit patients for test mode")
    ] = 100,
) -> None:
    """Run ablation study across ECG routes and model sizes."""
    cfg = _load_config(config)
    route_list = [r.strip() for r in routes.split(",")]
    size_list = [s.strip() for s in sizes.split(",")]

    from clif_protoecg.ablation import AblationRunner

    runner = AblationRunner(cfg, route_list, size_list)
    logger.info(f"Ablation: routes={route_list}, sizes={size_list}, mode={mode}")

    if mode == "test" and n_patients is not None:
        cfg.extraction.n_patients = n_patients

    runner.run()


# ──────────────────────── Run All ──────────────────────────


@app.command()
def run_all(
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to pipeline config YAML")
    ] = None,
) -> None:
    """Run the full pipeline: extract -> tokenize -> train -> evaluate."""
    cfg = _load_config(config)
    logger.info("Running full pipeline")
    logger.info(f"Config: {cfg.model_dump(mode='json')}")


if __name__ == "__main__":
    app()
