"""
NetOracle - Real CIC-IDS-2018 World Model Training

Split:
    TRAIN:
        Wednesday-14
        Friday-16
        Friday-23
        Thursday-22
        Wednesday-21

    VALIDATION:
        Thursday-15

    TEST:
        Wednesday-28

No synthetic data.
Test data is never used for training/model selection.
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.feature_extraction import FeatureExtractor
from src.world_model import NetworkWorldModel


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TRAIN_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-INFIL-TRAIN.csv",
]

VALIDATION_FILES = [
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
]

TEST_FILES = [
    "Wednesday-28-INFIL-TEST.csv",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(
    data,
    batch_size,
    shuffle=False,
):

    X, y_states, y_attacks, y_mitre = data

    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y_states).float(),
        torch.from_numpy(y_attacks).float(),
        torch.from_numpy(y_mitre).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    return loader


# ============================================================
# COLLECT MODEL PREDICTIONS
# ============================================================

def collect_predictions(
    model,
    loader,
    device,
):

    model.eval()

    attack_probabilities = []
    attack_targets = []

    stage_predictions = []
    stage_targets = []

    with torch.no_grad():

        for (
            X,
            _,
            y_attack,
            y_stage,
        ) in loader:

            X = X.to(
                device,
                non_blocking=True,
            )

            output = model(X)

            probabilities = (
                output["attack_probs"]
                .detach()
                .cpu()
                .numpy()
            )

            attack_probabilities.append(
                probabilities
            )

            attack_targets.append(
                y_attack.numpy()
            )

            predicted_stages = (
                output["mitre_logits"]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

            stage_predictions.append(
                predicted_stages
            )

            stage_targets.append(
                y_stage.numpy()
            )

    attack_probabilities = np.concatenate(
        attack_probabilities,
        axis=0,
    )

    attack_targets = np.concatenate(
        attack_targets,
        axis=0,
    ).astype(int)

    stage_predictions = np.concatenate(
        stage_predictions,
        axis=0,
    )

    stage_targets = np.concatenate(
        stage_targets,
        axis=0,
    ).astype(int)

    return (
        attack_probabilities,
        attack_targets,
        stage_predictions,
        stage_targets,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    probabilities,
    targets,
    max_fpr=0.10,
):
    """
    Select threshold using validation data only.

    Priority:
    1. Keep false-positive rate <= max_fpr.
    2. Among valid thresholds, maximize F1.
    """

    probabilities = probabilities[:, 0]
    targets = targets[:, 0]

    best_threshold = 0.5
    best_f1 = -1.0
    best_fpr = 1.0

    for threshold in np.arange(
        0.05,
        0.96,
        0.01,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        cm = confusion_matrix(
            targets,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        f1 = f1_score(
            targets,
            predictions,
            zero_division=0,
        )

        if (
            fpr <= max_fpr
            and f1 > best_f1
        ):
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_fpr = float(fpr)

    # Fallback if no threshold meets target.
    if best_f1 < 0:
        best_threshold = 0.5

        predictions = (
            probabilities >= best_threshold
        ).astype(int)

        best_f1 = float(
            f1_score(
                targets,
                predictions,
                zero_division=0,
            )
        )

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# BINARY METRICS
# ============================================================

def binary_metrics(
    probabilities,
    targets,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    if len(np.unique(targets)) > 1:

        auc = roc_auc_score(
            targets,
            probabilities,
        )

    else:

        auc = 0.5

    cm = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "auc_roc":
            float(auc),

        "false_positive_rate":
            float(fpr),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp),
    }


# ============================================================
# FORECAST-HORIZON METRICS
# ============================================================

def calculate_horizon_metrics(
    probabilities,
    targets,
    threshold,
):

    number_horizons = (
        probabilities.shape[1]
    )

    result = {}

    for horizon in range(
        number_horizons
    ):

        result[
            f"T+{horizon + 1}"
        ] = binary_metrics(

            probabilities[:, horizon],

            targets[:, horizon],

            threshold,
        )

    return result


# ============================================================
# MITRE STAGE METRICS
# ============================================================

def calculate_stage_metrics(
    predictions,
    targets,
):

    predictions_flat = (
        predictions.flatten()
    )

    targets_flat = (
        targets.flatten()
    )

    accuracy = accuracy_score(
        targets_flat,
        predictions_flat,
    )

    macro_f1 = f1_score(
        targets_flat,
        predictions_flat,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets_flat,
        predictions_flat,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),
    }


# ============================================================
# MAIN
# ============================================================

def run():

    set_seed()

    print("=" * 72)

    print(
        "NETORACLE - REAL CIC-IDS-2018 WORLD MODEL"
    )

    print("=" * 72)

    # ========================================================
    # CONFIG
    # ========================================================

    config_path = (
        "configs/config.yaml"
    )

    if not os.path.isfile(
        config_path
    ):

        raise FileNotFoundError(
            "configs/config.yaml not found."
        )

    with open(
        config_path,
        "r",
    ) as file:

        config = yaml.safe_load(file)

    # ========================================================
    # GPU ONLY
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is unavailable. "
            "Training stopped."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA runtime:",
        torch.version.cuda,
    )

    # Allow TensorFloat32 on modern NVIDIA GPUs.
    torch.set_float32_matmul_precision(
        "high"
    )

    # ========================================================
    # DISPLAY SPLIT
    # ========================================================

    print(
        "\nTRAINING DAYS:"
    )

    for filename in TRAIN_FILES:

        print(
            "  -",
            filename,
        )

    print(
        "\nVALIDATION DAY:"
    )

    for filename in VALIDATION_FILES:

        print(
            "  -",
            filename,
        )

    print(
        "\nUNSEEN TEST DAY:"
    )

    for filename in TEST_FILES:

        print(
            "  -",
            filename,
        )

    # ========================================================
    # PREPROCESS REAL DATA
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE 1: REAL DATA PREPROCESSING"
    )

    print(
        "=" * 72
    )

    extractor = FeatureExtractor(
        config_path
    )

    data = extractor.process_split(

        data_folder="data/raw",

        train_files=
            TRAIN_FILES,

        validation_files=
            VALIDATION_FILES,

        test_files=
            TEST_FILES,

        flows_per_window=250,
    )

    train_data = (
        data["train"]
    )

    validation_data = (
        data["validation"]
    )

    test_data = (
        data["test"]
    )

    feature_names = (
        data["feature_names"]
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    train_attack_ratio = (
        train_data[2].mean()
    )

    validation_attack_ratio = (
        validation_data[2].mean()
    )

    test_attack_ratio = (
        test_data[2].mean()
    )

    print(
        "\nAttack ratios:"
    )

    print(
        f"Train      : "
        f"{train_attack_ratio:.2%}"
    )

    print(
        f"Validation : "
        f"{validation_attack_ratio:.2%}"
    )

    print(
        f"Test       : "
        f"{test_attack_ratio:.2%}"
    )

    if validation_attack_ratio == 0:

        raise RuntimeError(
            "Validation contains zero attacks. "
            "Split is not suitable."
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    batch_size = int(
        config["model"][
            "batch_size"
        ]
    )

    train_loader = create_loader(
        train_data,
        batch_size,
        shuffle=True,
    )

    validation_loader = (
        create_loader(
            validation_data,
            batch_size,
            shuffle=False,
        )
    )

    test_loader = create_loader(
        test_data,
        batch_size,
        shuffle=False,
    )

    # ========================================================
    # MODEL
    # ========================================================

    input_dim = (
        train_data[0]
        .shape[-1]
    )

    model = NetworkWorldModel(

        input_dim=input_dim,

        state_dim=int(
            config["model"][
                "state_dim"
            ]
        ),

        hidden_dim=int(
            config["model"][
                "hidden_dim"
            ]
        ),

        num_heads=int(
            config["model"][
                "num_heads"
            ]
        ),

        num_layers=int(
            config["model"][
                "num_layers"
            ]
        ),

        forecast_horizon=int(
            config["data"][
                "forecast_horizon"
            ]
        ),

        num_mitre_stages=7,

        dropout=float(
            config["model"][
                "dropout"
            ]
        ),

    ).to(device)

    parameter_count = sum(

        p.numel()

        for p
        in model.parameters()

        if p.requires_grad
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE 2: WORLD MODEL"
    )

    print(
        "=" * 72
    )

    print(
        f"Input dimension   : "
        f"{input_dim}"
    )

    print(
        f"Parameters        : "
        f"{parameter_count:,}"
    )

    print(
        f"Sequence length   : "
        f"{config['data']['sequence_length']}"
    )

    print(
        f"Forecast horizon  : "
        f"{config['data']['forecast_horizon']}"
    )

    # ========================================================
    # TRAINING OBJECTIVES
    # ========================================================

    state_loss_fn = (
        nn.SmoothL1Loss()
    )

    attack_loss_fn = (
        nn.BCELoss()
    )

    stage_loss_fn = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=float(
                config["model"][
                    "learning_rate"
                ]
            ),

            weight_decay=1e-4,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=2,

            min_lr=1e-6,
        )
    )

    # ========================================================
    # TRAINING SETTINGS
    # ========================================================

    epochs = int(
        config["model"][
            "epochs"
        ]
    )

    EARLY_STOP_PATIENCE = 5

    epochs_without_improvement = 0

    best_validation_auc = -1.0

    best_epoch = 0

    os.makedirs(
        "models",
        exist_ok=True,
    )

    checkpoint_path = (
        "models/"
        "best_world_model.pth"
    )

    history = []

    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE 3: TEMPORAL WORLD MODEL TRAINING"
    )

    print(
        "=" * 72
    )

    for epoch in range(
        epochs
    ):

        model.train()

        total_loss_sum = 0.0

        state_loss_sum = 0.0

        attack_loss_sum = 0.0

        stage_loss_sum = 0.0

        batches = 0

        for (
            X,
            y_state,
            y_attack,
            y_stage,
        ) in train_loader:

            X = X.to(
                device,
                non_blocking=True,
            )

            y_state = y_state.to(
                device,
                non_blocking=True,
            )

            y_attack = y_attack.to(
                device,
                non_blocking=True,
            )

            y_stage = y_stage.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(X)

            # ------------------------------------------------
            # Future network-state prediction
            # ------------------------------------------------

            state_loss = (
                state_loss_fn(

                    output[
                        "predicted_states"
                    ],

                    y_state,
                )
            )

            # ------------------------------------------------
            # Attack probability forecasting
            # ------------------------------------------------

            attack_loss = (
                attack_loss_fn(

                    output[
                        "attack_probs"
                    ],

                    y_attack,
                )
            )

            # ------------------------------------------------
            # MITRE stage prediction
            # ------------------------------------------------

            stage_loss = (
                stage_loss_fn(

                    output[
                        "mitre_logits"
                    ]
                    .reshape(
                        -1,
                        7
                    ),

                    y_stage
                    .reshape(-1),
                )
            )

            # ------------------------------------------------
            # Multi-task objective
            # ------------------------------------------------

            loss = (

                1.0
                * state_loss

                + 2.0
                * attack_loss

                + 1.0
                * stage_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss_sum += (
                loss.item()
            )

            state_loss_sum += (
                state_loss.item()
            )

            attack_loss_sum += (
                attack_loss.item()
            )

            stage_loss_sum += (
                stage_loss.item()
            )

            batches += 1

        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_probabilities,
            val_targets,
            _,
            _,
        ) = collect_predictions(
            model,
            validation_loader,
            device,
        )

        # Fixed threshold during model selection.
        val_metrics_050 = (
            binary_metrics(

                val_probabilities[
                    :, 0
                ],

                val_targets[
                    :, 0
                ],

                threshold=0.5,
            )
        )

        validation_auc = (
            val_metrics_050[
                "auc_roc"
            ]
        )

        scheduler.step(
            validation_auc
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        average_total = (
            total_loss_sum
            / batches
        )

        average_state = (
            state_loss_sum
            / batches
        )

        average_attack = (
            attack_loss_sum
            / batches
        )

        average_stage = (
            stage_loss_sum
            / batches
        )

        history_entry = {

            "epoch":
                epoch + 1,

            "total_loss":
                average_total,

            "state_loss":
                average_state,

            "attack_loss":
                average_attack,

            "stage_loss":
                average_stage,

            "validation_auc":
                validation_auc,

            "validation_f1_at_0.5":
                val_metrics_050[
                    "f1_score"
                ],

            "learning_rate":
                current_lr,
        }

        history.append(
            history_entry
        )

        print(

            f"Epoch "
            f"[{epoch + 1:02d}/{epochs}] "

            f"| Loss "
            f"{average_total:.4f} "

            f"| State "
            f"{average_state:.4f} "

            f"| Attack "
            f"{average_attack:.4f} "

            f"| Stage "
            f"{average_stage:.4f} "

            f"| Val AUC "
            f"{validation_auc:.4f} "

            f"| Val F1@0.5 "
            f"{val_metrics_050['f1_score']:.4f} "

            f"| LR "
            f"{current_lr:.6f}"
        )

        # ====================================================
        # CHECKPOINT SELECTION
        # ====================================================

        if (
            validation_auc
            > best_validation_auc
            + 1e-5
        ):

            best_validation_auc = (
                validation_auc
            )

            best_epoch = (
                epoch + 1
            )

            epochs_without_improvement = 0

            torch.save(

                {
                    "model_state":
                        model.state_dict(),

                    "config":
                        config,

                    "input_dim":
                        input_dim,

                    "feature_names":
                        feature_names,

                    "dataset":
                        "CIC-IDS-2018",

                    "synthetic_data_used":
                        False,

                    "train_files":
                        TRAIN_FILES,

                    "validation_files":
                        VALIDATION_FILES,

                    "test_files":
                        TEST_FILES,

                    "best_epoch":
                        best_epoch,

                    "best_validation_auc":
                        best_validation_auc,

                    "scaler_mean":
                        data["scaler"].mean_,

                    "scaler_scale":
                        data["scaler"].scale_,
                },

                checkpoint_path,
            )

            print(
                "    [+] New best "
                "validation checkpoint."
            )

        else:

            epochs_without_improvement += 1

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            print(
                "\n[*] Early stopping."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\n[*] Loading best model..."
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    # ========================================================
    # SELECT THRESHOLD USING VALIDATION ONLY
    # ========================================================

    (
        validation_probabilities,
        validation_targets,
        validation_stage_predictions,
        validation_stage_targets,
    ) = collect_predictions(

        model,

        validation_loader,

        device,
    )

    (
        selected_threshold,
        validation_threshold_f1,
    ) = find_best_threshold(

        validation_probabilities,

        validation_targets,
    )

    validation_metrics = (
        calculate_horizon_metrics(

            validation_probabilities,

            validation_targets,

            selected_threshold,
        )
    )

    validation_stage_metrics = (
        calculate_stage_metrics(

            validation_stage_predictions,

            validation_stage_targets,
        )
    )

    print(
        f"\n[*] Validation-selected "
        f"threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"[*] Validation T+1 F1: "
        f"{validation_threshold_f1:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    # Test data has never been used for:
    #
    # - normalization fitting
    # - gradient updates
    # - checkpoint selection
    # - learning-rate decisions
    # - threshold selection

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE 4: UNSEEN-DAY TEST"
    )

    print(
        "=" * 72
    )

    (
        test_probabilities,
        test_targets,
        test_stage_predictions,
        test_stage_targets,
    ) = collect_predictions(

        model,

        test_loader,

        device,
    )

    test_metrics = (
        calculate_horizon_metrics(

            test_probabilities,

            test_targets,

            selected_threshold,
        )
    )

    test_stage_metrics = (
        calculate_stage_metrics(

            test_stage_predictions,

            test_stage_targets,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "dataset":
            "CIC-IDS-2018",

        "synthetic_data_used":
            False,

        "methodology":
            (
                "5 training days, "
                "1 validation day, "
                "1 unseen infiltration test day"
            ),

        "train_files":
            TRAIN_FILES,

        "validation_files":
            VALIDATION_FILES,

        "test_files":
            TEST_FILES,

        "best_epoch":
            best_epoch,

        "best_validation_auc":
            float(
                best_validation_auc
            ),

        "selected_threshold":
            float(
                selected_threshold
            ),

        "validation_metrics":
            validation_metrics,

        "validation_stage_metrics":
            validation_stage_metrics,

        "unseen_test_metrics":
            test_metrics,

        "unseen_test_stage_metrics":
            test_stage_metrics,

        "training_history":
            history,
    }

    with open(

        "models/"
        "evaluation_results.json",

        "w",

    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation AUC: "
        f"{best_validation_auc:.4f}"
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        "\nUNSEEN TEST FORECAST"
    )

    for (
        horizon,
        metrics
    ) in test_metrics.items():

        print(
            f"\n{horizon}:"
        )

        print(
            f"  Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1       : "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"  AUC-ROC  : "
            f"{metrics['auc_roc']:.4f}"
        )

        print(
            f"  FPR      : "
            f"{metrics['false_positive_rate']:.4f}"
        )

    print(
        "\nMITRE STAGE FORECAST"
    )

    print(
        f"Accuracy   : "
        f"{test_stage_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1   : "
        f"{test_stage_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1: "
        f"{test_stage_metrics['weighted_f1']:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
    "The model was exposed only to the earlier "
    "Wednesday-28 infiltration segment during training."
    )

    print(
    "The final test uses a later chronological segment "
    "separated by a 25,000-flow guard gap."
    )

    print(
        "Therefore this test measures "
        "unseen attack/day generalization."
    )

    print(
        "\n[+] Best model saved:"
    )

    print(
        "    models/best_world_model.pth"
    )

    print(
        "[+] Results saved:"
    )

    print(
        "    models/evaluation_results.json"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    run()