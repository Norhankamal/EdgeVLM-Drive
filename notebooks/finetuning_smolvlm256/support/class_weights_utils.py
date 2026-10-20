"""
class_weights_utils.py
========================

WHAT THIS FILE DOES:
Provides a small helper function to compute per-class "weights" from a
training CSV, plus a short example of how to plug those weights into a
PyTorch loss function during LoRA fine-tuning.

WHY CLASS WEIGHTS MATTER HERE:
Our training data is imbalanced (e.g. NORMAL frames vastly outnumber
DROWSY frames). Without any correction, a model can reach deceptively
high overall accuracy just by predicting the majority class (NORMAL)
most of the time, while doing poorly on the classes that actually
matter most for safety (DROWSY, LOOKING_AWAY, PHONE_USAGE).

Class weighting fixes this by making mistakes on RARE classes count
MORE in the loss function during training, without changing the
dataset itself (unlike augmentation, which physically adds more
images). Using both together (augmentation + class weights) is stronger
than either alone: augmentation adds some visual variety for the rare
class, and class weighting further compensates for whatever imbalance
remains after augmentation.

HOW WEIGHTS ARE COMPUTED:
We use the standard "inverse frequency" formula:
    weight(class) = total_samples / (num_classes * count(class))
A rarer class ends up with a LARGER weight, so the loss function
penalizes mistakes on it more heavily.

HOW TO USE (as a library):
    from class_weights_utils import compute_class_weights
    weights, class_to_idx = compute_class_weights("splits/train_augmented.csv")
    # weights is a list of floats in the same order as class_to_idx

HOW TO USE (standalone, just to print the weights):
    python class_weights_utils.py --train-csv splits/train_augmented.csv
"""

import argparse
import csv
from collections import Counter


CLASSES = ["NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE"]


def compute_class_weights(train_csv_path, classes=CLASSES):
    """
    Read a training CSV and compute one weight per class using the
    inverse-frequency formula. Returns:
        weights: list of floats, in the same order as `classes`
        class_to_idx: dict mapping class name -> index (0..num_classes-1)
    """
    with open(train_csv_path, newline="") as f:
        counts = Counter(row["label"] for row in csv.DictReader(f))

    total = sum(counts.get(c, 0) for c in classes)
    num_classes = len(classes)

    weights = []
    for c in classes:
        count = counts.get(c, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    return weights, class_to_idx


def print_weights_table(train_csv_path):
    weights, class_to_idx = compute_class_weights(train_csv_path)
    print(f"Class weights computed from: {train_csv_path}\n")
    print(f"{'Class':15s} {'Weight':>8s}")
    for c, idx in class_to_idx.items():
        print(f"{c:15s} {weights[idx]:8.3f}")
    return weights, class_to_idx


# ---------------------------------------------------------------------
# EXAMPLE: how to actually use these weights in a PyTorch training loop
# ---------------------------------------------------------------------
def example_pytorch_usage(train_csv_path):
    """
    This function is NOT meant to be run as-is (there's no real model
    here) -- it's a documented template showing exactly where the
    computed class weights plug into a standard PyTorch training setup,
    for whoever writes the actual LoRA fine-tuning loop.
    """
    import torch
    import torch.nn as nn

    weights, class_to_idx = compute_class_weights(train_csv_path)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # If you have a GPU, move the weights tensor to the same device as
    # your model and inputs, e.g.:
    #   weights_tensor = weights_tensor.to(device)

    # Plug the weights directly into the loss function. Every other part
    # of the training loop (forward pass, backward pass, optimizer step)
    # stays exactly the same as an unweighted setup.
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

    # --- inside your training loop, unchanged except for loss_fn above ---
    #   logits = model(images)                  # shape: [batch, num_classes]
    #   labels = ...                            # shape: [batch], integer class ids
    #   loss = loss_fn(logits, labels)
    #   loss.backward()
    #   optimizer.step()

    return loss_fn, class_to_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and print class weights from a training CSV."
    )
    parser.add_argument("--train-csv", required=True,
                         help="Path to train.csv or train_augmented.csv")
    args = parser.parse_args()

    print_weights_table(args.train_csv)
