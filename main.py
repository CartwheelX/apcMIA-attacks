import os
import sys
import math
import csv
from datetime import datetime
import argparse
import functools
import multiprocessing as mp

# Third-party
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
import wandb
from datetime import datetime
import pandas as pd

# Local modules
from meminf import *
from dataloader import *




def get_attack_dataset(train_set, test_set, batch_size):
    mem_length = int(len(train_set)*0.45)
    nonmem_length = int(len(test_set)*0.45)
    mem_train, mem_test, _ = torch.utils.data.random_split(train_set, [mem_length, mem_length, len(train_set)-(mem_length*2)])
    nonmem_train, nonmem_test, _ = torch.utils.data.random_split(test_set, [nonmem_length, nonmem_length, len(test_set)-(nonmem_length*2)])
    mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(nonmem_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)
        
    attack_train = mem_train + nonmem_train
    attack_test = mem_test + nonmem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)

    return attack_trainloader, attack_testloader

    
def target_train_func(PATH, device, train_set, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch, epochs, patience):
    print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    print(f"dataset Name: {dataset_name}")
    # Create a small stratified validation split from the training set (e.g., 10%)
    from sklearn.model_selection import train_test_split
    import numpy as np
    indices = np.arange(len(train_set))
    labels = []
    for idx in indices:
        _, y = train_set[idx]
        if isinstance(y, (list, tuple)):
            y = y[0]
        if torch.is_tensor(y):
            y = int(y.item())
        labels.append(int(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=47)
    train_sub = torch.utils.data.Subset(train_set, train_idx.tolist())
    val_sub = torch.utils.data.Subset(train_set, val_idx.tolist())

    train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Use val_loader during training; evaluate test once at the end with best checkpoint
    model = target_train_class(train_loader, val_loader, dataset_name,  target_model, device, use_DP, noise, norm, delta, arch, batch_size)
    

     # Training loop with early stopping based on validation accuracy
    
    acc_train = 0.0
    acc_test = 0.0
    # early stopping and best checkpoint tracking on validation accuracy
    # patience = 10
    epochs_no_improve = 0
    best_val = 0.0
    best_train = 0.0
    best_overfitting = None
    FILE_PATH_target = PATH + "_target.pth"
    for i in range(epochs):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")
        acc_train, acc_val = model.train()
        # print("target testing (validation)")
        # acc_val, val_loss = model.test()
        overfitting = round(acc_train - acc_val, 6)
        print('The overfitting rate is %s' % overfitting)
        # save best checkpoint by validation accuracy
        if acc_val > best_val:
            best_val = acc_val
            best_train = acc_train
            best_overfitting = overfitting
            epochs_no_improve = 0
            model.saveModel(FILE_PATH_target)
            print(f"Saved best model so far at epoch {i+1} (val acc={best_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {i+1} after {patience} epochs without improvement.")
                break

    # Evaluate the best checkpoint once on the true test set
    try:
        state_dict = torch.load(FILE_PATH_target, map_location=device)
        model.net.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: failed to reload best checkpoint for final test eval: {e}")
    model.testloader = test_loader
    print("Evaluating on test set with best checkpoint...")
    final_test_acc = model.test()

    print("Saved target model!!! (best on val)")
    print("Finished training!!!")

     # Save the accuracies to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = PATH + f"_accs_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_train', 'best_val', 'overfitting_at_best', 'final_test'])
        # write best observed metrics (saved checkpoint) and final test
        writer.writerow([best_train if best_overfitting is not None else acc_train,
                         best_val if best_overfitting is not None else acc_val,
                         best_overfitting if best_overfitting is not None else overfitting,
                         final_test_acc])
    print(f"Saved accuracies to {csv_file}")

    return best_overfitting if best_overfitting is not None else overfitting


def shadow_train_func(PATH, device, train_set, test_set, shadow_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch, epochs, patience):
    # print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    # print(f"dataset Name: {dataset_name}")
    # Create a small stratified validation split from the training set (e.g., 10%)
    from sklearn.model_selection import train_test_split
    import numpy as np
    indices = np.arange(len(train_set))
    labels = []
    for idx in indices:
        _, y = train_set[idx]
        if isinstance(y, (list, tuple)):
            y = y[0]
        if torch.is_tensor(y):
            y = int(y.item())
        labels.append(int(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=47)
    train_sub = torch.utils.data.Subset(train_set, train_idx.tolist())
    val_sub = torch.utils.data.Subset(train_set, val_idx.tolist())

    train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Use val_loader during training; evaluate test once at the end with best checkpoint
    model = target_train_class(train_loader, val_loader, dataset_name,  shadow_model, device, use_DP, noise, norm, delta, arch, batch_size)
    

     # Training loop with early stopping based on validation accuracy
    
    acc_train = 0.0
    acc_test = 0.0
    # early stopping and best checkpoint tracking on validation accuracy
    # patience = 10
    epochs_no_improve = 0
    best_val = 0.0
    best_train = 0.0
    best_overfitting = None
    # FILE_PATH_target = PATH + "_target.pth"
    FILE_PATH_target = PATH + "_shadow.pth"

    for i in range(epochs):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("Shadow training")
        acc_train, acc_val = model.train()
        # print("target testing (validation)")
        # acc_val, val_loss = model.test()
        overfitting = round(acc_train - acc_val, 6)
        print('The overfitting rate is %s' % overfitting)
        # save best checkpoint by validation accuracy
        if acc_val > best_val:
            best_val = acc_val
            best_train = acc_train
            best_overfitting = overfitting
            epochs_no_improve = 0
            model.saveModel(FILE_PATH_target)
            print(f"Saved best model so far at epoch {i+1} (val acc={best_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {i+1} after {patience} epochs without improvement.")
                break

    # Evaluate the best checkpoint once on the true test set
    try:
        state_dict = torch.load(FILE_PATH_target, map_location=device)
        model.net.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: failed to reload best checkpoint for final test eval: {e}")
    model.testloader = test_loader
    print("Evaluating on test set with best checkpoint...")
    final_test_acc = model.test()

    print("Saved Shadow model!!! (best on val)")
    print("Finished Shadow training!!!")

     # Save the accuracies to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = PATH + f"_accs_{timestamp}_shad.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_train', 'best_val', 'overfitting_at_best', 'final_test'])
        # write best observed metrics (saved checkpoint) and final test
        writer.writerow([best_train if best_overfitting is not None else acc_train,
                         best_val if best_overfitting is not None else acc_val,
                         best_overfitting if best_overfitting is not None else overfitting,
                         final_test_acc])
    print(f"Saved accuracies to {csv_file}")

    return best_overfitting if best_overfitting is not None else overfitting


def target_train_func_dp(
    PATH, device, train_set, test_set, target_model, batch_size,
    use_DP, noise, norm, delta, dataset_name, arch, epochs, patience
):
    """
    Trains for exactly `epochs` epochs, saves the model ONCE at the end,
    and evaluates on the test set using the final weights.
    No early stopping; no 'best' checkpoint logic.
    """
    import torch, csv
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    import numpy as np

    print("Training model: train set shape", len(train_set), " test set shape:", len(test_set), ", device:", device)
    print(f"dataset Name: {dataset_name}")
    print("(Note) Early stopping and 'best checkpoint' are disabled; training runs for all epochs.")

    # --- stratified 90/10 train/val split from the provided train_set ---
    indices = np.arange(len(train_set))
    labels = []
    for idx in indices:
        _, y = train_set[idx]
        if isinstance(y, (list, tuple)):
            y = y[0]
        if torch.is_tensor(y):
            y = int(y.item())
        labels.append(int(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=47)
    train_sub = torch.utils.data.Subset(train_set, train_idx.tolist())
    val_sub   = torch.utils.data.Subset(train_set,   val_idx.tolist())

    # DataLoaders
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = torch.utils.data.DataLoader(val_sub,   batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=pin)

    # Build training wrapper/model (assumes your target_train_class API)
    model = target_train_class(
        train_loader, val_loader, dataset_name, target_model, device,
        use_DP, noise, norm, delta, arch, batch_size
    )

    # --- Training for exactly `epochs` epochs; track last accs only ---
    FILE_PATH_target = PATH + "_target.pth"
    last_train, last_val, last_overfit = 0.0, 0.0, None

    for epoch in range(epochs):
        print(f"<======================= Epoch {epoch+1} / {epochs} =======================>")
        acc_train, acc_val = model.train()  # expected to run one epoch and return (train_acc, val_acc)
        last_train, last_val = float(acc_train), float(acc_val)
        last_overfit = round(last_train - last_val, 6)
        print(f"Train acc={last_train:.4f} | Val acc={last_val:.4f} | Overfit={last_overfit:.6f}")

    # --- Save ONCE at the very end (final weights after last epoch) ---
    model.saveModel(FILE_PATH_target)
    print(f"Saved final model to {FILE_PATH_target}")

    # --- Final test evaluation using the just-saved final weights (no reload needed) ---
    model.testloader = test_loader
    print("Evaluating on test set with final weights...")
    final_test_acc = model.test()

    print("Finished training (no best-checkpoint logic).")

    # --- Save summary CSV of the last-epoch metrics + final test ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = PATH + f"_accs_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['last_train', 'last_val', 'overfitting_last', 'final_test'])
        writer.writerow([last_train, last_val, last_overfit, final_test_acc])
    print(f"Saved accuracies to {csv_file}")

    return last_overfit


def get_attack_batch_size(arch: str, dataset_name: str, attack_name: str) -> int:
    """
    Return attack_dataset_batch_size based on attack_name, arch, and dataset_name.
    Mirrors the provided if/elif logic. Defaults to 64 when attack_name != "apcmia".
    Raises a ValueError for unknown architectures or datasets when attack_name == "apcmia".
    """
    attack_name = attack_name.lower()
    arch = arch.lower()
    dataset_name = dataset_name.lower()

    if attack_name != "apcmia":
        return 64

    conf = {
        "cnn": {
            "cifar10": 256, "cifar100": 256,
            "fmnist": 64, "stl10": 64,
            "utkface": 256,
        },
        "wrn_rmia": {  # cifar10->64, cifar100->256
            "cifar100": 256,
            "cifar10": 64, "stl10": 64, "fmnist": 64, "utkface": 64,
        },
        "van_cnn": {  # cifar100->128; stl10->64; fmnist->128; utkface->64
            "cifar10": 128, "cifar100": 128,
            "stl10": 64, "fmnist": 128, "utkface": 64,
        },
        "mlp": {  # location->64; purchase->256; texas->256; adult->64
            "location": 64, "purchase": 256, "texas": 256, "adult": 64,
        },
    }

    if arch not in conf:
        raise ValueError("Wrong arch name, should be one of: cnn, wrn_rmia, van_cnn, mlp")

    try:
        return conf[arch][dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{dataset_name}' for arch '{arch}'")


def get_p(attack_name: str, arch: str, dataset_name: str, dp_tr: bool = False) -> float:
   
    config = {
        ("apcmia", "cnn", "cifar10"): 0.01,
        ("apcmia", "wrn_rmia", "stl10"): 0.01,
        ("apcmia", "van_cnn", "fmnist"): 0.01,
        ("apcmia", "van_cnn", "utkface"): 0.01,
        ("apcmia", "van_cnn", "cifar10"): 0.001,
        ("apcmia", "cnn", "utkface"): 0.01,
        ("apcmia", "wrn_rmia", "utkface"): 0.09,
        ("apcmia", "mlp", "adult"): 0.09,     
        ("apcmia", "mlp", "purchase"): 0.01,  
        ("apcmia", "mlp", "texas"): 0.03,     
    }

    # Special-case: location + dp_tr
    if attack_name == "apcmia" and arch == "mlp" and dataset_name == "location" and dp_tr:
        return 0.0

    return config.get((attack_name, arch, dataset_name), 0.0)


def test_meminf(PATH, device, num_classes, target_train, target_test, batch_size,  target_model, shadow_model, mode, dataset_name, attack_name, entropy_dis_dr, apcmia_cluster, arch, use_DP, acc_gap, entropy_from_checkpoint=False, entropy_checkpoint_path=""):
    
    attack_dataset_batch_size = get_attack_batch_size(arch, dataset_name, attack_name)

    # if attack_name == "lira" or attack_name == "memia" or attack_name == "seqmia" or attack_name == "nsh" or attack_name == "apcmia" or attack_name == "mia" or attack_name == "m_lira":

    attack_trainloader, attack_testloader = get_attack_dataset(target_train, target_test, attack_dataset_batch_size)

    attack_model = CombinedShadowAttackModel_NEW(num_classes, device, mode, attack_name, hidden_dim=128, layer_dim=1, output_dim=1, batch_size=attack_dataset_batch_size)
    
    perturb_model = PerturbationModel(num_classes, device, hidden_dim=128, layer_dim=1, output_dim=1, batch_size=attack_dataset_batch_size)
    
    
    attack_mode0_com(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, 
                        target_model, shadow_model, attack_model, perturb_model, num_classes, mode, dataset_name, 
                        attack_name, entropy_dis_dr, apcmia_cluster, arch, attack_dataset_batch_size, acc_gap, get_p(attack_name, arch, dataset_name, use_DP),
                        entropy_from_checkpoint, entropy_checkpoint_path)

    # else:
    #     raise Exception("Wrong attack name")


def str_to_bool(string):
    if isinstance(string, bool):
       return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def load_fpr_tpr_for_all_attacks(dataset_name, directory="."):
    """
    Loads CSV files of the form:
        dataset_name_FPR_TPR_{attack_name}_.csv
    for the given dataset_name, where attack_name is in:
        ["apcmia", "mia", "seqmia", "memia", "nsh", "lira"].
    Returns a dict: { attack_name: (fpr_list, tpr_list) }.
    """
    attack_names = ["apcmia", "memia", "m_lira", "seqmia", "mia", "nsh"]
    # attack_names = ["apcmia","m_lira",]

    fpr_tpr_dict = {}

    for attack in attack_names:
        filename = f"{dataset_name}_FPR_TPR_{attack}_.csv"
        filepath = os.path.join(directory, filename)
        
        print(filepath)
        if os.path.isfile(filepath):
            try:
                df = pd.read_csv(filepath)
                fpr_list = df["FPR"].tolist()
                tpr_list = df["TPR"].tolist()
                fpr_tpr_dict[attack] = (fpr_list, tpr_list)
                print(f"Loaded {attack} from {filepath}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                fpr_tpr_dict[attack] = ([], [])
        else:
            print(f"File not found for attack '{attack}': {filepath}")
            fpr_tpr_dict[attack] = ([], [])
    # exit()
    return fpr_tpr_dict



def _prep_roc_for_interp(fpr_array: np.ndarray, tpr_array: np.ndarray):
    """
    Sort by FPR, collapse duplicate FPRs by taking the max TPR at that FPR,
    and enforce monotone non-decreasing TPR (cummax). Returns arrays ready for
    interpolation and AUC.
    """
    # Stable sort by FPR
    order = np.argsort(fpr_array, kind="mergesort")
    fpr_s = fpr_array[order].astype(float)
    tpr_s = tpr_array[order].astype(float)

    # Collapse duplicates: for each unique FPR take max TPR (vertical segments have zero width anyway)
    uniq_fprs = []
    max_tprs  = []
    i, n = 0, len(fpr_s)
    while i < n:
        f = fpr_s[i]
        j = i
        m = tpr_s[i]
        while j + 1 < n and fpr_s[j + 1] == f:
            j += 1
            if tpr_s[j] > m:
                m = tpr_s[j]
        uniq_fprs.append(f)
        max_tprs.append(m)
        i = j + 1

    uniq_fprs = np.array(uniq_fprs, dtype=float)
    max_tprs  = np.array(max_tprs,  dtype=float)

    # Enforce monotonicity (ROC should be non-decreasing in TPR)
    tpr_monotone = np.maximum.accumulate(max_tprs)
    return uniq_fprs, tpr_monotone

def _tpr_at_fpr_interp(fpr_array: np.ndarray, tpr_array: np.ndarray, fpr_target: float) -> float:
    """
    Linear interpolation of TPR at an arbitrary FPR target.
    If the target is below min(FPR) or above max(FPR), returns the boundary TPR.
    """
    fprs, tprs = _prep_roc_for_interp(fpr_array, tpr_array)
    return float(np.interp(fpr_target, fprs, tprs, left=tprs[0], right=tprs[-1]))



import os, sys, math, shutil

def _pretty_print_metric_report(
    dataset_name,
    attack_name,
    arch,
    acc,
    roc_auc,
    tpr_dict,
    path_all=None,
    master_csv=None,
    key_fprs=(0.001, 0.01),
    use_color=True,
    box_title="ROC METRICS SUMMARY"
):
    """
    Pretty console report for ROC metrics.
    - Unicode box, aligned columns
    - Optional ANSI colors (auto-disabled if not a TTY)
    - Mini gauge bar for TPR values
    """
    # ---------- helpers ----------
    def term_width(default=88):
        try:
            return max(70, min(shutil.get_terminal_size().columns, 120))
        except Exception:
            return default

    # ANSI color helpers (opt-in)
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    color_en = bool(use_color and is_tty)

    def c(txt, code):
        return f"\033[{code}m{txt}\033[0m" if color_en else txt

    BOLD   = "1"
    DIM    = "2"
    GREEN  = "32"
    CYAN   = "36"
    MAGENTA= "35"
    YELLOW = "33"
    BLUE   = "34"

    # Unicode/ASCII fallback for borders (Unicode works well in VS Code)
    TL, TR, BL, BR = "┌", "┐", "└", "┘"
    H, V, T, B     = "─", "│", "├", "┤"

    # Formatters
    def fmt_fpr(x):
        if x == 0: return "0"
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s

    def fmt_num(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "—"

    # Gauge bar for TPR (length scaled by terminal width)
    total_width = term_width()
    label_cols = 22  # space for "FPR" & "TPR" columns
    gauge_len = max(10, min(28, total_width - label_cols - 10))

    blocks = "▏▎▍▌▋▊▉█"
    def gauge(v):
        v = 0.0 if v is None else max(0.0, min(1.0, float(v)))
        full = int(v * gauge_len)
        frac = v * gauge_len - full
        frac_idx = min(len(blocks) - 1, max(0, int(frac * len(blocks))))
        bar = "█" * full + (blocks[frac_idx] if full < gauge_len else "")
        bar = bar.ljust(gauge_len, " ")
        return bar

    # Sort FPRs ascending
    items = sorted(tpr_dict.items(), key=lambda x: x[0])

    # Build body lines
    header = f"{TL}{H*2} {c(box_title, BOLD)} {H*(total_width - 4 - 2 - len(box_title))}{TR}"
    sep    = f"{T}{H*(total_width-2)}{B}"
    footer = f"{BL}{H*(total_width-2)}{BR}"

    lines = [header]
    # Title block
    meta_l = f"Dataset : {dataset_name}"
    meta_m = f"Attack  : {attack_name}"
    meta_r = f"Arch : {arch}"
    left_pad = 2
    def row3(a,b,c):
        # distribute columns fairly
        left  = a
        mid   = b
        right = c
        # spacing
        space_mid = max(2, (total_width - 2) - len(left) - len(right) - 6)
        return f"{V}  {left}{' ' * space_mid}{right}  {V}"

    lines.append(row3(c(meta_l, CYAN), "", c(meta_r, MAGENTA)))
    lines.append(f"{V}  {c(meta_m, CYAN)}{' ' * (total_width - 6 - len(meta_m))}{V}")
    lines.append(sep)

    # Metrics
    auc_line = f"AUC: {fmt_num(roc_auc)}"
    acc_line = f"Balanced Acc (proxy): {fmt_num(acc)}"
    auc_line = c(auc_line, GREEN)
    acc_line = c(acc_line, GREEN)
    lines.append(f"{V}  {auc_line}{' ' * (total_width - 6 - len(auc_line))}{V}")
    lines.append(f"{V}  {acc_line}{' ' * (total_width - 6 - len(acc_line))}{V}")
    lines.append(sep)

    # TPR table header
    hdr = f"{'FPR':<10} {'TPR':>7}   {'Gauge':<{gauge_len}}"
    lines.append(f"{V}  {c('TPR @ FPR (Interpolated)', BOLD)}{' ' * (total_width - 6 - len('TPR @ FPR (Interpolated)'))}{V}")
    lines.append(f"{V}  {hdr}{' ' * (total_width - 6 - len(hdr))}{V}")
    lines.append(f"{V}  {'-'*10} {'-'*7}   {'-'*gauge_len}{' ' * (total_width - 6 - 10 - 7 - 3 - gauge_len)}{V}")

    # Rows
    key_fprs = set(float(k) for k in key_fprs)  # normalize
    for fpr, tpr in items:
        fpr_txt = fmt_fpr(fpr)
        tpr_txt = fmt_num(tpr)
        row = f"{fpr_txt:<10} {tpr_txt:>7}   {gauge(tpr)}"
        is_key = any(abs(fpr - k) < 1e-12 for k in key_fprs)
        if is_key:
            row = c(row, YELLOW)  # highlight key FPRs
        lines.append(f"{V}  {row}{' ' * (total_width - 6 - len(row))}{V}")

    # Saved paths (dim)
    any_paths = path_all or master_csv
    if any_paths:
        lines.append(sep)
        lines.append(f"{V}  {c('Saved files', BOLD)}{' ' * (total_width - 6 - len('Saved files'))}{V}")
        if path_all:
            p = os.path.relpath(path_all)
            s = f"• TPR@FPR CSV : {p}"
            lines.append(f"{V}  {c(s, DIM)}{' ' * (total_width - 6 - len(s))}{V}")
        if master_csv:
            p = os.path.relpath(master_csv)
            s = f"• Master CSV  : {p}"
            lines.append(f"{V}  {c(s, DIM)}{' ' * (total_width - 6 - len(s))}{V}")

    lines.append(footer)

    # Print
    print("\n" + "\n".join(lines) + "\n")


def metric_results_new(fpr_list, tpr_list, attack_name, dataset_name, arch, directory="./tprs_at/"):
    """
    Same signature as before, but TPR@FPR values are computed by interpolation.
    Uses cleaned (sorted, deduped, monotone) ROC for ALL downstream metrics.
    """
    # Paths
    directory = os.path.join(directory, arch, dataset_name)
    os.makedirs(directory, exist_ok=True)

    fpr_array = np.array(fpr_list, dtype=float)
    tpr_array = np.array(tpr_list, dtype=float)

    # Clean once; use everywhere.
    fprs_u, tprs_u = _prep_roc_for_interp(fpr_array, tpr_array)

    # Max balanced accuracy along the ROC
    acc = np.max((tprs_u + (1.0 - fprs_u)) / 2.0)

    # AUC on cleaned curve (assumes fprs_u ascending)
    roc_auc = auc(fprs_u, tprs_u)

    # Interpolated TPRs at key FPRs (use cleaned arrays)
    fprs_of_interest = [0.01, 0.001, 0.0001, 0.00001]
    # Clamp to domain inside the helper (recommended) or guard here:
    # f_min, f_max = float(fprs_u.min()), float(fprs_u.max())
    # thr = min(max(thr, f_min), f_max)
    tpr_dict = {
        thr: _tpr_at_fpr_interp(fprs_u, tprs_u, thr)
        for thr in fprs_of_interest
    }
    tpr_dict = {k: round(float(v), 3) for k, v in tpr_dict.items()}

    print(f"Dataset: {dataset_name}, Attack: {attack_name}, Arch: {arch}")
    print(f"Balanced acc (proxy): {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    if 0.001 in tpr_dict:
        print(f"TPR@0.001 FPR (interp): {tpr_dict[0.001]}")

    # Save per-dataset CSV of TPR@FPRs (ascending FPR)
    df_all = (
        pd.DataFrame({"FPR": list(tpr_dict.keys()), "TPR": list(tpr_dict.values())})
        .sort_values("FPR", ascending=True)
        .reset_index(drop=True)
    )
    path_all = os.path.join(directory, f"{dataset_name}_tprAT_{attack_name}.csv")
    df_all.to_csv(path_all, index=False)
    print(f"TPR@FPR values saved to {path_all}")

    # ---- Master table update ----
    # Extend schema: add Location (and keep your existing ones)
    master_cols = [
        "Method",
        "CIFAR-10_tpr001", "STL-10_tpr001", "CIFAR-100_tpr001", "UTKFace_tpr001", "FMNIST_tpr001", "Location_tpr001",
        "CIFAR-10_acc",    "STL-10_acc",    "CIFAR-100_acc",    "UTKFace_acc",    "FMNIST_acc",    "Location_acc",
        "CIFAR-10_auc",    "STL-10_auc",    "CIFAR-100_auc",    "UTKFace_auc",    "FMNIST_auc",    "Location_auc",
    ]

    attack_map = {
        "apcmia": "apcMIA",
        "lira":   "LiRA",   # pick one spelling; using LiRA here
        "memia":  "meMIA",
        "seqmia": "seqMIA",
        "nsh":    "NSH",
        "mia":    "MIA",
        "m_lira": "m-LiRA",
    }
    dataset_map = {
        "cifar10":  ("CIFAR-10_tpr001", "CIFAR-10_acc", "CIFAR-10_auc"),
        "stl10":    ("STL-10_tpr001",   "STL-10_acc",   "STL-10_auc"),
        "cifar100": ("CIFAR-100_tpr001","CIFAR-100_acc","CIFAR-100_auc"),
        "utkface":  ("UTKFace_tpr001",  "UTKFace_acc",  "UTKFace_auc"),
        "fmnist":   ("FMNIST_tpr001",   "FMNIST_acc",   "FMNIST_auc"),
        "location": ("Location_tpr001", "Location_acc", "Location_auc"),
        "adult":    ("adult_tpr001", "adult_acc", "adult_auc"),  # alias
        "texas":    ("texas_tpr001", "texas_acc", "texas_auc"),  # alias
        "purchase": ("purchase_tpr001", "purchase_acc", "purchase_auc"),  # alias

    }

    method_row = attack_map.get(str(attack_name).lower())
    dataset_key = str(dataset_name).lower()

    if method_row is None or dataset_key not in dataset_map:
        print(f"Warning: Unrecognized attack '{attack_name}' or dataset '{dataset_name}'.")
        return roc_auc, acc, tpr_dict

    col_tpr, col_acc, col_auc = dataset_map[dataset_key]
    master_csv = os.path.join(os.path.dirname(directory), f"master_all_{arch}.csv")
    os.makedirs(os.path.dirname(master_csv), exist_ok=True)

    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
        # Harden schema: add any missing columns
        for col in master_cols:
            if col not in df_master.columns:
                df_master[col] = None
        # Reorder columns to canonical order
        df_master = df_master[[c for c in master_cols if c in df_master.columns]]
    else:
        df_master = pd.DataFrame(columns=master_cols)

    # Upsert row for method
    if "Method" in df_master.columns and (df_master["Method"] == method_row).any():
        mask = df_master["Method"] == method_row
        df_master.loc[mask, col_tpr] = tpr_dict.get(0.001, None)
        df_master.loc[mask, col_acc] = round(float(acc), 3)
        df_master.loc[mask, col_auc] = round(float(roc_auc), 3)
    else:
        new_row = {col: None for col in master_cols}
        new_row["Method"] = method_row
        new_row[col_tpr] = tpr_dict.get(0.001, None)
        new_row[col_acc] = round(float(acc), 3)
        new_row[col_auc] = round(float(roc_auc), 3)
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)

    df_master.to_csv(master_csv, index=False)
    print(f"Master CSV updated at: {master_csv}")

    _pretty_print_metric_report(
    dataset_name=dataset_name,
    attack_name=attack_name,
    arch=arch,
    acc=acc,
    roc_auc=roc_auc,
    tpr_dict=tpr_dict,
    path_all=path_all,
    master_csv=master_csv,
    key_fprs=(0.001, 0.01),   # rows will be highlighted
    use_color=True            # set False if you prefer no ANSI colors
)


    return roc_auc, acc, tpr_dict


from viz_metrics import (
    _parse_master,
    plot_metric_dotplot,
    plot_metric_ranking,
    plot_roc_curves,
    plot_tpr_marker_at,
)

def make_metric_figs(arch: str, figs_dir="./figs"):
    os.makedirs(figs_dir, exist_ok=True)
    master_csv = f"./tprs_at/{arch}/master_all_{arch}.csv"
    longdf = _parse_master(master_csv)

    # Dot plots per metric
    plot_metric_dotplot(longdf, "tpr001", f"{figs_dir}/{arch}_dot_tpr001", arch=arch)
    plot_metric_dotplot(longdf, "auc",    f"{figs_dir}/{arch}_dot_auc",    arch=arch)
    plot_metric_dotplot(longdf, "acc",    f"{figs_dir}/{arch}_dot_acc",    arch=arch)

    # Method rankings (mean across datasets)
    plot_metric_ranking(longdf, "tpr001", f"{figs_dir}/{arch}_rank_tpr001", arch=arch)
    plot_metric_ranking(longdf, "auc",    f"{figs_dir}/{arch}_rank_auc",    arch=arch)
    plot_metric_ranking(longdf, "acc",    f"{figs_dir}/{arch}_rank_acc",    arch=arch)


def metric_results_mlp(fpr_list, tpr_list, attack_name, dataset_name, arch, directory="./tprs_at/"):
    """
    Calculates ROC AUC, accuracy (computed from ROC data), and TPR values at specified FPR thresholds 
    from ROC data. Then updates a master CSV that accumulates three metrics for each (attack, dataset) combination:
      - TPR@0.001
      - Accuracy (acc)
      - ROC AUC
      
    The master CSV is structured with one row per attack method and three groups of columns (one group per metric) 
    for each dataset. In our case, we have four datasets: Location, Adult, Texas-100, and Purchase-100.
    
    Parameters:
      fpr_list: List of false positive rates.
      tpr_list: List of true positive rates.
      attack_name: Name of the attack (e.g., "apcMIA", "LiRa", etc.).
      dataset_name: Name of the dataset (e.g., "Location", "Adult", "Texas-100", "Purchase-100"). Case-insensitive.
      arch: Architecture (e.g., "vgg16", "cnn", etc.).
      directory: Base directory for storing CSV files.
    
    Returns:
      roc_auc: ROC AUC computed from the fpr_list and tpr_list.
      acc: Accuracy computed from the ROC data.
      tpr_dict: Dictionary mapping FPR thresholds to TPR values.
    """
    # Build directory: e.g., ./tprs_at/<arch>/<dataset_name>/
    directory = os.path.join(directory, arch, dataset_name)
    os.makedirs(directory, exist_ok=True)

    # Convert lists to numpy arrays.
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Compute accuracy from ROC data as: acc = max(1 - (FPR + (1 - TPR))/2)
    acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
    
    # Compute ROC AUC.
    roc_auc = auc(fpr_array, tpr_array)
    
    # Define FPR thresholds of interest.
    fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
    tpr_dict = {}
    for threshold in fprs:
        indices = np.where(fpr_array <= threshold)[0]
        if len(indices) > 0:
            tpr_val = tpr_array[indices[-1]]
        else:
            tpr_val = None
        tpr_dict[threshold] = tpr_val
    
    # Round TPR values to 3 decimals.
    tpr_dict = {k: round(v, 3) if v is not None else None for k, v in tpr_dict.items()}
    
    print(f"Dataset: {dataset_name}, Attack: {attack_name}, Arch: {arch}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")
    
    # Save CSV with all (FPR, TPR) pairs for this dataset+attack.
    df_all = pd.DataFrame(list(tpr_dict.items()), columns=["FPR", "TPR"])
    df_all.sort_values(by="FPR", ascending=False, inplace=True)
    file_name_all = f"{dataset_name}_tprAT_{attack_name}.csv"
    path_all = os.path.join(directory, file_name_all)
    df_all.to_csv(path_all, index=False)
    print(f"TPR values saved to {path_all}")

    # --- Update the master CSV ---
    # Define master CSV columns for four datasets:
    master_cols = ["Method",
                   "Location_tpr001", "Adult_tpr001", "Texas-100_tpr001", "Purchase-100_tpr001",
                   "Location_acc",   "Adult_acc",   "Texas-100_acc",   "Purchase-100_acc",
                   "Location_auc",   "Adult_auc",   "Texas-100_auc",   "Purchase-100_auc"]
    
    # Mapping from attack names (lower case) to display names for the "Method" column.
    attack_map = {
        "apcmia": "apcMIA",
        "m_lira":   "LiRA",
        "memia":  "meMIA",
        "seqmia": "seqMIA",
        "nsh":    "NSH",
        "mia":    "MIA",
    }
    
    # Mapping from dataset_name (lower case) to master CSV columns (tpr001, acc, auc).
    dataset_map = {
        "location":    ("Location_tpr001",    "Location_acc",    "Location_auc"),
        "adult":       ("Adult_tpr001",       "Adult_acc",       "Adult_auc"),
        "texas":   ("Texas-100_tpr001",   "Texas-100_acc",   "Texas-100_auc"),
        "purchase":("Purchase-100_tpr001","Purchase-100_acc","Purchase-100_auc")
    }
    
    method_row = attack_map.get(attack_name.lower())
    dataset_key = dataset_name.lower()
    if method_row is None or dataset_key not in dataset_map:
        print(f"Warning: Unrecognized attack '{attack_name}' or dataset '{dataset_name}'.")
        return roc_auc, acc, tpr_dict
    col_tpr, col_acc, col_auc = dataset_map[dataset_key]
    
    # Define master CSV path: stored in the parent directory of the current dataset folder (per architecture).
    master_csv = os.path.join(os.path.dirname(directory), f"master_all_{arch}.csv")
    
    # Load master CSV if it exists; otherwise, create a new DataFrame.
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
    else:
        df_master = pd.DataFrame(columns=master_cols)
    
    # Check if a row for the current method already exists.
    if method_row in df_master["Method"].values:
        # Update the corresponding columns for the given dataset.
        df_master.loc[df_master["Method"] == method_row, col_tpr] = round(tpr_dict[0.001], 3)
        df_master.loc[df_master["Method"] == method_row, col_acc] = round(acc, 3)
        df_master.loc[df_master["Method"] == method_row, col_auc] = round(roc_auc, 3)
    else:
        # Create a new row with None in all master columns, then set the current method's values.
        new_row = {col: None for col in master_cols}
        new_row["Method"] = method_row
        new_row[col_tpr] = round(tpr_dict[0.001], 3)
        new_row[col_acc] = round(acc, 3)
        new_row[col_auc] = round(roc_auc, 3)
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated master CSV.
    df_master.to_csv(master_csv, index=False)
    print(f"Master CSV updated at: {master_csv}")
    
    return roc_auc, acc, tpr_dict


def metric_results(fpr_list, tpr_list, attack_name, dataset_name, arch, directory="./tprs_at/"):
    """
    Calculates ROC AUC, accuracy, and TPR values at specified FPR thresholds from ROC data.
    Then it saves two CSV files:
      1) A CSV containing all threshold/TPR pairs for this (dataset, attack) combination.
      2) A master CSV (all_tpr_0.001.csv) that accumulates the TPR@0.001 for each dataset,
         with one row per dataset and one column per attack method.

    Parameters:
      fpr_list: List of false positive rates.
      tpr_list: List of true positive rates.
      attack_name: Name of the attack (e.g., "apcMIA", "LiRa", etc.).
      dataset_name: Name of the dataset (e.g., "FMNIST", "UTKFace", etc.).
      arch: Architecture (e.g., "vgg16", "cnn", etc.).
      directory: Base directory in which to store the CSV files (default "./tprs_at/").
    
    Returns:
      roc_auc: ROC AUC value.
      acc: Accuracy computed from the ROC curve data.
      tpr_dict: Dictionary mapping each specified FPR threshold to its TPR.
    """
    # Build directory: ./tprs_at/<arch>/<dataset_name>/
    directory = os.path.join(directory, arch, dataset_name)
    os.makedirs(directory, exist_ok=True)

    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Compute accuracy and ROC AUC from ROC curve data.
    acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
    roc_auc = auc(fpr_array, tpr_array)

    # Define FPR thresholds of interest.
    fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
    tpr_dict = {}
    for threshold in fprs:
        indices = np.where(fpr_array <= threshold)[0]
        if len(indices) > 0:
            tpr_val = tpr_array[indices[-1]]
        else:
            tpr_val = None
        tpr_dict[threshold] = tpr_val

    # Round TPR values to 3 decimals.
    tpr_dict = {k: round(v, 3) if v is not None else None for k, v in tpr_dict.items()}

    print(f"Dataset: {dataset_name}, Attack: {attack_name}, Arch: {arch}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")
    print("All TPR values:")
    for threshold, value in tpr_dict.items():
        print(f"  FPR {threshold}: TPR = {value}")

    # -------------------------------------------------------------------
    # 1) Save a CSV with all threshold/TPR pairs for this dataset+attack.
    # -------------------------------------------------------------------
    df_all = pd.DataFrame(list(tpr_dict.items()), columns=["FPR", "TPR"])
    df_all.sort_values(by="FPR", ascending=False, inplace=True)
    file_name_all = f"{dataset_name}_tprAT_{attack_name}.csv"
    path_all = os.path.join(directory, file_name_all)
    df_all.to_csv(path_all, index=False)
    print(f"TPR values saved to {path_all}")

    # -------------------------------------------------------------------
    # 2) Update a master CSV (one per architecture) that accumulates TPR@0.001 values.
    #    The master CSV has one row per dataset and one column for each attack.
    # -------------------------------------------------------------------
    # Master CSV is stored under the arch folder (one level up from the dataset folder)

    l_fpr = 0.001;
    master_csv = os.path.join(os.path.dirname(directory), f"all_tpr_{l_fpr}.csv")

    # Define expected columns. (You can add more columns if needed.)
    columns = ["Dataset", "apcMIA", "LiRa", "meMIA", "seqMIA", "NSH", "MIA"]

    # Mapping from attack names to CSV column names.
        # Mapping from attack names (in lower case) to CSV column names.
    attack_col_map = {
        "apcmia": "apcMIA",
        "lira":   "LiRa",
        "memia":  "meMIA",
        "seqmia": "seqMIA",
        "nsh":    "NSH",
        "mia":    "MIA",
    }

    # Get the TPR value for FPR=0.001.
    tpr_0_001 = tpr_dict[l_fpr]

    # Load master CSV if it exists; otherwise, create a new DataFrame.
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
    else:
        df_master = pd.DataFrame(columns=columns)

    # Check if a row for this dataset already exists.
    if dataset_name in df_master["Dataset"].values:
        # Update the row for this dataset.
        col_name = attack_col_map.get(attack_name.lower())
        if col_name is not None:
            df_master.loc[df_master["Dataset"] == dataset_name, col_name] = tpr_0_001
    else:
        # Create a new row.
        new_row = {col: None for col in columns}
        new_row["Dataset"] = dataset_name
        col_name = attack_col_map.get(attack_name.lower())
        if col_name is not None:
            new_row[col_name] = tpr_0_001
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)

    # Save the master CSV.
    df_master.to_csv(master_csv, index=False)
    print(f"Master CSV updated at: {master_csv}")

    return roc_auc, acc, tpr_dict

# plot_roc_curves_for_attacks(fpr_tpr_data, dataset_name, threshold_curves_pth, arch)
def plot_roc_curves_for_attacks(fpr_tpr_dict, dataset_name, save_path, arch):
    """
    Plots ROC curves for each attack on the same figure.
    `
    Parameters:
      - fpr_tpr_dict: dict of {attack_name: (fpr_list, tpr_list)}
      - dataset_name: Name of the dataset (string)
      - save_path: If provided, saves the figure to the given path (e.g., "roc_plot.pdf")
    """


    size = 30
    params = {
        'axes.labelsize': size,
        'font.size': size,
        'legend.fontsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'figure.figsize': [10, 9],
        "font.family": "arial",
    }


    plt.rcParams.update(params)

    # Define a color palette for consistency
    attack_colors = {
        "apcmia": "#0d0478",   # Blue
        "mia":    "#9467bd",   # Red
        "seqmia": "#2ca02c",   # Green
        "memia":  "#d62728",   # Purple
        "nsh":    "#ff7f0e",   # Orange
        "lira":   "#8c564b"    # Brownish
    }

   
    # Define line styles for each attack method
    attack_linestyles = {
        "apcmia": "-",
        "mia":    "--",
        "seqmia": "-.",
        "memia":  ":",
        "nsh":    "-",
        "lira":   "--"
    }

    # Assume fpr_tpr_dict, dataset_name, arch, and save_path are defined.
    # fpr_tpr_dict: a dictionary mapping attack names to (fpr_list, tpr_list)

    for attack_name, (fpr, tpr) in fpr_tpr_dict.items():
        if not fpr or not tpr:
            continue
        # Normalize attack names to desired display names:
        if attack_name.upper() == "M_LIRA":
            display_name = "LiRA"
        elif attack_name.upper() == "MEMIA":
            display_name = "meMIA"
        elif attack_name.upper() == "SEQMIA":
            display_name = "seqMIA"
        elif attack_name.upper() == "APCMIA":
            display_name = "apcMIA"
        elif attack_name.upper() == "NSH":
            display_name = "NSH"
        elif attack_name.upper() == "MIA":
            display_name = "MIA"
        else:
            display_name = attack_name

        # Retrieve marker and line style; default if not found.
        # marker = attack_markers.get(display_name.lower(), "o")
        linestyle = attack_linestyles.get(display_name.lower(), "-")
        color = attack_colors.get(display_name.lower(), "black")
        
        
        
        # Plot the ROC curve using specified style
        plt.plot(fpr, tpr,
                label=rf"{display_name}",
                color=color, lw=3.5, linestyle=linestyle, markersize=6)
        
        # If needed, update metrics in master CSV (function call remains the same)
        metric_results_new(fpr, tpr, display_name, dataset_name, arch)

    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], ls="--", color="gray", lw=2)

    # Set labels and other figure properties
    plt.xlabel("False Positive Rate", fontsize=size)
    plt.ylabel("True Positive Rate", fontsize=size)
    legend = plt.legend(loc="lower right")
    frame = legend.get_frame()
    frame.set_edgecolor('0.91')
    plt.subplots_adjust(left=0.60)
    plt.tight_layout()

    # Set axes to log scale
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1.01)

    # Save the figure as a PDF with 300 dpi
    filename = f"{dataset_name}_roc_curves_{arch}.pdf"
    filepath = os.path.join(save_path, filename)
    print(f"File path for roc curve: {filepath}")
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    # plt.show()


def plot_roc_curves_for_attacks(
    fpr_tpr_dict,
    dataset_name,
    save_path,
    arch,
    *,
    update_master=True,
    fig_size=(10, 9),
    xmin=1e-5,
    ymin=1e-5,
):
    # Typography & disable grid
    size = 30
    plt.rcParams.update({
        "axes.labelsize": size,
        "font.size": size,
        "legend.fontsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "figure.figsize": list(fig_size),
        "font.family": "arial",
        "axes.linewidth": 1.4,
        "savefig.dpi": 300,
        "axes.grid": False,
    })

    fig, ax = plt.subplots()
    ax.grid(False, which="both")

    # Colors (keep distinct)
    attack_colors = {
        "apcmia": "#0137D8",
        "memia":  "#CC2411",
        "lira":   "#00998C",
        "seqmia": "#EC6415",
        "mia":    "#0FB7F4",
        "nsh":    "#AB2B74",
    }

    def canon(n: str) -> str:
        n = n.lower()
        return "lira" if n in {"m_lira", "m-lira"} else n

    def display(n: str) -> str:
        m = n.lower()
        return {"apcmia":"apcMIA","memia":"meMIA","seqmia":"seqMIA","lira":"LiRA","mia":"MIA","nsh":"NSH"}.get(m, n.upper())

    # Same solid line for everyone; add subtle white outline
    line_outline = [pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()]

    for attack_name, (fpr, tpr) in fpr_tpr_dict.items():
        if not fpr or not tpr:
            continue

        key = canon(attack_name)
        color = attack_colors.get(key, "#444444")

        fpr = np.asarray(fpr, float)
        tpr = np.asarray(tpr, float)

        line, = ax.plot(
            fpr, tpr,
            label=display(key),
            color=color,
            lw=3.5,
            linestyle="-",           # same style for all
            zorder=3,
            solid_capstyle="butt",   # <-- prevents overhang at the frame
            dash_capstyle="butt",
            solid_joinstyle="round",
        )
        line.set_path_effects(line_outline)

        if update_master:
            metric_results_new(fpr.tolist(), tpr.tolist(), display(key), dataset_name, arch)

    # Diagonal baseline (also with butt caps)
    base, = ax.plot([xmin, 1], [xmin, 1], ls="--", color="#999999", lw=2, zorder=2,
                    solid_capstyle="butt", dash_capstyle="butt")

    # Log–log axes and limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, 1.0)
    ax.set_ylim(ymin, 1.01)

    # Put spines ABOVE the data so borders stay visible
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_zorder(10)   # <-- key: draw frame over the lines

    # Ticks only (no grid)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=tuple(range(2, 10)), numticks=100))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=tuple(range(2, 10)), numticks=100))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    leg = ax.legend(loc="lower right")
    leg.get_frame().set_edgecolor('0.91')
    leg.get_frame().set_linewidth(1.0)

    fig.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    base = f"{dataset_name}_roc_curves_{arch}"
    pdf_path = os.path.join(save_path, base + ".pdf")
    svg_path = os.path.join(save_path, base + ".svg")
    png_path = os.path.join(save_path, base + ".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    print(f"Figure saved to:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


def load_plot_thresholds_sub(base_dir, threshold_save_path):
    """
    Recursively finds all CSV files ending with 
    '_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv' under base_dir.
    
    For each file:
      - Infers the architecture from the directory structure,
      - Loads the CSV and slices data up to (and including) the row with minimal test_loss,
      - Smooths the curves (using a rolling average with window=3),
      - Plots the cosine threshold (\tau_c) and entropy threshold (\tau_e) in the top subplot,
        and the test loss in the bottom subplot (with a shared x-axis),
      - Saves the resulting plot in a subfolder (named by architecture) under threshold_save_path.
    """

    print(f"Base directory PLOTTING: {base_dir}")

    os.makedirs(threshold_save_path, exist_ok=True)

    # Plot settings
    size = 20
    params = {
       'axes.labelsize': size,
       'font.size': size,
       'legend.fontsize': size,
       'xtick.labelsize': size,
       'ytick.labelsize': size,
       'figure.figsize': [10, 8],
       "font.family": "arial",
    }
    plt.rcParams.update(params)

    # Walk recursively through base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"):
                filepath = os.path.join(root, file)

                # Determine the architecture from the relative path
                # Expected structure: ./demoloader/trained_model/<arch>/<dataset>/...
                rel_path = os.path.relpath(root, base_dir)
                arch = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path

                # Create a subfolder for the architecture
                arch_save_path = os.path.join(threshold_save_path, arch)
                os.makedirs(arch_save_path, exist_ok=True)

                # Infer dataset name by removing the known suffix from the filename.
                suffix = "_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"
                dataset_name = file.replace(suffix, "")

                print(f"Loading: {filepath}")
                df = pd.read_csv(filepath)

                # Ensure the required columns exist
                required_cols = {"epoch", "cosine_threshold", "entropy_threshold", "test_loss"}
                if not required_cols.issubset(df.columns):
                    print(f"Skipping {file} -- missing one of {required_cols}.")
                    continue

                # Slice the dataframe up to (and including) the row with minimal test_loss
                min_idx = df["test_loss"].idxmin()
                df = df.iloc[:min_idx + 1]

                # Smooth curves with a rolling average (window=3)
                epochs     = df["epoch"]
                cos_thresh = df["cosine_threshold"].rolling(window=3, min_periods=1).mean()
                ent_thresh = df["entropy_threshold"].rolling(window=3, min_periods=1).mean()
                test_loss  = df["test_loss"].rolling(window=3, min_periods=1).mean()

                # Create subplots: two rows sharing the same x-axis
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                # fig.suptitle(f"{dataset_name} Thresholds and Test Loss", fontsize=size+2)

                # Top subplot: plot thresholds
                ax1.plot(epochs, cos_thresh, label=r"$\tau_c$", marker="o", color="tab:blue")
                ax1.plot(epochs, ent_thresh, label=r"$\tau_e$", marker="s", color="tab:orange")
                ax1.set_ylabel("Threshold Value")
                ax1.legend(loc="best")
                ax1.grid(linestyle="dotted")

                # Bottom subplot: plot test loss
                ax2.plot(epochs, test_loss, label="Test Loss", marker="^", color="tab:green")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Attack Loss")
                ax2.legend(loc="best")
                ax2.grid(linestyle="dotted")

                # Adjust layout to ensure no clipping and reserve space for the suptitle
                plt.tight_layout(rect=[0, 0, 1, 0.95])

                # Save the figure
                out_name = f"{dataset_name}_thresholds_and_loss_min_subplot.pdf"
                out_path = os.path.join(arch_save_path, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Saved plot for '{dataset_name}' in arch '{arch}' -> {out_path}")

def get_train_params(dataset_name: str, arch: str) -> tuple[int, int]:
    """
    Return (batch_size, patience) based on dataset_name and arch.
    Mirrors the provided if/elif mapping exactly.

    Examples
    --------
    >>> get_train_hyperparams("cifar10", "wrn_rmia")
    (256, 20)
    >>> get_train_hyperparams("purchase", "mlp")
    (64, 5)
    """
    ds = (dataset_name or "").lower()
    ar = (arch or "").lower()

    if ds == "purchase":
        batch_size = 64
        patience = 5

    elif ds == "location":
        batch_size = 128
        patience = 5

    elif ds == "texas":
        batch_size = 128
        patience = 5

    elif ds == "adult":
        batch_size = 128
        patience = 5

    elif ds == "cifar10":
        if ar == "wrn_rmia":
            batch_size = 256
        else:
            batch_size = 100
        patience = 20

    elif ds == "cifar100":
        if ar == "wrn_rmia":
            batch_size = 256
        else:
            batch_size = 100
        patience = 20

    elif ds == "fmnist":
        if ar == "wrn_rmia":
            batch_size = 64
        else:
            batch_size = 64
        patience = 10

    elif ds == "stl10":
        batch_size = 256
        patience = 20

    elif ds == "utkface":
        if ar == "wrn_rmia":
            batch_size = 64
        else:
            batch_size = 64
        patience = 20

    elif ds == "imagenet":
        if ar == "resnet50":
            batch_size = 256
        else:
            batch_size = 64
        patience = 20

    else:
        batch_size = 64
        patience = 10

    return batch_size, patience

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="location")
    # parser.add_argument('-at', '--attack_type', type=int, default=0)
    parser.add_argument('-tm', '--train_model', action='store_true')
    parser.add_argument('-ts', '--train_shadow', action='store_true')
    parser.add_argument('-ta', '--attack', action='store_true')
    parser.add_argument('-trnn', '--train_rnn', action='store_true')
    parser.add_argument('-ud', '--use_DP', action='store_true')
    parser.add_argument('-ne', '--noise', type=float, default=1.3)
    parser.add_argument('-nm', '--norm', type=float, default=1.5)
    parser.add_argument('-d', '--delta', type=float, default=1e-5)
    parser.add_argument('-m', '--mode', type=int, default=0)
    parser.add_argument('-gg', '--acc_gap', type=float, default=0.009418)
    parser.add_argument('-dsize', '--DSize', type=int, default=30000)  
    parser.add_argument('-an', '--attack_name', type=str, default="apcmia")
    parser.add_argument('-plt', '--plot', action='store_true')
    parser.add_argument('-roc', '--plot_results', type=str, default="roc")
    parser.add_argument('-arch', '--arch', type=str, default="cnn")
    parser.add_argument('-l_tr', '--lira_train', action='store_true')
    parser.add_argument('-l_inf', '--lira_inference', action='store_true')
    parser.add_argument('-l_roc', '--lira_roc', action='store_true')
    parser.add_argument('-n_queries', '--aug', type=int, default=2)
    parser.add_argument('-plt_cls', '--apcmia_cluster', action='store_true')
 
    parser.add_argument('--entropy_from_checkpoint', action='store_true', help='Load saved attack bundle to compute entropy plots without retraining.')
    parser.add_argument('--entropy_checkpoint', type=str, default='', help='Optional path to attack bundle; defaults to generated checkpoint.')
    args = parser.parse_args()

    print(args.DSize)
   

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    dataset_name = args.dataset_name
    dataset_name = dataset_name.lower()
    arch = args.arch
    arch = arch.lower()
    
    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")
    root = "./data"
    use_DP = args.use_DP
    noise = args.noise
    norm = args.norm
    delta = args.delta
    acc_gap = args.acc_gap
    mode = args.mode
    apcmia_cluster = args.apcmia_cluster
    
    
    entropy_from_checkpoint = args.entropy_from_checkpoint
    entropy_checkpoint = args.entropy_checkpoint
    attack_name = args.attack_name.lower()

    
  

    # First, check if the dataset requires mlp
    if dataset_name.lower() in ('location', 'texas', 'adult', 'purchase'):
        if arch.lower() != 'mlp':
            print("For datasets 'location', 'texas', 'adult', 'purchase', only the 'mlp' architecture is allowed!")
            exit()

    # Then, if arch is among the allowed architectures, set up the paths.
    if arch.lower() in ('van_cnn', 'cnn', "wrn_rmia",'mlp'):
        TARGET_ROOT = f"./demoloader/trained_model/{arch}/{dataset_name}/"
        roc_curves_pth = f"./roc_curves/{arch}/{dataset_name}/"
        entropy_dis_dr = f"./entropy_dis/{arch}/{dataset_name}/"
        threshold_curves_pth = f"./thresh_curves/{arch}/{dataset_name}/" # ./demoloader/trained_model/cnn/cifar10/adult_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv
        # roc_curves_pth = f"./thresh_curves/{arch}/{dataset_name}/"
        
    else:
        print("Incorrect architecture type! Provide one of these => ['vgg16', 'van_cnn', 'cnn', 'wrn', 'wrn_rmia','resnet50', 'mlp']")
        exit()
    
    if not os.path.exists(TARGET_ROOT):
        print(f"Create directory named {TARGET_ROOT}")
        os.makedirs(TARGET_ROOT)

    MODEL_SAVE_PATH = TARGET_ROOT + dataset_name
    print("Target_patth: ",  MODEL_SAVE_PATH)
    batch_size, patience = get_train_params(dataset_name, arch)

    epochs = 100
    
   
    if args.plot:
        if args.plot_results.lower() == "roc":
            print("roc")

            # save fpr_tpr_data to a csv file at filename = f"{dataset_name}_roc_curves_{arch}.pdf", filepath = os.path.join(roc_curves_pth, filename) with time stamp,
            

            fpr_tpr_data = load_fpr_tpr_for_all_attacks(dataset_name, directory=TARGET_ROOT)
           
            # Save fpr_tpr_data to a CSV file
            if "apcmia" in fpr_tpr_data:
                fpr, tpr = fpr_tpr_data["apcmia"]
                df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})

                fpr_array = np.array(fpr)
                tpr_array = np.array(tpr)

                # Compute accuracy and ROC AUC from ROC curve data.
                acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
                roc_auc = auc(fpr_array, tpr_array)

                fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
                tpr_dict = {}
                for threshold in fprs:
                    indices = np.where(fpr_array <= threshold)[0]
                    if len(indices) > 0:
                        tpr_val = tpr_array[indices[-1]]
                    else:
                        tpr_val = None
                    tpr_dict[threshold] = tpr_val

                print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")

                # Create the metrics DataFrame
                metrics = {
                    "AUC": [roc_auc],
                    "Accuracy": [acc],
                    "TPR @ 0.1% FPR": [tpr_dict[0.001]],
                    "TPR @ 1% FPR": [tpr_dict[0.01]],
                }
                df_metrics = pd.DataFrame(metrics)

                os.makedirs(roc_curves_pth, exist_ok=True)
                # Build the output path with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{dataset_name}_roc_curves_{arch}_{timestamp}.xlsx"
                filepath = os.path.join(roc_curves_pth, filename)

               

                # Save both DataFrames to different sheets in the same Excel file.
                with pd.ExcelWriter(filepath) as writer:
                    df_roc.to_excel(writer, sheet_name="ROC_Curves", index=False)
                    df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
            else:
                print("apcmia data not found.")
             # 2. Plot the ROC curves for all attacks in a single figure.
            print(f"ROC saved to {roc_curves_pth}")
            # exit()

            

            plot_roc_curves_for_attacks(fpr_tpr_data, dataset_name, roc_curves_pth, arch)
            
        elif args.plot_results.lower() == "th":

            print(f"attack name is {attack_name}")
            if attack_name == "apcmia":
                print("plotting thresholds for apcmia")
                base_directory = "./demoloader/trained_model"  # top-level directory to search
                output_dir     = "./threshold_plots"          # where to save the figures
                load_plot_thresholds_sub(base_directory, output_dir)
                exit()
            else:

                print(f"can't plot thresholds for {attack_name}, try apcmia")
                exit()
        else:
            print("Incorrect plot argument")
        exit()
    
 
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(dataset_name, attr, root, device, arch, args.DSize)
   

    if args.train_model:
        print("Training Target model")
       
        if ~use_DP:
            acc_gap = target_train_func(MODEL_SAVE_PATH, device, target_train, target_test, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch, epochs, patience)
        else:
            acc_gap = target_train_func_dp(MODEL_SAVE_PATH, device, target_train, target_test, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch, epochs, patience)
        
        print("acc_gap: ", acc_gap)
        # exit()

    if args.train_shadow:
        print("Training Shadow model")
        shadow_train_func(MODEL_SAVE_PATH, device, shadow_train, shadow_test, shadow_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch, epochs, patience)
        
        # exit()


    if args.attack:
        test_meminf(MODEL_SAVE_PATH, device, num_classes, target_train, target_test, batch_size,  target_model, shadow_model,  mode, dataset_name, attack_name, entropy_dis_dr, apcmia_cluster, arch, use_DP, acc_gap, entropy_from_checkpoint, entropy_checkpoint)
    else:
        sys.exit("attack has not been launched, use --attack flage to trigger it")

if __name__ == "__main__":
    main()
