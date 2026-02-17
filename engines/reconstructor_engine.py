import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from typing import List, Dict, Tuple
from tqdm.auto import tqdm


def compute_data_range(dataloader:DataLoader, device=torch.device("cpu")):
    global_min = torch.tensor(float("inf")).to(device)
    global_max = torch.tensor(float("-inf")).to(device)

    for _, target in dataloader:
        # assuming batch = (input, target)
        target = target.to(device)

        batch_min = target.min()
        batch_max = target.max()

        global_min = torch.minimum(global_min, batch_min)
        global_max = torch.maximum(global_max, batch_max)

    data_range = (global_max - global_min).item()

    return data_range


def train_step(model:nn.Module, 
               train_loader:DataLoader, 
               criterion:nn.Module, 
               optimizer:Optimizer,
               device:torch.device, 
               ssim_metric, 
               psnr_metric) -> Tuple[float]:
    
    model.train()
    total_loss = 0
    total_samples = 0

    ssim_metric.reset()
    psnr_metric.reset()

    for X, Y in tqdm(train_loader, desc="Training Step", leave=False, total=len(train_loader)):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()

        batch_size = X.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        ssim_metric.update(Y_pred, Y)
        psnr_metric.update(Y_pred, Y)
        
    avg_loss = total_loss/total_samples
    avg_ssim = ssim_metric.compute().item()
    avg_psnr = psnr_metric.compute().item()

    return avg_loss, avg_ssim, avg_psnr


def val_step(model:nn.Module, 
             val_loader:DataLoader, 
             criterion:nn.Module, 
             device:torch.device, 
             ssim_metric, 
             psnr_metric) -> Tuple[float]:
    
    model.eval()
    total_loss = 0
    total_samples = 0

    ssim_metric.reset()
    psnr_metric.reset()

    with torch.no_grad():
        for X, Y in tqdm(val_loader, desc="Validation Step", leave=False, total=len(val_loader)):
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            
            batch_size = X.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            ssim_metric.update(Y_pred, Y)
            psnr_metric.update(Y_pred, Y)

    avg_loss = total_loss/total_samples
    avg_ssim = ssim_metric.compute().item()
    avg_psnr = psnr_metric.compute().item()


    return avg_loss, avg_ssim, avg_psnr


def model_fit(model:nn.Module, 
              train_loader:DataLoader, 
              val_loader:DataLoader, 
              loss_function:nn.Module, 
              optimizer:Optimizer,
              device:torch.device,
              total_epochs:int, 
              latest_checkpoint_path:str=None,
              save_dir:str="./checkpoint"
              ) -> Dict:
    
    start_epoch = 0
    best_val_ssim = 0.0
    history = {"train_loss":[], "train_ssim":[], "train_psnr":[], 
               "val_loss":[], "val_ssim":[], "val_psnr":[]}
    
    os.makedirs(save_dir, exist_ok=True)
    if latest_checkpoint_path is None:
        latest_checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")

    if os.path.exists(latest_checkpoint_path):
        print(f"[INFO] LOADING CHECKPOINT FROM {latest_checkpoint_path} ..........")
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
        start_epoch = latest_checkpoint["epoch"]
        best_val_ssim = latest_checkpoint["best_val_ssim"]
        history = latest_checkpoint["history"]
        print(f"[INFO] RESUMING TRAINING AFTER EPOCH {start_epoch} ..........")
        print(f"[INFO] CURRENT BEST VALIDATION SSIM {best_val_ssim:0.6f} ..........")
    else:
        print(f"[INFO] NO CHECKPOINT FOUND. STARTING TRAINING FROM SCRATCH ..........")

    if start_epoch >= total_epochs:
        print(f"[INFO] TRAINING COMPLETED ALREADY. LAST TRAINED EPOCH {start_epoch} ..........")

    train_data_range = compute_data_range(dataloader=train_loader, device=device)
    TRAINSSIM = StructuralSimilarityIndexMeasure(data_range=train_data_range).to(device=device)
    TRAINPSNR = PeakSignalNoiseRatio(data_range=train_data_range).to(device=device)

    val_data_range = compute_data_range(dataloader=val_loader, device=device)
    VALSSIM = StructuralSimilarityIndexMeasure(data_range=val_data_range).to(device=device)
    VALPSNR = PeakSignalNoiseRatio(data_range=val_data_range).to(device=device)


    for epoch in tqdm(range(start_epoch, total_epochs), desc="Epochs", total=total_epochs, initial=start_epoch, leave=False):
        trainLOSS, trainSSIM, trainPSNR = train_step(model=model, train_loader=train_loader, 
                                                     criterion=loss_function, optimizer=optimizer, device=device,
                                                     ssim_metric=TRAINSSIM, psnr_metric=TRAINPSNR)

        valLOSS, valSSIM, valPSNR = val_step(model=model, val_loader=val_loader, criterion=loss_function, device=device, 
                                             ssim_metric=VALSSIM, psnr_metric=VALPSNR)

        history["train_loss"].append(trainLOSS); history["val_loss"].append(valLOSS)
        history["train_ssim"].append(trainSSIM); history["val_ssim"].append(valSSIM)
        history["train_psnr"].append(trainPSNR); history["val_psnr"].append(valPSNR)

        print(f"||      Epoch {epoch+1}/{total_epochs} Summary      ||")
        print(f"train_loss:{trainLOSS:0.6f}, train_ssim:{trainSSIM:0.6f}, train_psnr:{trainPSNR:0.6f}")
        print(f"valid_loss:{valLOSS:0.6f}, valid_ssim:{valSSIM:0.6f}, valid_psnr:{valPSNR:0.6f}") 



        is_best = False
        if valSSIM > best_val_ssim:
            print(f"[INFO] VALIDATION SSIM IMPROVED FROM {best_val_ssim:0.6f} TO {valSSIM:0.6f}.")
            best_val_ssim = valSSIM
            is_best = True

        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "history":history,
            "best_val_ssim":best_val_ssim
        }

        torch.save(checkpoint, latest_checkpoint_path)

        if is_best:
            best_checkpoint_path = os.path.join(save_dir, "best_checkpoint_v2_MSELoss.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"[INFO] BEST CHECKPOINT SAVED AT {best_checkpoint_path}")

    return history