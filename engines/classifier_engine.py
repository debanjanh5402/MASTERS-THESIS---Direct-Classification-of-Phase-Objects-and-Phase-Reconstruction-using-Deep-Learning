import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm
from typing import Dict, List


# -------------------- Training Scheme --------------------
def train_step(model:nn.Module, 
               train_loader:DataLoader, 
               loss_fn:nn.Module, 
               optimizer:Optimizer, 
               device:torch.device) -> Dict[str, float]:
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in tqdm(train_loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)

    return {"train_loss": train_loss / len(train_loader), 
            "train_acc": train_acc / len(train_loader)}


# -------------------- Validation Scheme --------------------
def val_step(model:nn.Module, 
             val_loader:DataLoader, 
             loss_fn:nn.Module, 
             device:torch.device) -> Dict[str, float]:
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for X, y in tqdm(val_loader, desc="Validation", leave=False):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)          
            val_loss += loss_fn(y_pred, y).item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            val_acc += (y_pred_class == y).sum().item() / len(y)

    return {"val_loss": val_loss / len(val_loader), 
            "val_acc": val_acc / len(val_loader)}


# -------------------- Model fit function --------------------
def model_fit(model:nn.Module, 
              train_loader:DataLoader, 
              val_loader:DataLoader, 
              loss_fn:nn.Module, 
              optimizer:Optimizer, 
              device:torch.device, 
              epochs:int,
              latest_checkpoint_path:str="./latest_checkpoint.pth",
              best_checkpoint_path:str = "./best_checkpoint.pth") -> Dict[str, List[float]]:
    
    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    
    if os.path.exists(latest_checkpoint_path):
        print(f"[INFO] Loading checkpoint from {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint["history"]

        print(f"[INFO] Resuming from Epoch {start_epoch}...")
        print(f"[INFO] Current best validation accuracy: {best_val_acc:0.6f}")

    else:
        print(f"[INFO] No checkpoint found. Starting from scratch.....")

    if start_epoch > epochs:
        print(f"Training is already done. Last trained epoch:{start_epoch}")

    for epoch in tqdm(range(start_epoch, epochs), desc="Total Epochs", initial=start_epoch, total=epochs):
        train_results = train_step(model, train_loader, loss_fn, optimizer, device)
        val_results = val_step(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_results['train_loss']:0.6f} Acc: {train_results['train_acc']:0.4f} | Val Loss: {val_results['val_loss']:0.6f} Acc: {val_results['val_acc']:0.4f}")

        history["train_loss"].append(train_results["train_loss"])
        history["train_acc"].append(train_results["train_acc"])
        history["val_loss"].append(val_results["val_loss"])
        history["val_acc"].append(val_results["val_acc"])

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history
        }
        torch.save(checkpoint, latest_checkpoint_path)

        if val_results["val_acc"] > best_val_acc:
            print(f"[INFO] Validation accuracy improved from {best_val_acc:0.6f} to {val_results['val_acc']:0.6f}")
            best_val_acc = val_results["val_acc"]
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, best_checkpoint_path)
            print(f"[INFO] Best checkpoint saved at {best_checkpoint_path}")

    return history