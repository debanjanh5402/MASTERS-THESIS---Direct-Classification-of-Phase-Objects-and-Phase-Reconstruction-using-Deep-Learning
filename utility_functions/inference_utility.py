import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm.auto import tqdm

from engines.reconstructor_engine import compute_data_range

def get_result_for_dataset(model:nn.Module, loader:DataLoader, device:torch.device):
    data_range = compute_data_range(loader, device)

    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)

    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Predicting", total=len(loader)):
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            ssim_metric.update(Y_pred, Y)
            psnr_metric.update(Y_pred, Y)
    avg_ssim = ssim_metric.compute().item()
    avg_psnr = psnr_metric.compute().item()
    print(f"[INFO] FOR THE DATASET AVERAGE SSIM: {avg_ssim*100:0.4f}%, AVERAGE PSNR: {avg_psnr:0.4f}dB")


def predict(X:torch.Tensor, model:nn.Module, device:torch.device, Y:torch.Tensor=None):
    model.eval()

    with torch.no_grad():
        X = X.to(device)
        Y_pred = model(X)
    if Y is not None:
        Y = Y.to(device)
        data_range = Y.max() - Y.min()
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
        ssim_metric.update(Y_pred, Y)
        psnr_metric.update(Y_pred, Y)
        return Y_pred.cpu(), ssim_metric.compute().item(), psnr_metric.compute().item()
    else:
        return Y_pred.cpu()