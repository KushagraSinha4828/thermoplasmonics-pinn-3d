import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import gc

from PGI_Hnet import PhysicsGuidedInversionUNet, AsyncLagrangianLoss
# ==========================================
# 2. SYNCHRONOUS DATASET
# ==========================================

# ==========================================
# 1. SYNCHRONOUS DATASET (UNCHANGED)
# ==========================================
class TQSIDatasetRAM(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        N = len(self.file_paths)
        
        print(f"[*] Pre-allocating contiguous memory blocks for {N} files...")
        self.phase  = torch.empty((N, 1, 128, 128), dtype=torch.float16)
        self.temp   = torch.empty((N, 1, 32, 128, 128), dtype=torch.float16)
        self.t_inv  = torch.empty((N, 1, 32, 128, 128), dtype=torch.float16)
        self.k      = torch.empty((N, 1, 32, 128, 128), dtype=torch.float16)
        self.beta   = torch.empty((N, 1, 32, 128, 128), dtype=torch.float16)
        self.source = torch.empty((N, 1, 32, 128, 128), dtype=torch.float16)
        self.z0_idx = torch.empty((N,), dtype=torch.long)
        
        print("[*] Sliding data directly into memory blocks (Zero Fragmentation!)...")
        for i, path in enumerate(tqdm(self.file_paths, desc="Loading to RAM")):
            data = torch.load(path, weights_only=False, map_location='cpu')
            self.phase[i]  = data['phase'].squeeze(0).nan_to_num_(posinf=0.0, neginf=0.0)
            self.temp[i]   = data['temp'].squeeze(0).nan_to_num_(posinf=0.0, neginf=0.0)
            self.t_inv[i]  = data['t_inv'].squeeze(0).nan_to_num_(posinf=0.0, neginf=0.0)
            self.k[i]      = data['k'].squeeze(0).nan_to_num_(posinf=0.0, neginf=0.0)
            self.beta[i]   = data['beta'].squeeze(0).nan_to_num_(posinf=0.0, neginf=0.0)
            self.source[i] = data['source'].squeeze(0).nan_to_num_(posinf=0.0, neginf=0.0)
            self.z0_idx[i] = data['z0_idx'].squeeze(0)

    def __len__(self): 
        return len(self.file_paths)
        
    def __getitem__(self, idx): 
        return {
            'phase': self.phase[idx],
            'temp': self.temp[idx],
            't_inv': self.t_inv[idx],
            'k': self.k[idx],
            'beta': self.beta[idx],
            'source': self.source[idx],
            'z0_idx': self.z0_idx[idx],
            'source_mask': (self.source[idx] > 0).float()
        }

# ==========================================
# 3. LOAD & APPEND FINE-TUNING LOOP
# ==========================================
def resume_and_finetune(data_dir, epochs=40, lr=3e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cudnn.benchmark = True  
    cudnn.deterministic = False 
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    
    ckpt_dir = r"D:\kushagra\experimental data\simulation data loader\PGI_Hnet checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("") 
    LOAD_VERSION = input("Enter PRE-TRAINED version to load (e.g., 7.305): ").strip()
    # Allowing easy overwrite by just pressing Enter
    NEW_VERSION  = input(f"Enter NEW version to save as (Press Enter to overwrite V{LOAD_VERSION}): ").strip()
    
    if not NEW_VERSION:
        NEW_VERSION = LOAD_VERSION
    print("")
    
    load_path = os.path.join(ckpt_dir, f"pinn_latest_V{LOAD_VERSION}.pth")
    save_path = os.path.join(ckpt_dir, f"pinn_latest_V{NEW_VERSION}.pth")
    
    dataset = TQSIDatasetRAM(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 
    
    model = PhysicsGuidedInversionUNet(bf=8)
    
                if not os.path.exists(load_path):
        print(f"[!] Warning: Checkpoint not found at: {load_path}")
        print(f"[*] Initializing model with NEW random weights and creating {load_path}...")
        torch.save(model.state_dict(), load_path)
    else:
        print(f"[*] Loading pre-trained weights from: {load_path}", flush=True)
        model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True), strict=True)
    
    if load_path == save_path:
        print(f"[*] Info: Model will be saved to the same file (In-place overwrite / Resume).")
        
    model = model.to(device)

    # Use our newly defined asynchronous loss
    criterion = AsyncLagrangianLoss(dx_um=12.8/128, dz_um=200.0/32).to(device) 
    
    try:
        optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=1e-4, fused=True)
    except Exception:
        optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda') 
    
    accumulation_steps = 2 
    
    print(f"[*] Starting fine-tuning for {epochs} epochs. Outputting to: V{NEW_VERSION}...\n", flush=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
                        metrics_sum = {k: torch.tensor(0.0, device=device) for k in 
                       ['loss', 'shape', 'peak', 'grad', 'z_prof', 'lap', 'ssim', 'z_corr', 'z_err']}
        
        optimizer.zero_grad(set_to_none=True)
        loop = tqdm(dataloader, desc=f"Epoch [{epoch:03d}/{epochs:03d}]", leave=False)
        
        for batch_idx, batch in enumerate(loop):
            
            phase       = batch['phase'].to(device, non_blocking=True)
            temp        = batch['temp'].to(device, non_blocking=True)
            t_inv       = batch['t_inv'].to(device, non_blocking=True)
            k           = batch['k'].to(device, non_blocking=True)
            beta        = batch['beta'].to(device, non_blocking=True)
            source_mask = batch['source_mask'].to(device, non_blocking=True)
            true_z      = batch['z0_idx'].to(device, non_blocking=True)
            
            with autocast('cuda'):
                T_pred, dynamic_peak = model(phase, t_inv, k, beta)
                loss, metrics = criterion(T_pred, dynamic_peak, temp, k, source_mask)
                loss = loss / accumulation_steps 
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                bs = phase.size(0)
                pred_z = T_pred.view(bs, T_pred.size(2), -1).max(dim=2)[0].argmax(dim=1)
                
                # Accumulate directly on the GPU
                # Accumulate directly on the GPU
                metrics_sum['z_corr'] += (pred_z == true_z).sum()
                metrics_sum['z_err'] += torch.abs(pred_z - true_z).float().sum()
                
                real_loss_tensor = loss.detach() * accumulation_steps
                metrics_sum['loss'] += real_loss_tensor * bs
                
                # Added 'ssim' to the tracking loop
                for k_metric in ['shape', 'peak', 'grad', 'z_prof', 'lap', 'ssim']:
                    metrics_sum[k_metric] += metrics[k_metric] * bs
            
                        if batch_idx % 25 == 0:
                current_z_acc = (metrics_sum['z_corr'].item() / ((batch_idx + 1) * bs)) * 100
                loop.set_postfix(Loss=f"{real_loss_tensor.item():.4f}", Z_Acc=f"{current_z_acc:.1f}%")
            
            del phase, temp, t_inv, k, beta, source_mask, T_pred, dynamic_peak, loss
            
        scheduler.step()
        n = len(dataset)
        2
        # Pull everything to CPU (.item()) ONLY at the very end of the epoch
        print(f"\nEpoch [{epoch:03d}] | Loss: {metrics_sum['loss'].item()/n:.4f} | "
              f"Shape: {metrics_sum['shape'].item()/n:.4f} | SSIM-L: {metrics_sum['ssim'].item()/n:.4f} | "
              f"Peak-Err: {metrics_sum['peak'].item()/n:.4f} | Grad: {metrics_sum['grad'].item()/n:.4f} | "
              f"Z-Prof: {metrics_sum['z_prof'].item()/n:.4f} | Z-Acc: {(metrics_sum['z_corr'].item()/n)*100:.1f}%", flush=True)
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    DATA_DIRECTORY = r"D:\kushagra\experimental data\simulation data loader\tqsi_data_diverse_V11"
    resume_and_finetune(data_dir=DATA_DIRECTORY, epochs=50, lr=3e-5)