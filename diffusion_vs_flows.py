
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from cleanfid import fid
import scipy

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256 # Reduced batch size for stability
IMG_SIZE = 128 # Reduced resolution for faster training in this demo context
EPOCHS = 50 # Small number of epochs for demonstration
LR = 1e-3

# Setup directories
os.makedirs("results", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Data Loading
# -----------------------------------------------------------------------------
def get_dataloader():
    print("Loading AFHQ dataset...")
    # Using 'huggan/AFHQ' - simplified to 'dog' category or just all for simplicity
    # Since AFHQ is large, we stream or load a subset if possible. 
    # For this script we will load the first N samples to simulate the dataset if full download is too slow,
    # but 'huggan/AFHQ' usually requires full download.
    # To keep it robust, we'll try to load 'train' split.
    
    try:
        dataset = load_dataset("huggan/AFHQ", split="train[:2000]") # Limit to 2000 images for speed
    except Exception as e:
        print(f"Failed to load huggan/AFHQ directly: {e}. Trying alternative or dummy.")
        # Fallback or error handling
        raise e

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
    ])

    def transforms_fn(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms_fn)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader

# -----------------------------------------------------------------------------
# 2. Diffusion Model (Simplified DDPM / Unet)
# -----------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified UNET
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # Input: 3x32x32
        self.inc = nn.Conv2d(3, 64, 3, padding=1)
        self.down1 = Block(64, 128, 32) # -> 128x16x16
        self.down2 = Block(128, 256, 32) # -> 256x8x8
        self.up1 = Block(256, 128, 32, up=True) # -> 128x16x16
        self.up2 = Block(128, 64, 32, up=True) # -> 64x32x32
        self.outc = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x = self.up1(x3, t)
        # Simple skip connection (add)
        x = x + x2 
        x = self.up2(x, t)
        x = x + x1
        return self.outc(x)

class Diffusion:
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.timesteps = 300 # Reduced for speed
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(DEVICE)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.to(DEVICE))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def get_loss(self, x_0):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=DEVICE).long()
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise, noise_pred)

    @torch.no_grad()
    def sample(self, shape, start_img=None):
        if start_img is not None:
            img = start_img
        else:
            img = torch.randn(shape, device=DEVICE)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling Diffusion', total=self.timesteps, leave=False):
            t = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)
            img = self.p_sample(img, t, i)
            if i % 100 == 0 or i == 0:
                imgs.append(img.cpu())
        return img, imgs

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return mean
        else:
            posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise

# -----------------------------------------------------------------------------
# 3. Normalizing Flow Model (RealNVP)
# -----------------------------------------------------------------------------
class FlowCouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        self.in_channels = in_channels
        # Logic to match torch.chunk(2, dim=1) behavior
        # chunk(2) on 3 channels -> split sizes (2, 1)
        self.ch1 = (in_channels + 1) // 2
        self.ch2 = in_channels - self.ch1
        
        # Scale and translation networks
        self.st_net = nn.Sequential(
            nn.Conv2d(self.ch1, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, self.ch2 * 2, 3, padding=1)
        )
    
    def forward(self, x, forward=True):
        x1, x2 = x.chunk(2, dim=1)
        st = self.st_net(x1)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        
        if forward:
            # Forward: x -> z (Normalizing)
            z1 = x1
            z2 = x2 * torch.exp(s) + t
            z = torch.cat([z1, z2], dim=1)
            log_det = torch.sum(s.view(s.size(0), -1), dim=1)
            return z, log_det
        else:
            # Inverse: z -> x (Generative)
            z1 = x1
            z2 = x2
            x1 = z1
            x2 = (z2 - t) * torch.exp(-s)
            x = torch.cat([x1, x2], dim=1)
            return x

class RealNVP(nn.Module):
    def __init__(self, num_scales=3, in_channels=3):
        super().__init__()
        self.layers = nn.ModuleList()
        # Squeeze logic is implicit or we just treat channels. 
        # For simplicity on low res, we just use channel coupling.
        # We need to permute channels between layers to mix information.
        current_channels = in_channels
        self.num_layers = 6
        
        for i in range(self.num_layers):
            self.layers.append(FlowCouplingLayer(current_channels))
            
    def forward(self, x):
        # x -> z
        log_det_sum = 0
        z = x
        for i, layer in enumerate(self.layers):
            z, log_det = layer(z, forward=True)
            log_det_sum += log_det
            # Permute channels (simple flip)
            z = z.flip(1) 
        return z, log_det_sum

    def inverse(self, z):
        # z -> x
        x = z
        for i, layer in enumerate(reversed(self.layers)):
            x = x.flip(1) # Undo permutation first
            x = layer(x, forward=False)
        return x
    
    def get_loss(self, x):
        z, log_det = self.forward(x)
        # NLL = 0.5 * ||z||^2 - log_det + const
        prior_ll = -0.5 * torch.sum(z.view(z.size(0), -1) ** 2, dim=1) - 0.5 * z.size(1) * z.size(2) * z.size(3) * np.log(2 * np.pi)
        log_likelihood = prior_ll + log_det
        return -torch.mean(log_likelihood)

# -----------------------------------------------------------------------------
# 4. Training & Visualization
# -----------------------------------------------------------------------------

def save_images(images, path, grid_size=8):
    # images: [B, C, H, W] in [-1, 1] usually or unnormalized
    # Denormalize to [0, 1]
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    grid = torch.cat([torch.cat([img for img in images[i*grid_size:(i+1)*grid_size]], dim=2) for i in range(len(images)//grid_size)], dim=1)
    plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.savefig(path)
    plt.close()

def main():
    print(f"Running on {DEVICE}")
    dataloader = get_dataloader()
    
    # ------------------
    # Train Diffusion
    # ------------------
    print("\n--- Training Diffusion Model ---")
    diff_unet = SimpleUnet().to(DEVICE)
    diffusion = Diffusion(diff_unet)
    opt_diff = torch.optim.Adam(diff_unet.parameters(), lr=LR)
    
    diff_losses = []
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        for batch in pbar:
            batch = batch["pixel_values"].to(DEVICE)
            loss = diffusion.get_loss(batch)
            
            opt_diff.zero_grad()
            loss.backward()
            opt_diff.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        diff_losses.append(epoch_loss / len(dataloader))
    
    print(f"Diffusion Final Loss: {diff_losses[-1]:.4f}")

    # ------------------
    # Train Flow
    # ------------------
    print("\n--- Training Flow Model ---")
    flow_model = RealNVP(in_channels=3).to(DEVICE)
    opt_flow = torch.optim.Adam(flow_model.parameters(), lr=LR)
    
    flow_losses = []
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        for batch in pbar:
            batch = batch["pixel_values"].to(DEVICE)
            loss = flow_model.get_loss(batch)
            
            opt_flow.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            opt_flow.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        flow_losses.append(epoch_loss / len(dataloader))
    
    print(f"Flow Final Loss: {flow_losses[-1]:.4f}")

    # ------------------
    # Visualization
    # ------------------
    print("\n--- Generating Visualizations ---")
    
    # Get a fixed batch
    batch = next(iter(dataloader))
    fixed_batch = batch["pixel_values"].to(DEVICE)[:8]
    
    # Combined Diffusion: Image -> Noise -> Reconstructed
    # 1. Forward process (Noising)
    t_vals = [0, 50, 150, 299]
    forward_imgs = []
    start_noise = None
    
    for t in t_vals:
        t_tensor = torch.full((8,), t, device=DEVICE, dtype=torch.long)
        n_img, _ = diffusion.forward_diffusion_sample(fixed_batch, t_tensor)
        forward_imgs.append(n_img)
        if t == 299:
            start_noise = n_img
            
    # 2. Reverse process (Denoising) from start_noise
    sample_out, intermediates = diffusion.sample((8, 3, IMG_SIZE, IMG_SIZE), start_img=start_noise)
    
    combined_vis = []
    # Forward: Original, t=50, t=150, t=299
    combined_vis.extend(forward_imgs)
    
    # Reverse intermediates: t=200, t=100. (t=0 is sample_out)
    if len(intermediates) >= 1:
        combined_vis.append(intermediates[0].to(DEVICE)) # t=200
    if len(intermediates) >= 2:
        combined_vis.append(intermediates[1].to(DEVICE)) # t=100
        
    combined_vis.append(sample_out) # Final reconstructed
    
    save_images(torch.cat(combined_vis, dim=0), "results/diffusion_combined.png", grid_size=8)
    
    # Flow: Normalizing Direction (x -> z)
    z_out, _ = flow_model(fixed_batch)
    # Visualizing z is tricky as it's not strictly an image, but we can interpret it as one
    # to show the "noise-like" structure
    # Normalize z for vis: centered at 0 with unit var mostly
    save_images(torch.cat([fixed_batch, z_out], dim=0), "results/flow_normalizing.png", grid_size=8)
    
    # Flow: Forward Direction (z -> x)
    z_sample = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    x_gen = flow_model.inverse(z_sample)
    save_images(torch.cat([z_sample, x_gen], dim=0), "results/flow_forward.png", grid_size=8)
    
    # ------------------
    # Evaluation (FID)
    # ------------------
    print("\n--- Computing FID ---")
    # Generate 100 samples for each for FID estimation (small sample for speed)
    def gen_diffusion_fn(batch_size):
        s, _ = diffusion.sample((batch_size, 3, IMG_SIZE, IMG_SIZE))
        return ((s + 1) / 2 * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

    def gen_flow_fn(batch_size):
        z = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        x = flow_model.inverse(z)
        return ((x + 1) / 2 * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        
    print("Generating samples for FID...")
    # CleanFID typically needs path to real dataset. We can use the cached HF cache or save real images to folder.
    os.makedirs("results/real_images", exist_ok=True)
    os.makedirs("results/diff_images", exist_ok=True)
    os.makedirs("results/flow_images", exist_ok=True)
    
    # Save real reference images
    ref_batch = next(iter(dataloader))["pixel_values"]
    for i, img in enumerate(ref_batch):
        img_np = ((img + 1) / 2 * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        plt.imsave(f"results/real_images/{i}.png", img_np)
        if i >= 64: break

    # Save Diffusion samples
    d_imgs = gen_diffusion_fn(64)
    for i, img in enumerate(d_imgs):
        plt.imsave(f"results/diff_images/{i}.png", img)

    # Save Flow samples
    f_imgs = gen_flow_fn(64)
    for i, img in enumerate(f_imgs):
        plt.imsave(f"results/flow_images/{i}.png", img)
        
    try:
        score_diff = fid.compute_fid("results/diff_images", "results/real_images", device=DEVICE, num_workers=0)
        score_flow = fid.compute_fid("results/flow_images", "results/real_images", device=DEVICE, num_workers=0)
        
        print(f"Diffusion FID: {score_diff:.4f}")
        print(f"Flow FID: {score_flow:.4f}")
        
    except Exception as e:
        print(f"FID Calculation failed: {e}")
        score_diff = -1
        score_flow = -1

    # Save final metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Diffusion Loss: {diff_losses[-1]}\n")
        f.write(f"Flow Loss: {flow_losses[-1]}\n")
        f.write(f"Diffusion FID: {score_diff}\n")
        f.write(f"Flow FID: {score_flow}\n")

if __name__ == "__main__":
    main()
