import os
import math
from sklearn.cluster import KMeans
from skimage import io, color
import math, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import combinations
from open_clip import create_model_and_transforms

# 1. Load AADB metadata
def load_aadb_metadata(aadb_csv_path, images_dir):
    """
    Load AADB CSV metadata and normalize scores.
    Automatically matches images case-insensitively.
    """
    df = pd.read_csv(aadb_csv_path)

    if 'score' not in df.columns:
        raise ValueError("Cannot find 'score' column in CSV")
    raw_scores = df['score']

    # Normalize scores to [0,1]
    df['score_norm'] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    if 'ImageFile' not in df.columns:
        raise ValueError("Cannot find 'ImageFile' column in CSV")

    # Build a dict of lower-case filename -> actual file path in folder
    all_files = {f.lower(): os.path.join(images_dir, f) for f in os.listdir(images_dir)}

    # Map CSV filenames to actual files
    df['image_path'] = df['ImageFile'].apply(lambda fn: all_files.get(fn.lower(), None))

    missing = df[df['image_path'].isnull()]
    if len(missing) > 0:
        print(f"Warning: {len(missing)} images not found. Examples:\n", missing.head())

    df = df[df['image_path'].notnull()].reset_index(drop=True)
    return df[['image_path', 'score_norm']]

# 2. Palette extraction (same as before)
def extract_palette_from_image(image_path, K=5, image_size=(256,256)):
    img = io.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    from skimage.transform import resize
    img_small = resize(img, image_size, anti_aliasing=True)
    img_small = img_small.astype(np.float32)
    if img_small.max() > 1.0:
        img_small = img_small / 255.0
    pixels = img_small.reshape(-1, 3)
    lab_pixels = color.rgb2lab(pixels.reshape((-1,1,3))).reshape(-1,3)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(lab_pixels)
    centers = kmeans.cluster_centers_
    centers = centers[np.argsort(centers[:,0])]  # sort by luminance
    return centers  # shape (K,3)

# 3. Dataset class
class AADBBasedPaletteDataset(Dataset):
    def __init__(self, metadata_df, K=5, transform=None):
        self.metadata = metadata_df
        self.K = K
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row['image_path']
        score = row['score_norm']
        try:
            palette = extract_palette_from_image(img_path, K=self.K)
        except Exception as e:
            # if image read fails, fallback
            palette = np.zeros((self.K,3), dtype=np.float32)
        return palette.astype(np.float32), np.float32(score)

# 4. Model
class PaletteAestheticNet(nn.Module):
    def __init__(self, K=5, hidden_dim=128):
        super().__init__()
        self.K = K
        self.input_dim = K * 3
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, palette_lab):
        # palette_lab: batch_size x K x 3
        x = palette_lab.view(palette_lab.size(0), self.K * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))  # output in [0,1]
        return x.squeeze(1)

# 5. Training using AADB

def train_model_aadb(aadb_csv, aadb_images_dir, K=8,
                     batch_size=32, epochs=15, lr=1e-3, val_frac=0.1):
    """
    Train PaletteAestheticNet on AADB dataset.
    """
    # Load metadata
    df = load_aadb_metadata(aadb_csv, aadb_images_dir)
    if len(df) == 0:
        raise ValueError("No images found. Check CSV and image folder paths.")

    # Split into train/val
    n_val = int(len(df) * val_frac)
    df_train = df.sample(n=len(df)-n_val, random_state=42).reset_index(drop=True)
    df_val   = df.drop(df_train.index).reset_index(drop=True)

    # Create datasets
    train_ds = AADBBasedPaletteDataset(df_train, K=K)
    val_ds   = AADBBasedPaletteDataset(df_val, K=K)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = PaletteAestheticNet(K=K, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for palettes, scores in train_loader:
            palettes = palettes.to(device)
            scores = scores.to(device)
            preds = model(palettes)
            loss = loss_fn(preds, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for palettes, scores in val_loader:
                palettes = palettes.to(device)
                scores = scores.to(device)
                preds = model(palettes)
                loss = loss_fn(preds, scores)
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, "
              f"train_loss={np.mean(train_losses):.4f}, "
              f"val_loss={np.mean(val_losses):.4f}")

    return model

# -------------------------- Color conversion utilities --------------------------
def hex_to_rgb(hex_color):
    h = hex_color.strip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def lab_to_rgb8(L, a, b):
    """
    Convert CIELAB to 8-bit RGB tuple (0-255)
    """
    x, y, z = lab_to_xyz(L, a, b)
    r_lin, g_lin, b_lin = xyz_to_rgb(x, y, z)
    # Ensure 0-255 integers
    r = max(0, min(255, int(round(r_lin))))
    g = max(0, min(255, int(round(g_lin))))
    b = max(0, min(255, int(round(b_lin))))
    return (r, g, b)

def rgb_to_linear(c):
    c = c/255.0
    if c <= 0.04045:
        return c/12.92
    return ((c+0.055)/1.055)**2.4

def rgb_to_xyz(rgb):
    r,g,b = [rgb_to_linear(c) for c in rgb]
    x = r*0.4124564 + g*0.3575761 + b*0.1804375
    y = r*0.2126729 + g*0.7151522 + b*0.0721750
    z = r*0.0193339 + g*0.1191920 + b*0.9503041
    return x,y,z

def xyz_to_lab(x,y,z):
    xr, yr, zr = 0.95047, 1.0, 1.08883
    x, y, z = x/xr, y/yr, z/zr
    def f(t):
        return t**(1/3) if t>0.008856 else 7.787*t + 16/116
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return (L,a,b)

def lab_to_xyz(L,a,b):
    fy = (L+16)/116
    fx = a/500 + fy
    fz = fy - b/200
    def inv_f(t):
        return t**3 if t**3>0.008856 else (t-16/116)/7.787
    xr, yr, zr = 0.95047,1.0,1.08883
    x = inv_f(fx)*xr
    y = inv_f(fy)*yr
    z = inv_f(fz)*zr
    return x,y,z

def xyz_to_rgb(x,y,z):
    r_lin =  3.2404542*x -1.5371385*y -0.4985314*z
    g_lin = -0.9692660*x +1.8760108*y +0.0415560*z
    b_lin =  0.0556434*x -0.2040259*y +1.0572252*z
    def to_srgb(u):
        if u <= 0.0031308:
            v = 12.92*u
        else:
            v = 1.055*u**(1/2.4) - 0.055
        return int(round(max(0,min(1,v))*255))
    return (to_srgb(r_lin), to_srgb(g_lin), to_srgb(b_lin))

def hex_to_lab(h):
    return xyz_to_lab(*rgb_to_xyz(hex_to_rgb(h)))

def lab_to_hex(L,a,b):
    return '#{:02X}{:02X}{:02X}'.format(*xyz_to_rgb(*lab_to_xyz(L,a,b)))

def lab_distance(c1,c2):
    return math.sqrt(sum((c1[i]-c2[i])**2 for i in range(3)))

def palette_hexes_to_lab_array(hex_list):
    return np.array([hex_to_lab(h) for h in hex_list], dtype=np.float32)

# -------------------------- WCAG contrast --------------------------
def relative_luminance(rgb):
    r,g,b = [x/255.0 for x in rgb]
    def f(c): return c/12.92 if c<=0.03928 else ((c+0.055)/1.055)**2.4
    r,g,b = f(r), f(g), f(b)
    return 0.2126*r + 0.7152*g + 0.0722*b

def contrast_ratio(hex1, hex2):
    L1 = relative_luminance(hex_to_rgb(hex1))
    L2 = relative_luminance(hex_to_rgb(hex2))
    Lmax,Lmin = max(L1,L2), min(L1,L2)
    return (Lmax+0.05)/(Lmin+0.05)

def wcag_score(fg, bg):
    ratio = contrast_ratio(fg,bg)
    # sigmoid mapping: 7+ = AAA, 4.5+ = AA
    return 1/(1+math.exp(-1.5*(ratio-4.5)))

def contrast_score(lab_palette, roles):
    """
    Compute contrast score for accessibility (WCAG).
    Penalizes any secondary/accent color that fails against primary.
    """
    primary_idx = roles.get('primary', [])
    if not primary_idx:
        return 0.5  # fallback
    primary_lab = lab_palette[primary_idx[0]]

    # Combine secondary and accent indices
    other_indices = roles.get('secondary', []) + roles.get('accent', [])
    if not other_indices:
        return 1.0  # nothing to compare, full score

    other_labs = lab_palette[other_indices]
    scores = []
    for lab in other_labs:
        # approximate WCAG contrast using relative luminance
        r1, g1, b1 = lab_to_rgb8(*primary_lab)
        r2, g2, b2 = lab_to_rgb8(*lab)
        L1 = relative_luminance((r1, g1, b1))
        L2 = relative_luminance((r2, g2, b2))
        ratio = (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)
        # sigmoid mapping: below 4.5 is heavily penalized
        score = 1/(1 + math.exp(-1.5*(ratio-4.5)))
        scores.append(score)
    return float(np.mean(scores))

# -------------------------- Harmony --------------------------
def harmony_score(lab_palette):
    ab = lab_palette[:,1:3]
    angles = np.degrees(np.arctan2(ab[:,1],ab[:,0]))%360
    diffs=[]
    K=len(angles)
    for i,j in combinations(range(K),2):
        d = min(abs(angles[i]-angles[j]), 360-abs(angles[i]-angles[j]))
        diffs.append(d)
    if not diffs: return 0.5
    mean_diff = np.mean(diffs)
    low, high = 20.0, 110.0
    val = (mean_diff-low)/(high-low)
    return float(max(0.0,min(1.0,val)))

# -------------------------- Distinctness --------------------------
def distinctness_score(lab_palette):
    dists=[lab_distance(lab_palette[i],lab_palette[j]) for i,j in combinations(range(len(lab_palette)),2)]
    mean_d = np.mean(dists) if dists else 0
    val = (mean_d-6)/(40-6)
    return float(max(0.0,min(1.0,val)))

# -------------------------- Weight / Dominance --------------------------
def weight_score(lab_palette, roles):
    total = len(lab_palette)
    ratios = {'primary':0.6,'secondary':0.25,'accent':0.15}
    score=0.0
    for role in ratios:
        obs = len(roles.get(role,[])) / total
        score += max(0,1-abs(obs-ratios[role]))
    return float(score/3)

# -------------------------- Novelty --------------------------
def novelty_score(palette_lab, reference_labs=None):
    if reference_labs is None:
        return 0.5
    dists = np.linalg.norm(reference_labs - palette_lab, axis=2).mean(axis=1)
    score = np.clip(dists.min() / 50.0, 0, 1)  # scale to [0,1]
    return float(score)

# -------------------------- Semantic relevance (CLIP) --------------------------
def semantic_score(palette_lab, prompt_embedding=None, clip_model=None):
    if prompt_embedding is None or clip_model is None:
        return 0.5
    # Convert Lab to approximate RGB image or embedding
    palette_rgb = np.array([lab_to_rgb8(*c) for c in palette_lab], dtype=np.float32)/255.0
    palette_tensor = torch.tensor(palette_rgb).unsqueeze(0)  # (1,K,3)
    with torch.no_grad():
        palette_embedding = clip_model.encode_image(palette_tensor)
    # cosine similarity
    score = F.cosine_similarity(palette_embedding, prompt_embedding, dim=-1).item()
    return float(np.clip(score,0,1))

# -------------------------- Cohesion --------------------------
def cohesion_score(lab_palette):
    from sklearn.metrics import pairwise_distances
    ab = lab_palette[:,1:3]
    dists = pairwise_distances(ab)
    mean_dist = np.mean(dists)
    return float(np.exp(-mean_dist/20))  # scale 20 chosen empirically

# Contrast penalty
def contrast_penalty(primary_lab, secondary_lab_list):
    def lab_to_luminance(L,a,b):
        # approximate relative luminance for WCAG
        r,g,b = xyz_to_rgb(*lab_to_xyz(L,a,b))
        R,G,B = np.array([r,g,b])/255.0
        R = R/12.92 if R<=0.03928 else ((R+0.055)/1.055)**2.4
        G = G/12.92 if G<=0.03928 else ((G+0.055)/1.055)**2.4
        B = B/12.92 if B<=0.03928 else ((B+0.055)/1.055)**2.4
        return 0.2126*R + 0.7152*G + 0.0722*B
    L_primary = lab_to_luminance(*primary_lab)
    penalty = 0
    for sec_lab in secondary_lab_list:
        L_sec = lab_to_luminance(*sec_lab)
        cr = (max(L_primary,L_sec)+0.05)/(min(L_primary,L_sec)+0.05)
        if cr < 4.5:  # penalize insufficient contrast
            penalty += (4.5 - cr)/4.5
    return np.clip(1 - penalty/len(secondary_lab_list),0,1)

# 6. learned_aesthetic_score hook

def learned_aesthetic_score(lab_palette, model, device=None):
    if model is None:
        return 0.5
    model.eval()
    pal = torch.from_numpy(lab_palette.astype(np.float32)).unsqueeze(0)  # (1, K,3)
    if device:
        pal = pal.to(device)
        model = model.to(device)
    with torch.no_grad():
        score = model(pal).item()
    return float(max(0.0, min(1.0, score)))

# -------------------------- Composite Reward --------------------------
def composite_reward(hex_palette, roles, weights=None, dataset=None, prompt_emb=None, model_L=None):
    lab_palette = np.array(palette_hexes_to_lab_array(hex_palette))
    H = harmony_score(lab_palette)
    D = distinctness_score(lab_palette)
    P = cohesion_score(lab_palette)
    W = weight_score(lab_palette, roles)
    C = contrast_score(lab_palette, roles)
    N = novelty_score(lab_palette, dataset)
    S = semantic_score(lab_palette, prompt_emb)
    L = learned_aesthetic_score(lab_palette, model_L)

    if weights is None:
        weights = {'H':0.2,'C':0.25,'D':0.15,'W':0.15,'S':0.05,'N':0.1,'P':0.05,'L':0.05}

    reward = (weights['H']*H + weights['C']*C + weights['D']*D + weights['W']*W +
              weights['S']*S + weights['N']*N + weights['P']*P + weights['L']*L)

    components = {'H':H,'C':C,'D':D,'W':W,'S':S,'N':N,'P':P,'L':L}
    return float(reward), components

# -------------------------- Role assignment --------------------------
def assign_roles(hex_palette):
    lab = palette_hexes_to_lab_array(hex_palette)
    K = len(hex_palette)
    sat = np.linalg.norm(lab[:,1:3],axis=1)
    L = lab[:,0]
    idx_primary = [int(np.argmax(sat))]
    remaining = [i for i in range(K) if i not in idx_primary]
    if remaining:
        dists = [lab_distance(lab[i],lab[idx_primary[0]]) for i in remaining]
        idx_secondary = sorted(remaining,key=lambda i:-dists[i%len(dists)])[:2]
    else: idx_secondary=[]
    remaining = [i for i in range(K) if i not in idx_primary+idx_secondary]
    idx_accent = sorted(remaining,key=lambda i:-sat[i])[:2] if remaining else []
    return {'primary':idx_primary,'secondary':idx_secondary,'accent':idx_accent}

# -------------------------- REINFORCE Optimization --------------------------
class PolicyNet(nn.Module):
    def __init__(self,K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K*3,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.mean = nn.Linear(64,K*3)
        self.log_std = nn.Parameter(torch.zeros(K*3)-3)
    def forward(self,x):
        h = self.net(x)
        mu = self.mean(h)
        std = torch.exp(self.log_std)
        return mu,std

def optimize_palette(init_hex, steps=200, episodes_per_step=8, lr=1e-3,
                     seed=42, model_L=None, dataset=None, prompt_emb=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    K = len(init_hex)
    lab_flat = torch.tensor(palette_hexes_to_lab_array(init_hex).flatten(), dtype=torch.float32)
    policy = PolicyNet(K)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    baseline = None
    
    for step in range(steps):
        batch_logp, batch_rewards = [], []
        for _ in range(episodes_per_step):
            mu, std = policy(lab_flat.unsqueeze(0))
            eps = torch.randn_like(mu) * std
            action = mu + eps
            new_lab = lab_flat + action.squeeze(0)
            
            # Convert to hex and assign roles
            new_hex = [lab_to_hex(*tuple(c)) for c in new_lab.detach().numpy().reshape(K, 3)]
            roles = assign_roles(new_hex)
            
            # Compute reward using full components
            r, _ = composite_reward(new_hex, roles, dataset=dataset, 
                                    prompt_emb=prompt_emb, model_L=model_L)
            
            logp = -0.5 * (((action / std)**2) + 2 * torch.log(std) + math.log(2*math.pi))
            batch_logp.append(logp.sum())
            batch_rewards.append(r)
        
        rewards_tensor = torch.tensor(batch_rewards)
        baseline = rewards_tensor.mean().item() if baseline is None else 0.9*baseline + 0.1*rewards_tensor.mean().item()
        
        loss = torch.mean(torch.stack([- (R - baseline) * lp for lp, R in zip(batch_logp, batch_rewards)]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Pick the best palette
    best_score = -1e9
    for _ in range(100):
        mu, std = policy(lab_flat.unsqueeze(0))
        action = mu + torch.randn_like(mu) * std
        new_lab = lab_flat + action.squeeze(0)
        new_hex = [lab_to_hex(*tuple(c)) for c in new_lab.detach().numpy().reshape(K, 3)]
        roles = assign_roles(new_hex)
        r, components = composite_reward(new_hex, roles, dataset=dataset, 
                                         prompt_emb=prompt_emb, model_L=model_L)
        if r > best_score:
            best_score = r
            best_palette = new_hex
            best_roles = roles
            best_components = components
    
    return best_palette, best_roles, best_score, best_components
 
 # -------------------------- Example Run --------------------------
if __name__=="__main__":
    # Define a path to save/load the model
    MODEL_SAVE_PATH = "palette_aesthetic_model.pth"
    K_VALUE = 8 # Number of colors in the palette

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if a pre-trained model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading pre-trained model from {MODEL_SAVE_PATH}...")
        # 1. Initialize the model architecture
        model_L = PaletteAestheticNet(K=K_VALUE)
        # 2. Load the saved weights into the model
        model_L.load_state_dict(torch.load(MODEL_SAVE_PATH))
        # 3. Set the model to evaluation mode (important!)
        model_L.eval()
        print("Model loaded successfully.")

    else:
        print(f"No pre-trained model found. Training a new one...")
        # --- Paths to AADB (only needed for training) ---
        aadb_csv = "AADB/Dataset_test.csv"
        aadb_images_dir = "AADB/datasetImages_warp256/datasetImages_warp256"

        # 1. Train the model as before
        model_L = train_model_aadb(aadb_csv, aadb_images_dir, K=K_VALUE, epochs=10)

        # 2. Save the trained model's state dictionary
        torch.save(model_L.state_dict(), MODEL_SAVE_PATH)
        print(f"Model trained and saved to {MODEL_SAVE_PATH}")

    # Move the model to the correct device (GPU or CPU)
    model_L.to(device)

    # -------------------- Example palette --------------------
    input_hex = ["#FF6F61","#FFD662","#6B5B95","#88B04B",
                 "#F7CAC9","#92A8D1","#955251","#B565A7"]
    print("\nInitial palette:", input_hex)

    # -------------------- Optimize with learned model --------------------
    # The loaded model is now ready to be used for optimization
    final_palette, roles, score, comps = optimize_palette(
        input_hex,
        steps=200,
        model_L=model_L  # <- plug in trained or loaded predictor
    )

    # -------------------- Results --------------------
    print("\nOptimized palette:", final_palette)
    print("Roles:", roles)
    print("Composite reward: {:.4f}".format(score))
    print("Components:", comps)

    # Show usage guidance
    primary = final_palette[roles['primary'][0]] if roles['primary'] else None
    secondary = [final_palette[i] for i in roles['secondary']]
    accent = [final_palette[i] for i in roles['accent']]
    print("\nUsage guidance:")
    print(f"- Primary: {primary}")
    print(f"- Secondary: {secondary}")
    print(f"- Accent: {accent}")