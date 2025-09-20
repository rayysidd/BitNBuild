# ai/advanced_ai_palette.py
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import combinations

# -------------------------- Color conversion utilities --------------------------
def hex_to_rgb(hex_color):
    h = hex_color.strip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_xyz(rgb):
    def rgb_to_linear(c):
        c = c / 255.0
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4
    
    r, g, b = [rgb_to_linear(c) for c in rgb]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z

def xyz_to_lab(x, y, z):
    xr, yr, zr = 0.95047, 1.0, 1.08883
    x, y, z = x / xr, y / yr, z / zr
    
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return (L, a, b)

def lab_to_xyz(L, a, b):
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def inv_f(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16/116) / 7.787
    
    xr, yr, zr = 0.95047, 1.0, 1.08883
    x = inv_f(fx) * xr
    y = inv_f(fy) * yr
    z = inv_f(fz) * zr
    return x, y, z

def xyz_to_rgb(x, y, z):
    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    
    def to_srgb(u):
        if u <= 0.0031308:
            v = 12.92 * u
        else:
            v = 1.055 * u ** (1/2.4) - 0.055
        return int(round(max(0, min(1, v)) * 255))
    
    return (to_srgb(r_lin), to_srgb(g_lin), to_srgb(b_lin))

def hex_to_lab(h):
    return xyz_to_lab(*rgb_to_xyz(hex_to_rgb(h)))

def lab_to_hex(L, a, b):
    return '#{:02X}{:02X}{:02X}'.format(*xyz_to_rgb(*lab_to_xyz(L, a, b)))

def lab_distance(c1, c2):
    return math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3)))

def palette_hexes_to_lab_array(hex_list):
    return np.array([hex_to_lab(h) for h in hex_list], dtype=np.float32)

# -------------------------- WCAG contrast --------------------------
def relative_luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    
    def f(c): 
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    
    r, g, b = f(r), f(g), f(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(hex1, hex2):
    L1 = relative_luminance(hex_to_rgb(hex1))
    L2 = relative_luminance(hex_to_rgb(hex2))
    Lmax, Lmin = max(L1, L2), min(L1, L2)
    return (Lmax + 0.05) / (Lmin + 0.05)

# -------------------------- Scoring Functions --------------------------
def harmony_score(lab_palette):
    ab = lab_palette[:, 1:3]
    angles = np.degrees(np.arctan2(ab[:, 1], ab[:, 0])) % 360
    diffs = []
    K = len(angles)
    for i, j in combinations(range(K), 2):
        d = min(abs(angles[i] - angles[j]), 360 - abs(angles[i] - angles[j]))
        diffs.append(d)
    if not diffs:
        return 0.5
    mean_diff = np.mean(diffs)
    low, high = 20.0, 110.0
    val = (mean_diff - low) / (high - low)
    return float(max(0.0, min(1.0, val)))

def distinctness_score(lab_palette):
    dists = [lab_distance(lab_palette[i], lab_palette[j]) 
             for i, j in combinations(range(len(lab_palette)), 2)]
    mean_d = np.mean(dists) if dists else 0
    val = (mean_d - 6) / (40 - 6)
    return float(max(0.0, min(1.0, val)))

def contrast_score(lab_palette, roles):
    primary_idx = roles.get('primary', [])
    if not primary_idx:
        return 0.5
    
    primary_lab = lab_palette[primary_idx[0]]
    other_indices = roles.get('secondary', []) + roles.get('accent', [])
    if not other_indices:
        return 1.0
    
    scores = []
    for idx in other_indices:
        other_lab = lab_palette[idx]
        # Convert to RGB for contrast calculation
        r1, g1, b1 = xyz_to_rgb(*lab_to_xyz(*primary_lab))
        r2, g2, b2 = xyz_to_rgb(*lab_to_xyz(*other_lab))
        L1 = relative_luminance((r1, g1, b1))
        L2 = relative_luminance((r2, g2, b2))
        ratio = (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)
        score = 1 / (1 + math.exp(-1.5 * (ratio - 4.5)))
        scores.append(score)
    
    return float(np.mean(scores))

def cohesion_score(lab_palette):
    ab = lab_palette[:, 1:3]
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(ab)
    mean_dist = np.mean(dists)
    return float(np.exp(-mean_dist / 20))

def weight_score(lab_palette, roles):
    total = len(lab_palette)
    ratios = {'primary': 0.6, 'secondary': 0.25, 'accent': 0.15}
    score = 0.0
    for role in ratios:
        obs = len(roles.get(role, [])) / total
        score += max(0, 1 - abs(obs - ratios[role]))
    return float(score / 3)

# -------------------------- Model Definition --------------------------
class PaletteAestheticNet(nn.Module):
    def __init__(self, K=8, hidden_dim=128):
        super().__init__()
        self.K = K
        self.input_dim = K * 3
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, palette_lab):
        x = palette_lab.view(palette_lab.size(0), self.K * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x.squeeze(1)

# -------------------------- Role Assignment --------------------------
def assign_roles(hex_palette):
    lab = palette_hexes_to_lab_array(hex_palette)
    K = len(hex_palette)
    sat = np.linalg.norm(lab[:, 1:3], axis=1)
    L = lab[:, 0]
    
    idx_primary = [int(np.argmax(sat))]
    remaining = [i for i in range(K) if i not in idx_primary]
    
    if remaining:
        dists = [lab_distance(lab[i], lab[idx_primary[0]]) for i in remaining]
        idx_secondary = sorted(remaining, key=lambda i: -dists[i % len(dists)])[:2]
    else:
        idx_secondary = []
    
    remaining = [i for i in range(K) if i not in idx_primary + idx_secondary]
    idx_accent = sorted(remaining, key=lambda i: -sat[i])[:2] if remaining else []
    
    return {
        'primary': idx_primary,
        'secondary': idx_secondary,
        'accent': idx_accent
    }

# -------------------------- Composite Reward --------------------------
def composite_reward(hex_palette, roles, weights=None, model_L=None):
    lab_palette = palette_hexes_to_lab_array(hex_palette)
    
    H = harmony_score(lab_palette)
    D = distinctness_score(lab_palette)
    P = cohesion_score(lab_palette)
    W = weight_score(lab_palette, roles)
    C = contrast_score(lab_palette, roles)
    
    # Learned aesthetic score
    L = 0.5  # Default
    if model_L is not None:
        try:
            model_L.eval()
            pal = torch.from_numpy(lab_palette.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                L = float(model_L(pal).item())
                L = max(0.0, min(1.0, L))
        except:
            L = 0.5
    
    if weights is None:
        weights = {'H': 0.25, 'C': 0.25, 'D': 0.2, 'W': 0.1, 'P': 0.1, 'L': 0.1}
    
    reward = (weights['H'] * H + weights['C'] * C + weights['D'] * D + 
              weights['W'] * W + weights['P'] * P + weights['L'] * L)
    
    components = {'H': H, 'C': C, 'D': D, 'W': W, 'P': P, 'L': L}
    return float(reward), components

# -------------------------- Simple Optimization --------------------------
def optimize_palette(init_hex, steps=100, episodes_per_step=4, lr=1e-3, 
                    seed=42, model_L=None, **kwargs):
    """
    Simplified palette optimization using random search with gradient-like improvements
    """
    random.seed(seed)
    np.random.seed(seed)
    
    K = len(init_hex)
    best_palette = init_hex.copy()
    best_roles = assign_roles(best_palette)
    best_score, best_components = composite_reward(best_palette, best_roles, model_L=model_L)
    
    for step in range(steps):
        for episode in range(episodes_per_step):
            # Create a variation of current best palette
            candidate_palette = []
            for hex_color in best_palette:
                lab = hex_to_lab(hex_color)
                # Add small random variations
                noise_L = random.gauss(0, 5)
                noise_a = random.gauss(0, 10)
                noise_b = random.gauss(0, 10)
                
                new_lab = (
                    max(0, min(100, lab[0] + noise_L)),
                    max(-128, min(127, lab[1] + noise_a)),
                    max(-128, min(127, lab[2] + noise_b))
                )
                
                try:
                    new_hex = lab_to_hex(*new_lab)
                    candidate_palette.append(new_hex)
                except:
                    candidate_palette.append(hex_color)  # Keep original if conversion fails
            
            # Evaluate candidate
            candidate_roles = assign_roles(candidate_palette)
            candidate_score, candidate_components = composite_reward(
                candidate_palette, candidate_roles, model_L=model_L
            )
            
            # Update if better
            if candidate_score > best_score:
                best_palette = candidate_palette
                best_roles = candidate_roles
                best_score = candidate_score
                best_components = candidate_components
    
    return best_palette, best_roles, best_score, best_components

# -------------------------- Mock Training Function --------------------------
def train_model_aadb(*args, **kwargs):
    """
    Mock training function - returns a simple model
    In production, this would train on actual AADB dataset
    """
    print("Creating mock aesthetic model...")
    model = PaletteAestheticNet(K=8)
    # Initialize with reasonable weights
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    return model

if __name__ == "__main__":
    # Test the module
    test_palette = ["#FF6F61", "#FFD662", "#6B5B95", "#88B04B", "#F7CAC9"]
    roles = assign_roles(test_palette)
    score, components = composite_reward(test_palette, roles)
    print(f"Test palette score: {score:.4f}")
    print(f"Components: {components}")
