# 🗂️ ai/image_to_palette.py
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import colour

def extract_palette(image_file, num_colors=10):  # Changed default from 5 to 10
    """
    Extracts a color palette from an image file object and returns it in HEX, RGB, and HSL formats.
    """
    try:
        # 1. Open image from file object and resize for processing
        img = Image.open(image_file).convert("RGB")
        img.thumbnail((200, 200))  # Slightly larger for better color diversity
        pixels = np.array(img).reshape(-1, 3)
        
        # Remove very similar colors for better diversity
        unique_pixels = np.unique(pixels, axis=0)
        
        # Ensure we don't request more colors than available unique colors
        actual_num_colors = min(num_colors, len(unique_pixels), 15)  # Max 15 colors
        
        # Use at least 3 colors minimum
        actual_num_colors = max(3, actual_num_colors)
        
        # 2. Find dominant colors using K-means clustering
        kmeans = KMeans(
            n_clusters=actual_num_colors, 
            random_state=42, 
            n_init=10,
            max_iter=300  # More iterations for better clustering
        )
        kmeans.fit(pixels)
        dominant_colors_rgb_int = kmeans.cluster_centers_.astype(int)
        
        # 3. Sort colors by cluster size (most dominant first)
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        
        # Sort colors by frequency (most common first)
        sorted_indices = np.argsort(color_counts)[::-1]
        dominant_colors_rgb_int = dominant_colors_rgb_int[sorted_indices]
        
        # 4. Convert colors into the desired formats
        palette = []
        for i, color_rgb in enumerate(dominant_colors_rgb_int):
            color_rgb_float = color_rgb / 255.0

            # --- Perform Conversions ---
            hex_code = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}".upper()
            rgb_str = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            
            # Convert to HSL using colour-science
            hsl_raw = colour.RGB_to_HSL(color_rgb_float)
            hsl_str = f"hsl({hsl_raw[0]*360:.0f}, {hsl_raw[1]*100:.0f}%, {hsl_raw[2]*100:.0f}%)"

            palette.append({
                "hex": hex_code,
                "rgb": rgb_str,
                "hsl": hsl_str,
                "frequency": int(color_counts[sorted_indices[i]])  # Add frequency info
            })
            
        return palette

    except Exception as e:
        print(f"An error occurred during color extraction: {e}")
        return None

# ---------------------------
# Example run (for testing)
# ---------------------------
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use it in your Flask app with: from image_to_palette import extract_palette")
