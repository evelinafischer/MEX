"""
Simple REACTIV run on real data - NO changes to original code
Test
"""
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm

# Import from original reactiv.py (no modifications needed)
from reactiv import Stack2reactiv, reactiv_image, fusion2polar4reactiv

def load_tif_stack(file_list):
    """Simple function to load GeoTIFF files"""
    with rasterio.open(file_list[0]) as src:
        h, w = src.height, src.width
    
    stack = np.zeros((h, w, len(file_list)), dtype=np.float32)
    
    for i, f in enumerate(tqdm(file_list, desc="Loading")):
        with rasterio.open(f) as src:
            stack[:, :, i] = src.read(1)
    
    return stack

# Main
data_folder = Path("../Datasets")

vh_files = sorted(glob.glob(str(data_folder / "*_VH_amplitude.tif")))
vv_files = sorted(glob.glob(str(data_folder / "*_VV_amplitude.tif")))

print(f"Found {len(vh_files)} VH + {len(vv_files)} VV images")

# Load
print("\nLoading data...")
stack_vh = load_tif_stack(vh_files)
stack_vv = load_tif_stack(vv_files)

# Run REACTIV (original function, no changes)
print("\nRunning REACTIV...")
CV_vh, K_vh, Amax_vh = Stack2reactiv(stack_vh, timeaxis=2, L=4.9)
CV_vv, K_vv, Amax_vv = Stack2reactiv(stack_vv, timeaxis=2, L=4.9)

# Fuse (original function)
CV, K, Amax = fusion2polar4reactiv(CV_vh, K_vh, Amax_vh, CV_vv, K_vv, Amax_vv)

# Visualize
rgb = reactiv_image(CV, K, Amax)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(CV, cmap='jet', vmin=0, vmax=1)
plt.title('CV')
plt.colorbar()
plt.subplot(132)
plt.imshow(K, cmap='hsv', vmin=0, vmax=1)
plt.title('K (Time)')
plt.colorbar()
plt.subplot(133)
plt.imshow(rgb)
plt.title('REACTIV RGB')
plt.tight_layout()

# Efter att du har kört REACTIV...

# Skapa en mer detaljerad visualisering
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Rad 1: VH-kanal
im0 = axes[0, 0].imshow(CV_vh, cmap='jet', vmin=0, vmax=1)
axes[0, 0].set_title('CV - VH (Variabilitet)')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(K_vh, cmap='hsv', vmin=0, vmax=1)
axes[0, 1].set_title('K - VH (När max inträffar)')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(Amax_vh, cmap='gray')
axes[0, 2].set_title('Amax - VH (Maximal amplitud)')
plt.colorbar(im2, ax=axes[0, 2])

# Rad 2: VV-kanal
im3 = axes[1, 0].imshow(CV_vv, cmap='jet', vmin=0, vmax=1)
axes[1, 0].set_title('CV - VV (Variabilitet)')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(K_vv, cmap='hsv', vmin=0, vmax=1)
axes[1, 1].set_title('K - VV (När max inträffar)')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(Amax_vv, cmap='gray')
axes[1, 2].set_title('Amax - VV (Maximal amplitud)')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("../Results/Sentinel1/detailed_results.png", dpi=200)
plt.show()

# Fusionerat resultat
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

im0 = axes[0].imshow(CV, cmap='jet', vmin=0, vmax=1)
axes[0].set_title('Fusionerad CV')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(K, cmap='hsv', vmin=0, vmax=1)
axes[1].set_title('Fusionerad K (Temporal position)')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(Amax, cmap='gray')
axes[2].set_title('Fusionerad Amax')
plt.colorbar(im2, ax=axes[2])

axes[3].imshow(rgb)
axes[3].set_title('REACTIV RGB\n(Hue=K, Sat=CV, Val=Amax)')

plt.tight_layout()
plt.savefig("../Results/Sentinel1/fused_results.png", dpi=200)
plt.show()

Path("../Results/Sentinel1").mkdir(parents=True, exist_ok=True)
plt.savefig("../Results/Sentinel1/result.png", dpi=200)
print("\n✅ Done! Saved to Results/Sentinel1/result.png")
plt.show()