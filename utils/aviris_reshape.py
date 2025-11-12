import numpy as np
import os
from spectral import envi
import torch

# Linux/WSL absolute path
base_dir = "/mnt/c/Users/aleks/Documents/GitHub/EDAA/f100517t01p00r10rdn_b"
hdr_file = os.path.join(base_dir, "f100517t01p00r10rdn_b_sc01_ort_img.hdr")
img_file = os.path.join(base_dir, "f100517t01p00r10rdn_b_sc01_ort_img")
gain_file = os.path.join(base_dir, "f100517t01p00r10rdn_b.gain")


# Load gain data
def load_aviris_gain():
    gain_data = np.loadtxt(gain_file)
    gain_factor = gain_data[:, 0] # First column (300, ..., 600, ..., 1200) (L,)
    #print(gain_data)
    #print(f"Shape: {gain_data.shape}")
    #print(f"Channelnumbers: {gain_data[:, 1]}")
    
    return gain_factor


# Load Aviris Dataset    
def load_aviris():
    # Read ENVI format file
    img = envi.open(hdr_file, img_file)

    # Crop to manageable size for testing
    # Read a 500x500 pixel region starting at (1000, 100)
    data = img.read_subregion((1000, 1500), (100, 600))  

    # Use memory mapping - doesn't load into RAM
    # data = img.open_memmap(writable=False)

    # Get metadata
    #print(f" Printing image shape: {img.shape} \r")    # (lines, samples, bands)
    #print(img.metadata) # All header info

    # Reshape to EDAA format (L, N)
    L = data.shape[2] # bands
    H = data.shape[0] # lines (height)
    W = data.shape[1] # samples (width)
    N = H * W         # total pixels

    Y = data.reshape(H * W, L).T # (N, L) -> transpose -> (L,N)

    # Optional: Apply gain correction
    if os.path.exists(gain_file):
        gain_factor = load_aviris_gain()
        #print(f"Y shape: {Y.shape}")
        #print(f"gain shape: {gain_factor.shape}")
        #print(f"Before: {Y[0, 0]}")
        Y = Y / gain_factor.reshape(-1,1) # Reshape to (L, 1) so it broadcast with (L, N)
        #print(f"After: {Y[0, 0]}")

    return Y, H, W, L, N


if __name__ == "__main__":
    # Only runs when: python aviris_reshape.py
    Y, H, W, L, N = load_aviris()
else:
    pass