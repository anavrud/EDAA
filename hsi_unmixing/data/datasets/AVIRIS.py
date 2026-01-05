from .base import HSI
import numpy as np
import os
import sys
from spectral import envi
from hsi_unmixing.data.normalizers import GlobalMinMax as GMM
import torch

class AVIRISDataset(HSI): # Adding the methodes from HSI base class into AVIRISDataset class
    def __init__(self, 
                 row_range=(1000, 1500), 
                 col_range=(100, 600),
                 p=3,
                 normalizer=None,          
                 setter=None,
                 figs_dir="./figs",
                 **kwargs):
        
        self.load_aviris_data(row_range, col_range)

        # Set p (number of endmembers)
        self.p = p

        # Create dummy ground truth since the AVIRIS dont have any
        self.E = np.random.rand(self.L, self.p)      # Random endmembers (L, p)
        self.A = np.ones((self.p, self.N)) / self.p  # Uniform abundances (p, N)
        
        # Apply optional normalization only if requested through Hydra config
        if normalizer is not None:
            self.Y = normalizer.transform(self.Y)
            self.scaledE = normalizer.transform(self.E)
        else:
            self.scaledE = self.E.copy()

        # Set labels and metadata
        self.labels = [f"Material_{i+1}" for i in range(self.p)]    # List of material names for the endmembers
        self.shortname = "AVIRIS_Oil_detection"                     # Set a name for the dataset used in file naming and plot titles
        self.figs_dir = figs_dir                                    # Store the directory path where figures will be saved 
        os.makedirs(self.figs_dir, exist_ok=True)                   # Create the figures directory if it dosent exist

    # Load gain data    
    def load_aviris_gain(self, gain_file):
        gain_data = np.loadtxt(gain_file)   # First column (300, ..., 600, ..., 1200) (L,)
        gain_factor = gain_data[:, 0]

        return gain_factor
    
    # Load wavelength data from .spc file
    def load_aviris_wavelengths(self, spc_file):
        spc_data = np.loadtxt(spc_file)
        wavelengths = spc_data[:, 0]  # First column contains wavelengths in nm
        return wavelengths
    
    ## Load Aviris Dataset    
    def load_aviris_data(self, row_range, col_range):
        # Dataest run 10
        #base_dir = "/mnt/c/Users/aleks/Documents/GitHub/EDAA/f100517t01p00r10rdn_b"
        #hdr_file = os.path.join(base_dir, "f100517t01p00r10rdn_b_sc01_ort_img.hdr")
        #img_file = os.path.join(base_dir, "f100517t01p00r10rdn_b_sc01_ort_img")
        #gain_file = os.path.join(base_dir, "f100517t01p00r10rdn_b.gain")
        #spc_file = os.path.join(base_dir, "f100517t01p00r10rdn_b.spc")

        # Dataset run 11
        base_dir = "/mnt/c/Users/aleks/Documents/GitHub/EDAA/f100517t01p00r11rdn_b"
        hdr_file = os.path.join(base_dir, "f100517t01p00r11rdn_b_sc01_ort_img.hdr")
        img_file = os.path.join(base_dir, "f100517t01p00r11rdn_b_sc01_ort_img")
        gain_file = os.path.join(base_dir, "f100517t01p00r11rdn_b.gain")
        spc_file = os.path.join(base_dir, "f100517t01p00r11rdn_b.spc")
        
        # Read ENVI format file
        img = envi.open(hdr_file, img_file)
        data = img.read_subregion(row_range, col_range)
        
        # Reshape to EDAA format (L, N)
        self.L = data.shape[2]      # Bands
        self.H = data.shape[0]      # Lines (height)
        self.W = data.shape[1]      # Samples (width)
        self.N = self.H * self.W    # Total pixels
        self.Y = data.reshape(self.N, self.L).T    # (N, L) -> transpose -> (L,N)
        
        # Optional: Apply gain correction
        if os.path.exists(gain_file):
            gain_factor = self.load_aviris_gain(gain_file)
            self.Y = self.Y / gain_factor.reshape(-1, 1) # Reshape to (L, 1) so it broadcast with (L, N)
        
        # Load wavelengths if available
        if os.path.exists(spc_file):
            self.wavelengths = self.load_aviris_wavelengths(spc_file)
        else:
            self.wavelengths = None
        
        return self.Y, self.H, self.W, self.L, self.N
    
    # Makes the dataset object callable like a function
    def __call__(self, asTensor=False):

        if asTensor:
            Y = torch.Tensor(self.Y)
            E = torch.Tensor(self.E)
            A = torch.Tensor(self.A)

        else:
            Y = np.copy(self.Y)
            E = np.copy(self.E)
            A = np.copy(self.A)

        return (Y, E, A)

    # Set up how the dataset is displayed when you print it or view it in console
    def __repr__(self):
        msg = f"AVIRIS Dataset => {self.shortname}\n"
        msg += "---------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples, ({self.N} pixels),\n"
        msg += f"{self.p} endmembers (dummy for blind unmixing)\n"
        msg += f"Y range: [{self.Y.min():.3f}, {self.Y.max():.3f}]\n" # Shows data range after normalization, Y range: [0.000, 1.000]
        return msg # Return the complete string