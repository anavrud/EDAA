import logging

import scipy.io as sio
import numpy as np
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_estimates(Ehat, Ahat, hsi):
    # E hat: estimated endmembers (L x p)
    # hsi: dataset object with labels and wavelengths
    data = {
        "Ehat": Ehat,
        "Egt": hsi.scaledE,
        "Ahat": Ahat,
        "Agt": hsi.A,
        "H": hsi.H,
        "W": hsi.W,
    }
    sio.savemat("estimates.mat", data)


def save_endmembers_json(Ehat, hsi, filename="endmembers.json"):
    data = {
        "labels": hsi.labels,
        "endmembers": {}
    }
    
    # Add wavelengths if available
    if hasattr(hsi, 'wavelengths') and hsi.wavelengths is not None:
        data["wavelengths"] = hsi.wavelengths.tolist()
    else:
        data["wavelengths"] = list(range(hsi.L))
    
    # Save each endmember spectrum
    for idx, label in enumerate(hsi.labels):
        data["endmembers"][label] = Ehat[:, idx].tolist()
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved endmembers to {filename}")


def load_estimates(path: str):
    data = sio.loadmat(path)
    return data
