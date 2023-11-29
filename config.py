import os, sys

project_root_dir = os.path.dirname(__file__)

# Dependencies
ipdl_dir = os.path.join(project_root_dir, 'modules/IPDL')
if ipdl_dir not in sys.path:
    sys.path.append(ipdl_dir)

ae_dir = os.path.join(project_root_dir, 'modules/AutoEncoder')
if ae_dir not in sys.path:
    sys.path.append(ae_dir)

# Dataset Directory
BRAIN_HSI_DIR = '/home/abian/Data/Dataset/IUMA/Experimento (Abian)/'

# Results Directory
RESULTS_DIR = '/home/abian/Data/Thesis/Thesis/'