import os, sys

project_root_dir = os.path.dirname(__file__)

# Dependencies
ipdl_dir = os.path.join(project_root_dir, 'modules/IPDL')
if ipdl_dir not in sys.path:
    sys.path.append(ipdl_dir)

ae_dir = os.path.join(project_root_dir, 'modules/AutoEncoder')
if ae_dir not in sys.path:
    sys.path.append(ae_dir)

hyspeclab_dir = os.path.join(project_root_dir, 'modules/HySpecLab')
if hyspeclab_dir not in sys.path:
    sys.path.append(hyspeclab_dir)

#######################
# Important to modify #
#######################
# Dataset Directory
    # DFU
DFU_DATASET_DIR = "/home/abian/Data/Dataset/GTMA/DiabeticFootDataset/"

    # HSI Datasets
BRAIN_HSI_DIR = '/home/abian/Data/Dataset/IUMA/Experimento (Abian)/'

# Results Directory
RESULTS_DIR = '/media/abian/E6FC-C8C4/Thesis/Results/'