import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import NamedTuple


class DiabeticFootData(NamedTuple):
    img: str
    root_temp: float
    left_foot_temp: float
    right_foot_temp: float

    def __repr__(self):
        return "Data(root_temp={}, left_foot_temp={}, right_foot_temp={})".format(self.root_temp, self.left_foot_temp, self.right_foot_temp)
    

class DiabeticFootSubject(NamedTuple):
    id: str
    diabetic: bool
    data: list


class DiabeticFootDataset():
    '''
        Diabetic Foot Dataset...

        Esta clase puede ser utilizada para los datos procesados del IACTEC y el INOAE
    '''
    def __init__(self, dataset_dirs: list):

        if not isinstance(dataset_dirs, (str, list)):
            raise IOError('dataset_dirs have to be a list of str or a unique str')

        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]

        self.data = []
        for dataset_dir in dataset_dirs:
            self.data.extend(self.__process__(dataset_dir))
            
    
    def __process__(self, root_dir: str):
        subjects = os.listdir(root_dir)
        dataset = [None] * len(subjects)
        for idx, subject in enumerate(subjects):
            subject_dir = os.path.join(root_dir, subject)
            data =  os.listdir(subject_dir)
            imgs_files = list(filter(lambda i: '.png' in i, data))
            csv_files = sorted(list(filter(lambda i: '.csv' in i, data)))
            
            subject_id = subject
            df = pd.read_csv(os.path.join(subject_dir,csv_files[0]), delimiter=",", index_col=0)
            diabetic = df['Diabetic'].item()
            data = []

            for csv_file in csv_files:
                df = self.__read_csv__(os.path.join(subject_dir,csv_file))
                basename = os.path.splitext(csv_file)[0]
                img = list(filter(lambda i: basename in i, imgs_files))
                dfu_data = DiabeticFootData(os.path.join(subject_dir, img[0]),
                                    df['Root temperature'].item(),
                                    df['Left foot temperature'].item(),
                                    df['Right foot temperature'].item())
                data.append(dfu_data)

            dataset[idx] = DiabeticFootSubject(subject_id, diabetic, data)
        
        return dataset

    def __read_csv__(self, path):
        return pd.read_csv(path, delimiter=",", index_col=0, dtype={
            'Age': 'float',
            'Sex': 'str',
            'Diabetic': 'bool',
            'Weight': 'float',
            'Root temperature': 'float',
            'Left foot temperature': 'float',
            'Right foot temperature': 'float'
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def getSubject(self, subject_name: str):
        return next((subject for subject in self.data if subject.id == subject_name), None)


class DiabeticFootTorchDataset(Dataset):
    '''
       This class is prepare to work with DiabeticFootDataset. The main purpose of this one 
       is to use in Deep Learning processes. 
    '''
    def __init__(self, dataset:DiabeticFootDataset, transform=transforms.Compose([transforms.ToTensor()])):
        '''
            @param dataset (ThermalAnalysis.dataset.DiabeticFootDataset): This parameters contains the dataset
                loaded and structured.
            @param transform (torchvision.transforms): Transform to apply to the dataset, by default
                it is used in order to convert the image input to tensor.
        '''
        super(DiabeticFootTorchDataset, self).__init__()

        if not isinstance(dataset, DiabeticFootDataset):
            raise IOError('dataset parameters have to be a DiabeticFootDataset object')

        self.imgs = []
        self.targets = []
        self.transform = transform
        self.__process__(dataset)

    def __process__(self, dataset):
        for subject in dataset:
            target = int(subject.diabetic)
            imgs = list(map(lambda x: x.img, subject.data))
            self.targets.extend([target]*len(imgs))
            self.imgs.extend(imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):        
        return (self.transform(Image.open(self.imgs[idx])), self.targets[idx])
