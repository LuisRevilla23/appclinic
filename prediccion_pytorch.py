from PIL import Image
import glob
#@title **C. IMPORTANDO LIBRERIAS**
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchvision.models as models
#import torchmetrics
#from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.classification import Accuracy
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from datetime import datetime
import sys, os
from glob import glob
import imageio
from torch.utils.data import Dataset, DataLoader
import timm
import shutil
from tqdm.notebook  import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn import metrics as sk_metrics
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report

def prediccion(modelo,filename,device):
    dat_norm=modelo.get_mean_std()
    transform = transforms.Compose([
        transforms.Resize(size = 224),transforms.CenterCrop(size=224),transforms.ToTensor(),  transforms.Normalize(dat_norm[0], dat_norm[1])
    ])

    image = Image.open(filename).convert('RGB')
    img_tensor = transform(image)

    img_tensor=torch.unsqueeze(img_tensor,0)
    img_tensor=img_tensor.to(device)
    preds = modelo(img_tensor)
    y_pred =preds.argmax(dim=1).cpu()
    return y_pred.numpy()