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
import timm


class NetworkTransferLearning(pl.LightningModule):
    def __init__(self, type_net, optimizer="Adam", num_classes = 5, lr = 1e-3, pretrained = True):
        super().__init__()
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr = lr
        self.backbone = timm.create_model(type_net, pretrained = pretrained)
        if pretrained:
            # freeze  weights
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.num_in_feat = self.backbone.get_classifier().in_features

        block = nn.Sequential(nn.Linear(self.num_in_feat, 1024), 
                              nn.Dropout(0.5), 
                              nn.Linear(1024, 512), 
                              #nn.Dropout(0.5), 
                              nn.Linear(512, self.num_classes))

        name, module = list(self.backbone.named_children())[-1]
        self.backbone._modules[name] = block        

        # 3 Loss function
        self.loss = nn.CrossEntropyLoss()

        # 4 Metrics
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy(compute_on_step=False)  
        self.test_acc  = Accuracy(compute_on_step=False)       

    def forward(self, x):
        out = self.backbone.forward(x)
        #out = self.backbone(x)
        return out

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = self.loss(outputs, targets)

        #preds = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        self.train_acc(preds, targets)

        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        return loss

    def training_epoch_end(self, outs):
        loss =self.train_acc.compute()        
        self.log('avg_train_acc',loss)
        print(f"avg_train_acc: {loss}, ", end=" ")
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)

        #preds = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        self.valid_acc(preds, targets)
        self.log('val_loss', loss)
        self.log('val_acc', self.valid_acc)
        return loss
    
    def validation_epoch_end(self, val_step_outputs):
        avg_val_acc = self.valid_acc.compute()
        self.log('avg_val_acc',avg_val_acc)
        print(f"avg_val_acc: {avg_val_acc}")

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)

        #preds = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        #print(preds)
        self.test_acc(preds, targets)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        return loss
    
    def test_epoch_end(self, val_step_outputs):
        self.log('avg_test_acc', self.test_acc.compute())

    def configure_optimizers(self):
      if self.optimizer == "Adam":
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
      else:
        #torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
      return optimizer

    def get_mean_std(self):
      return self.backbone.default_cfg["mean"], self.backbone.default_cfg["std"]

