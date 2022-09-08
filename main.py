from dataclasses import dataclass
import io
from operator import truediv
import streamlit as st
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from humanize import naturalsize


#para red
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
from network_transfer import NetworkTransferLearning
from prediccion_pytorch import prediccion
import glob
import cv2

model_t = 'vit_base_patch8_224'#'resnet101d'
modelo2 = NetworkTransferLearning.load_from_checkpoint("D:/emilio/epoch=0-valid_loss=0.000.ckpt", type_net=model_t, num_classes = 2)
#modelo2=modelo2.cuda()


filename=glob.glob("D:/app_senales/dataset2/test/edema/*")
device="cpu"
pred=prediccion(modelo2,filename[1],device)



class WrongFileType(ValueError):
    pass
# UTILIDAD
def read_image(imgpath):
    if (str(imgpath.name).lower().find("jpg") != -1) or (str(imgpath).find("png") != -1):
        sample = Image.open(imgpath)
        return np.array(sample)
    if str(imgpath.name).lower().find("dcm") != -1:
        img = pydicom.dcmread(imgpath).pixel_array
        return img

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False
def dcm_visualization(dcm_file,st_container,is_bytes=True,key_slider="slider"):
    try:
        # Definicion del dataset
        if is_bytes:
            dataset = pydicom.dcmread(io.BytesIO(dcm_file.getvalue()))
        else:
            dataset = pydicom.dcmread(dcm_file)
        # Configuracion para el ploteo
        fig, ax = plt.subplots()
        plt.axis('off')
        # Condicional: dimensiones del archivo DCM
        ax.imshow(dataset.pixel_array, cmap="gray")

        #if len(dataset.pixel_array.shape) == 3:
        #    if dataset.pixel_array.shape[0]!=1:
        #        ax.imshow(dataset.pixel_array[:, :,:], cmap="gray")

        #else:
        #    ax.imshow(dataset.pixel_array, cmap="gray")
        # PLOTEO DE IMAGEN
        st_container.pyplot(fig=fig)
    except NameError:
        st_container.write(NameError)
        st_container.write(WrongFileType("No es un archivo DCM o imagen."))
        raise st_container.stop()

def clasificacion(dcm_file,st_container):
    pred=prediccion(modelo2,dcm_file,device)
    if pred[0] ==1:
        st.markdown("<h1 style='text-align: center; color: white;'>Predicción: Edema</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: white;'>Predicción: No Edema</h1>", unsafe_allow_html=True)


def guardarpng(dcm_file):
    dataset = pydicom.dcmread(io.BytesIO(dcm_file.getvalue()))
    arr=dataset.pixel_array
    cv2.imwrite(os.path.join("images/png/"+uploaded_file.name[:-4]+".png"),arr)


# ------------------------------------------------------------------

# CONFIGURACION DE LA PAGINA
st.set_page_config(
    page_title="INTERMED",
    page_icon="images/Content/favicon.png",
    layout="wide",
    menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': None
     }
)
# TITULO
st.title("INTERMED")
st.subheader('Administración de archivos DICOM')
# PESTAÑAS
tab1, tab2, tab3 = st.tabs(["Carga", "Descarga","Convertidor a png"])
# Definir path de la imagen a mostrar
with tab1:
    # CARGA
    tab1_col1, tab1_col2 = st.columns([2,3])

    with st.container():
        with tab1_col1:
            # CARGA DE ARCHIVOS
            st.header("Carga de archivos DICOM")
            uploaded_file = st.file_uploader("Seleccione un archivo .DCM")
            if uploaded_file is not None:
                if st.button('Guardar'): 
                    guardarpng(uploaded_file)
                    #Saving upload
                    with open("images/saved/"+uploaded_file.name,"wb") as f:
                        f.write((uploaded_file).getbuffer())
                    st.success("Archivo guardado")

        with tab1_col2:
            # Condicional: Si la imagen está cargada
            if uploaded_file is not None:
                st.header("Visualización de imagen")
                # Leer la imagen
                imgdef = read_image(uploaded_file)
                # visualizar dcm file
                dcm_visualization(uploaded_file,st)

with tab2:
    # DESCARGA
    if 'file_to_show' not in st.session_state:
        st.session_state.file_to_show = ""


    if 'file_to_predict' not in st.session_state:
        st.session_state.file_to_predict= ""

    # Definir columnas
    colms = st.columns([2,2,1,1,1,6,1])
    # Definir los encabezados de columna
    fields = ["Archivo", 'Tamaño', '👁', '⬇',"Predicción"]
    for col, field_name in zip(colms, fields):
        # header
        col.subheader(field_name)
    # Definir la ruta de archivos a mostrar
    files_path = 'images/saved'
    files_path_predict='images/png'
    files = os.listdir(files_path)
    # Mostrar los archivos disponibles
    for file in files:
        # Nombre
        colms[0].write(file)
        # Tamaño
        size = os.stat(files_path+"/"+file).st_size
        colms[1].write(naturalsize(size))
        # Boton visualizar
        button_visualizate = colms[2].empty()  # crear placeholder
        if file.lower().find(".dcm") != -1:
            click_visualizate = button_visualizate.button("Ver", key='vis_'+file)
            if click_visualizate:
                    st.session_state.file_to_show = files_path+"/"+file
        else:
            button_visualizate.write("-")
        # Boton descargar
        button_download = colms[3].empty()  # crear placeholder
        if file.lower().find(".dcm") != -1:
            with open(files_path+"/"+file, 'rb') as f:
                click_download = button_download.download_button('Descargar', f, file_name='archivo.dmc',key='down_'+file)
        else:
            button_download.write("-")
        
        #Boton predecir
        button_predecir= colms[4].empty()  # crear placeholder
        if file.lower().find(".dcm") != -1:
            click_predecir = button_predecir.button("Predecir", key='pred_'+file)
            if click_predecir:
                st.session_state.file_to_predict = files_path_predict+"/"+file[:-4]+".png"
        else:
            button_predecir.write("-")


    # Plotear la imagen si está definido el Path
    if st.session_state.file_to_show != "":
        dcm_visualization(st.session_state.file_to_show,colms[5],False,key_slider="slider"+st.session_state.file_to_show)
    else:
        colms[5].empty()

    # Plotear la imagen si está definido el Path
    if st.session_state.file_to_predict != "":
        clasificacion(st.session_state.file_to_predict,colms[6])
    else:
        colms[6].empty()
