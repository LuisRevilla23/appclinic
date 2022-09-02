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
        if len(dataset.pixel_array.shape) == 3:
            dcm_slice = st_container.slider('Corte', 1, dataset.pixel_array.shape[0], 1,key=key_slider)
            ax.imshow(dataset.pixel_array[dcm_slice-1, :, :], cmap="gray")
        else:
            ax.imshow(dataset.pixel_array, cmap="gray")
        # PLOTEO DE IMAGEN
        st_container.pyplot(fig=fig)
    except NameError:
        st_container.write(NameError)
        st_container.write(WrongFileType("No es un archivo DCM o imagen."))
        raise st_container.stop()
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
                    #Saving upload
                    with open(os.path.join("images/saved","saved_"+uploaded_file.name),"wb") as f:
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
    # Definir columnas
    colms = st.columns([2,2,1,1,6])
    # Definir los encabezados de columna
    fields = ["Archivo", 'Tamaño', '👁', '⬇']
    for col, field_name in zip(colms, fields):
        # header
        col.subheader(field_name)
    # Definir la ruta de archivos a mostrar
    files_path = 'images/DCMs'
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
    # Plotear la imagen si está definido el Path
    if st.session_state.file_to_show != "":
        dcm_visualization(st.session_state.file_to_show,colms[4],False,key_slider="slider"+st.session_state.file_to_show)
    else:
        colms[4].empty()
