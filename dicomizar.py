import SimpleITK as sitk
import cv2
import numpy as np

def dicomizar(imagen, nombre):
    # Convertir a SimpleITK
    img = sitk.GetImageFromArray(imagen)
    # Guardar
    sitk.WriteImage(img, nombre)

if __name__ == "__main__":
    # Leer la imagen
    img = cv2.imread("C:/Users/Emilio/Documents/GitHub/appclinic/patient64555_study1_view1_frontal.jpg")
    # Dicomizar
    img=np.expand_dims(img, axis=0)
    print(img.shape)
    dicomizar(img, "dicom1.dcm")


