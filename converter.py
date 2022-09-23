from PIL import Image
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
im_frame = Image.open('C:/Users/Emilio/Documents/GitHub/appclinic/images/png/dicom1.png')
np_frame = np.array(im_frame.getdata(), dtype=np.uint8)[:,:3]
ds.Rows = im_frame.height
ds.Columns = im_frame.width
ds.PhotometricInterpretation = "RGB"
ds.SamplesPerPixel = 3
ds.BitsStored = 8
ds.BitsAllocated = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
ds.PixelData = np_frame.tobytes()
ds.save_as('ct_head_rgb.dcm')
