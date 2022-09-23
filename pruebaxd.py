from pydicom import dcmread
from pynetdicom import AE, debug_logger
from pynetdicom.sop_class import CTImageStorage,MultiFrameGrayscaleByteSecondaryCaptureImageStorage

debug_logger()

# Initialise the Application Entity
ae = AE(ae_title=b'MY_STORAGE_SCU')

# Add a requested presentation context
ae.add_requested_context(CTImageStorage)
#ae.add_requested_context(DigitalXRayImageStorageForPresentation)
# Read in our DICOM CT dataset
#ds = dcmread('C:/Users/Emilio/OneDrive/Documentos/GitHub/appclinic/images/saved/ct_head_rgb.dcm')
ds = dcmread('C:/Users/Emilio/OneDrive/Documentos/GitHub/appclinic/images/saved/patient64542_study1_view2_lateral.dcm')

#ds = dcmread('C:/Users/Emilio/Downloads/1B5C1F93.dcm')
# Associate with peer AE at IP 127.0.0.1 and port 11112
assoc = ae.associate('216.238.69.235', 11112, ae_title=b'DCM4CHEE')
if assoc.is_established:
    # Use the C-STORE service to send the dataset
    # returns the response status as a pydicom Dataset
    status = assoc.send_c_store(ds)

    # Check the status of the storage request
    if status:
        # If the storage request succeeded this will be 0x0000
        print('C-STORE request status: 0x{0:04x}'.format(status.Status))
    else:
        print('Connection timed out, was aborted or received invalid response')

    # Release the association
    assoc.release()
else:
    print('Association rejected, aborted or never connected')
