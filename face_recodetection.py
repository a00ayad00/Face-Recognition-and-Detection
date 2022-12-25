import os
import glob

import cv2
import numpy as np

from face_recognition import face_encodings, compare_faces, face_locations, face_distance





def name_and_feature_extraction(images_path):
    '''
    Parameters
    ----------
    images_path : str
        Path for your images in the database.

    Returns
    -------
    encodings : list
        List of the extracted features from your images.
        Number of the extracted encodings has to equal the num of your images.
    names : list
        Names of your images.
        Number of the extracted names has to equal the num of your images.
    '''
    images_path = glob.glob(images_path+'/*')
    
    encodings = []
    names = []

    for img_path in images_path:

        # Features Extraction
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_enc = face_encodings(img)[0]
        encodings.append(img_enc)

        # Name Extraction
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]
        names.append(img_name)

    return encodings, names


def detect_and_recognize(img, encodings, names):
    '''
    Parameters
    ----------
    img : array
        RGB image array.
    encodings : list or tuple
        Features of your own images to compare.
    names : list or tuple
        Names of your own images to make the recognition.

    Returns
    -------
    pts : array
        Coordinates of the detected faces in the img.
    face_names : list
        Names of the detected faces in the img.

    '''
    
    face_names=[]
    
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_locs = face_locations(rgb)
    face_encs = face_encodings(rgb, face_locs)
    
    for face_enc in face_encs:
        matches = compare_faces(encodings, face_enc)
        name = 'Unknown'
        
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]
        
        face_dis = face_distance(encodings, face_enc)
        idx = np.argmin(face_dis)
        
        if matches[idx]:
            name = names[idx]
            
        face_names.append(name)
        
    pts = np.array(face_locs)
        
    return pts, face_names






