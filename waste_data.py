# input data processing for organic vs recyclable wastes

import cv2
import matplotlib.pyplot as mp
import numpy as np
import os
import random
from tqdm import tqdm

training_data=[]
size=70
DIVISION=["R","O"]
DIR="A:\BITS\AOS project\image classifier\wastes1" #directory of training docs

def data_input():  #function for processing test image        
    for div in DIVISION:
        root=os.path.join(DIR,div)
        var = DIVISION.index(div) 
        for img in tqdm(os.listdir(root)): #loops through each img in dir
            try:
                img_array=cv2.imread(os.path.join(root,img),cv2.IMREAD_GRAYSCALE) #reads img as gray-scale
                img_array_new=cv2.resize(img_array,(size, size)) #resizes images
                training_data.append([img_array_new,var])
            except Exception as e:
                pass

data_input()
random.shuffle(training_data) #shuffles data to prevent rote learning