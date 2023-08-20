import tensorflow 
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import tqdm
import pickle
import cv2
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model = tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])
feature_list=pickle.load(open('embeddings.pkl','rb'))
filenames=pickle.load(open('filenames.pkl','rb'))
feature_list=np.array(feature_list)
print(feature_list.shape)
img_path=r"/Users/harshadhikari/Downloads/1_mfs-13126-p-03-red_1.jpg"
img=image.load_img(img_path,target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
preprocessed_img=preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result=result/norm(result)
neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices=neighbors.kneighbors([normalized_result])
print(indices)
for file in indices[0]:
    print(filenames[file])