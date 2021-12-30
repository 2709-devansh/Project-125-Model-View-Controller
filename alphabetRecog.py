import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.txt')['labels']

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
classNumber = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 500, train_size = 3500, random_state = 10)
xtrainScale = X_train/255.0
xtestScale = X_test/255.0

lg = LogisticRegression(solver = 'saga', multi_class = 'multinomial')

def get_prediction(image):
    imagePIL = Image.open(image)
    imageL = imagePIL.convert('L')
    imageLR = imageL.resize((22, 30), Image.ANTIALIAS)
    imageLRI = PIL.ImageOps.invert(imageLR)    
    minPixel = np.percentile(imageLRI, 20)
    imageLRIS = np.clip(imageLRI - minPixel, 0, 255)
    maxPixel = np.max(imageLRI)
    imageLRIS = np.asarray(imageLRIS)/maxPixel
    testSample = np.array(imageLRIS).reshape(1, 660)
    prediction = lg.predict(testSample)
    return prediction[0]