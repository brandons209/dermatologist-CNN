from keras.models import load_model
from preprocess_data import process_single_image
import numpy as np
import sys

def prediction(img_path, model):
    diease_names = ['melanoma', 'nevus', 'seborrheic keratosis']
    return diease_names[np.argmax(model.predict(process_single_image(img_path)))]

model = load_model(sys.arg[2])

print("That lesion is predicted to be: {}".format(prediction(sys.argv[1], model)))
