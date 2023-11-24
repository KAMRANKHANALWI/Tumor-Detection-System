import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumorCategorical.h5')

image = cv2.imread('/Users//kamrankhanalwi//Desktop//BrainTumorProject//pred//pred5.jpg')

img = Image.fromarray(image)

img = img.resize((64,64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# result = model.predict_classes(img)
# print(result)

predict_img=model.predict(input_img) 
classes_img=np.argmax(predict_img)
# print(predict_img)
print(classes_img)