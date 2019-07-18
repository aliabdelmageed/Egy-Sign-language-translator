from flask import Flask, request, jsonify, render_template, request
from keras.optimizers import Adam
from keras.applications import imagenet_utils
import cv2
import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Activation, Flatten, Dropout
import logging
import json
import urllib.request
from urllib.request import Request, urlopen
import base64
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

global model
app = Flask(__name__,)

def load_img(url):
    image_url = url.split(',')[1]
    image_url = image_url.replace(" ", "+")
    image_array = base64.b64decode(image_url)
    image_array = np.fromstring(image_array, np.uint8)
    image_array = cv2.imdecode(image_array, -1)
    return image_array  

def preprocess_img(img):
    img = cv2.resize(img,(300,300))
    img = np.reshape(img,[1,300,300,3])
    return img

def build_finetune_model():
    base_model = ResNet50(weights=None,include_top=False,input_shape=(300, 300, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    for fc in [1024, 1024]:
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x) 
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    finetune_model.load_weights('F:/Egyptian Sign Language Translator/weights.h5') 
    finetune_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.1), metrics=['accuracy'])
    print("Model is loaded")                                    
    return finetune_model

model = build_finetune_model()
def predict(img):
    img=preprocess_img(img)
    preds = model.predict(img)
    predsstr = preds.argmax()
    return str(predsstr)

@app.route('/classify', methods=['GET'])
def classify():
    image_url = request.args.get('imageurl')
    image_array = load_img(image_url)
    class_index = predict(image_array)
    class_list = ["أنا","يغلق","يستمر","يأكل","أين","يسمع","يفتح","قصير","طويل","يشاهد"]
    class_name = class_list[int(class_index)]
    print(class_name)
    result = []
    result.append({"class_name":class_name})
    return jsonify({'results':result})


@app.route('/', methods=['GET','POST'])
def root():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host = '127.0.0.8',debug=False)                   