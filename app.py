import imp
from unittest import result
from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np


#model = Sequential()

# model.add(Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model = keras.models.load_model('HAM_Cnn.h5')



COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (32,32))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 32,32,3)
    prediction = model.predict(img_arr)
    CATEGORIES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    result = []
    result.append(CATEGORIES[prediction.argmax()])
    result.append(np.amax(prediction))
    print(result)
    
    COUNT += 1
    return render_template('prediction.html', data=result)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



