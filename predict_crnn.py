import itertools

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from crnn_model import build_cnn
from cfg_crnn import *
import cv2
import numpy as np


K.set_learning_phase(0)

def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))
    print(out_best)
    out_best = [k for k, g in itertools.groupby(out_best)]
    print(out_best)

model = build_cnn(128, 64, max_text_len, False)
model.load_weights('model/crnn-30-29.957.h5')
img = cv2.imread('train/A02rh7345.jpg', cv2.IMREAD_GRAYSCALE)
img_pred = img.astype(np.float32)
img_pred = cv2.resize(img_pred, (128, 64))
img_pred = (img_pred/255.0)*2.0 - 1.0
print(img_pred.shape)
img_pred = img_pred.T
print(img_pred.shape)
img_pred = np.expand_dims(img_pred, axis=-1)
print(img_pred.shape)
img_pred = np.expand_dims(img_pred, axis=0)
print(img_pred.shape)
out = model.predict(img_pred)
print(out)
print(out.shape)
decode_label(out)