from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from crnn_model import build_cnn
from cfg_crnn import *
from crnn_image_generator import TextImageGenerator

K.set_learning_phase(0)

model = build_cnn(img_w, img_h, max_text_len, True)

try:
    model.load_weights('a.h5')
    print('load previous weight data...')
except:
    print('new weight data ....')
    pass

train_file_path = './train/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

test_file_path = './train/'
tiger_test = TextImageGenerator(test_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_test.build_data()

optim = Adam()
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='crnn-{epoch:02d}-{val_loss:.3f}.h5', monitor='loss', verbose=1, mode='min',
                             period=1)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optim)

model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=tiger_test.next_batch(),
                    validation_steps=(tiger_test.n / batch_size))
