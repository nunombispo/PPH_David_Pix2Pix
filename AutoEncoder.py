#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=Te3YieMUYd8
from PIL import Image
from tensorflow import keras

"""
@author: Sreenivas Bhattiprolu
Good example to demo image reconstruction using autoencoders
Try different optimizers and loss
To launch tensorboard type this in the console: !tensorboard --logdir=logs/ --host localhost --port 8088
then go to: http://localhost:8088/
"""
from Settings import Settings
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import os
from tqdm import tqdm

import tensorflow as tf
import time

SIZE = 256

from PyQt5.QtCore import *

import time
import traceback, sys


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, edit_line):
        self.edit_line = edit_line

    def printSummary(self, text):
        self.edit_line.insertPlainText('\n' + text)

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        self.printSummary("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        self.printSummary("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        self.printSummary("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.printSummary("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        self.printSummary("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        self.printSummary("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        self.printSummary("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        self.printSummary("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        self.printSummary("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.printSummary("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        self.printSummary("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.printSummary("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        self.printSummary("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.printSummary("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


class AutoEncoder:
    def __init__(self, folder_source, folder_output, folder_model, edit_line, epoch_number, original_image, original_model):
        self.folder_source = folder_source
        self.folder_output = folder_output
        self.folder_model = folder_model
        self.edit_line = edit_line
        self.threadpool = QThreadPool()
        self.progress_callback = None
        self.epoch_number = epoch_number
        self.original_image = original_image
        self.original_model = original_model

    def printSummary(self, text):
        self.edit_line.insertPlainText('\n' + str(text))

    def callback_model(self, text):
        self.progress_callback.emit(text)

    def execute_training(self,  progress_callback):
        img_data = []
        path1 = self.folder_source
        files = os.listdir(path1)
        for i in tqdm(files):
            img = cv2.imread(path1 + '/' + i, 1)  # Change 0 to 1 for color images
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB to show images in true colors
            img = cv2.resize(img, (SIZE, SIZE))
            img_data.append(img_to_array(img))

        img2_data = []
        path2 = self.folder_output
        files = os.listdir(path2)
        for i in tqdm(files):
            img = cv2.imread(path2 + '/' + i, 1)  # Change 0 to 1 for color images
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB to show images in true colors
            img = cv2.resize(img, (SIZE, SIZE))
            img2_data.append(img_to_array(img))

        img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
        img_array = img_array.astype('float32') / 255.

        img_array2 = np.reshape(img2_data, (len(img2_data), SIZE, SIZE, 3))
        img_array2 = img_array2.astype('float32') / 255.

        start = time.time()
        self.printSummary('\nStart Time: ' + str(start))

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.summary(print_fn=self.callback_model)

        #callbacks = [tf.keras.callbacks.TensorBoard(log_dir='einstein_logs')]

        self.printSummary('\nUsing Epoch Number:' + str(self.epoch_number))
        self.printSummary('\n')

        model.fit(img_array, img_array2,
                  epochs=int(self.epoch_number),
                  shuffle=True,
                  callbacks=[CustomCallback(self.edit_line)])

        finish = time.time()
        self.printSummary('\nEnd Time: ' + str(finish))
        self.printSummary('\nTotal Time = ' + str(finish - start))

        # Save Model
        self.printSummary('\nSaving model: ' + os.path.join(self.folder_model, 'Model.model'))
        model.save(os.path.join(self.folder_model, 'Model.model'))

    def execute_model(self,  progress_callback):
        self.printSummary('\nUsing model : ' + self.original_model)
        model = load_model(self.original_model)
        model.summary(print_fn=self.callback_model)

        path_images = self.original_image
        files = os.listdir(path_images)
        for file in files:
            img_data3 = []
            img3 = cv2.imread(os.path.join(path_images, file), 1)  # Change 0 to 1 for color images
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB to show images in true colors
            img3 = cv2.resize(img3, (SIZE, SIZE))
            img_data3.append(img_to_array(img3))

            img_array3 = np.reshape(img_data3, (len(img_data3), SIZE, SIZE, 3))
            img_array3 = img_array3.astype('float32') / 255.

            self.printSummary('\nPredicting file: ' + os.path.join(path_images, file))
            pred = model.predict(img_array3)

            self.printSummary('\nSaving file: ' + os.path.join(self.original_model, file))
            image = Image.fromarray(pred[0], 'RGB')
            image.save(os.path.join(self.original_model, file))


    def progress_fn(self, text):
        self.printSummary(text)

    def execute_this_fn(self, progress_callback):
        self.progress_callback = progress_callback
        self.execute_training(progress_callback)

    def execute_this_fn_run(self, progress_callback):
        self.progress_callback = progress_callback
        self.execute_model(progress_callback)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def run_thread(self, is_run=False):
        if is_run:
            # Pass the function to execute
            worker = Worker(self.execute_this_fn_run)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.progress_fn)

            # Execute
            self.threadpool.start(worker)
        else:
            # Pass the function to execute
            worker = Worker(self.execute_this_fn)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.progress_fn)

            # Execute
            self.threadpool.start(worker)


if __name__ == '__main__':
    settings = Settings()
    settings.read_settings()
    auto_encoder = AutoEncoder(settings.get_folder_source(), settings.get_folder_output(),
                               settings.get_folder_model())
    auto_encoder.execute(False)