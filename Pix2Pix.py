# example of pix2pix gan for satellite to map image-to-image translation
# load, split and scale the maps dataset ready for training
import os, time
from Settings import Settings
from os import listdir
from numpy import asarray
from numpy import savez_compressed
from numpy import zeros
from numpy import ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from numpy.random import randint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from PyQt5.QtCore import *
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


class Pix2Pix:
    def __init__(self, source_path, original_path, folder_model, learning_rate, batch_size, epoch_number, edit_line,
                 image, model):
        self.source_path = source_path
        self.original_path = original_path
        self.folder_model = folder_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_number = epoch_number
        self.edit_line = edit_line
        self.threadpool = QThreadPool()
        self.progress_callback = None
        self.image = image
        self.model = model

    def callback_model(self, text):
        self.progress_callback.emit(text)

    def printSummary(self, text):
        self.edit_line.insertPlainText('\n' + str(text))
        time.sleep(0.25)

    # define the discriminator model
    def define_discriminator(self, image_shape):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=image_shape)
        # target image input
        in_target_image = Input(shape=image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model

    # define an encoder block
    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add downsampling layer
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # conditionally add batch normalization
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)
        return g

    # define a decoder block
    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        # conditionally add dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)
        return g

    # define the standalone generator model
    def define_generator(self, image_shape=(256,256,3)):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=image_shape)
        # encoder model
        e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        # decoder model
        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
        # output
        g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model, image_shape):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # define the source image
        in_src = Input(shape=image_shape)
        # connect the source image to the generator input
        gen_out = g_model(in_src)
        # connect the source input and generator output to the discriminator input
        dis_out = d_model([in_src, gen_out])
        # src image as input, generated image and classification output
        model = Model(in_src, [dis_out, gen_out])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
        return model

    # load and prepare training images
    def load_real_samples(self, filename):
        # load compressed arrays
        data = load(filename)
        # unpack arrays
        X1, X2 = data['arr_0'], data['arr_1']
        # scale from [0,255] to [-1,1]
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5
        return [X1, X2]

    # select a batch of random samples, returns images and target
    def generate_real_samples(self, dataset, n_samples, patch_shape):
        # unpack dataset
        trainA, trainB = dataset
        # choose random instances
        ix = randint(0, trainA.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y

    # generate a batch of images, returns images and targets
    def generate_fake_samples(self, g_model, samples, patch_shape):
        # generate fake instance
        X = g_model.predict(samples)
        # create 'fake' class labels (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    # generate samples and save as a plot and save the model
    def summarize_performance(self, step, g_model, dataset):
        n_samples = int(self.learning_rate)
        # select a sample of input images
        [X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake samples
        X_fakeB, _ = self.generate_fake_samples(g_model, X_realA, 1)
        # scale all pixels from [-1,1] to [0,1]
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0
        # plot real source images
        # for i in range(n_samples):
        #     pyplot.subplot(3, n_samples, 1 + i)
        #     pyplot.axis('off')
        #     pyplot.imshow(X_realA[i])
        # # plot generated target image
        # for i in range(n_samples):
        #     pyplot.subplot(3, n_samples, 1 + n_samples + i)
        #     pyplot.axis('off')
        #     pyplot.imshow(X_fakeB[i])
        # # plot real target image
        # for i in range(n_samples):
        #     pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        #     pyplot.axis('off')
        #     pyplot.imshow(X_realB[i])
        # save plot to file
        #filename1 = 'plot_%06d.png' % (step+1)
        #pyplot.savefig(os.path.join(self.folder_model, filename1))
        #pyplot.close()
        # save the generator model
        filename2 = 'model_%06d.h5' % (step+1)
        g_model.save(os.path.join(self.folder_model, filename2))
        self.printSummary('\n>Saved: %s ' % (os.path.join(self.folder_model, filename2)))

    # train pix2pix models
    def train(self, d_model, g_model, gan_model, dataset):
        n_epochs = int(self.epoch_number)
        n_batch = int(self.batch_size)
        # determine the output square shape of the discriminator
        n_patch = d_model.output_shape[1]
        # unpack dataset
        trainA, trainB = dataset
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        self.printSummary("\ntrain A: " + str(len(trainA)) + " ; n_batch: " + str(n_batch) + " ; bat_per_epo: " + str(bat_per_epo))
        for i in range(n_steps):
            # select a batch of real samples
            [X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeB, y_fake = self.generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            self.printSummary('\n>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            # summarize model performance
            if (i+1) % (bat_per_epo * 10) == 0:
                self.summarize_performance(i, g_model, dataset)

    # load all images in a directory into memory
    def load_images(self):
        size = (256, 256)
        src_list, tar_list = list(), list()
        # enumerate filenames in directory, assume all are images
        for filename in listdir(self.source_path):
            # load and resize the image
            pixels = load_img(os.path.join(self.source_path, filename), target_size=size)
            # convert to numpy array
            pixels = img_to_array(pixels)
            # split into satellite and map
            src_list.append(pixels)

        for filename in listdir(self.original_path):
            # load and resize the image
            pixels = load_img(os.path.join(self.original_path, filename), target_size=size)
            # convert to numpy array
            pixels = img_to_array(pixels)
            # split into satellite and map
            tar_list.append(pixels)

        # load dataset
        [src_images, tar_images] = [asarray(src_list), asarray(tar_list)]
        self.printSummary(('Loaded: ', src_images.shape, tar_images.shape))
        # save as compressed numpy array
        filename = 'set_256.npz'
        savez_compressed(os.path.join(self.folder_model, filename), src_images, tar_images)
        self.printSummary(('Saved dataset: ', os.path.join(self.folder_model, filename)))

    def execute_training(self):
        # load images to dataset
        self.load_images()
        # load image data
        filename = 'set_256.npz'
        dataset = self.load_real_samples(os.path.join(self.folder_model, filename))
        self.printSummary(('Loaded', dataset[0].shape, dataset[1].shape))
        # define input shape based on the loaded dataset
        image_shape = dataset[0].shape[1:]
        # define the models
        self.printSummary("\nDefine Discriminator")
        d_model = self.define_discriminator(image_shape)
        self.printSummary("\nDefine Generator")
        g_model = self.define_generator(image_shape)
        # define the composite model
        self.printSummary("\nDefine Gan")
        gan_model = self.define_gan(g_model, d_model, image_shape)
        # train model
        self.printSummary("\nTrain Model")
        self.train(d_model, g_model, gan_model, dataset)
        self.printSummary("\nExecution ENDED")

    def progress_fn(self, text):
        self.printSummary(text)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        self.printSummary("EXECUTION COMPLETE!")

    def execute_this_fn(self, progress_callback):
        self.progress_callback = progress_callback
        self.execute_training()

    def execute_training_thread(self):
        worker = Worker(self.execute_this_fn)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        self.threadpool.start(worker)

    # example of loading a pix2pix model and using it for one-off image translation

    # load an image
    def load_image(self, filename, size=(256, 256)):
        # load image with the preferred size
        pixels = load_img(filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # scale from [0,255] to [-1,1]
        pixels = (pixels - 127.5) / 127.5
        # reshape to 1 sample
        pixels = expand_dims(pixels, 0)
        return pixels

    def execute_run(self):
        # load source image
        for filename in listdir(self.image):
            src_image = self.load_image(os.path.join(self.image, filename))
            self.printSummary('\nPredicting file: ' + os.path.join(self.image, filename))
            self.printSummary(('Loaded', src_image.shape))
            # load model

            self.printSummary('\nLoading model: ' + self.model)
            model = load_model(self.model)
            # generate image from source
            self.printSummary('\nPredicting image...')
            gen_image = model.predict(src_image)
            # scale from [-1,1] to [0,1]
            gen_image = (gen_image + 1) / 2.0

            self.printSummary('\nSaving file: ' + os.path.join(self.folder_model, filename))
            image = Image.fromarray(gen_image[0], 'RGB')
            image.save(os.path.join(self.folder_model, filename))

        self.printSummary('\nExecution ENDED')

    def execute_this_fn_run(self, progress_callback):
        self.progress_callback = progress_callback
        self.execute_run()

    def execute_run_thread(self):
        worker = Worker(self.execute_this_fn_run)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        self.threadpool.start(worker)


if __name__ == '__main__':
    settings = Settings()
    settings.read_settings()
    pix = Pix2Pix(settings.get_folder_source(), settings.get_folder_output())
    pix.execute_training()

