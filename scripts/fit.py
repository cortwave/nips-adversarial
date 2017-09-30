import logging
from os import environ
from functools import partial
import warnings
from threading import Lock

from fire import Fire
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
import h5py
from keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy
from keras.models import load_model as _load_model
from keras.applications.xception import preprocess_input as preprocess_xcept
from keras import backend as K
import numpy as np
from scipy.misc import imresize

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# environ['CUDA_VISIBLE_DEVICES'] = ''

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def preprocess_input_densenet(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017  # scale values
    return x


def load_model(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _load_model(model)


def form_batch(X, y, batch_size):
    idx = np.random.choice(np.arange(X.shape[0]), int(batch_size), replace=False)
    idx = sorted(idx.tolist())
    # when h5 dataset is not loaded into the memory, it accepts only lists as indices
    return X[idx], y[idx]


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def smooth(arr, temperature=10):
    arr = (np.log(arr + 1e-6)) / temperature
    arr = np.exp(arr) / np.sum(np.exp(arr), axis=1).reshape(-1, 1)

    return arr


@threadsafe_generator
def batch_generator(x_data, x_data_adv, y_data, batch_size, preprocess, resize):
    while True:
        x_batch1, y_batch1 = form_batch(x_data, y_data, batch_size=batch_size / 2)
        x_batch2, y_batch2 = form_batch(x_data_adv, y_data, batch_size=batch_size / 2)

        x = preprocess(resize(np.vstack((x_batch1, x_batch2))).astype('float32'))
        y = smooth(np.vstack((y_batch1, y_batch2)), temperature=2)
        yield x, y


def create_generator(dataset, batch_size, preprocess, resize):
    dataset = h5py.File(dataset, 'r')

    x_data = dataset['x_data']
    x_data_adv = dataset['x_data_adv']
    y_data = dataset['y_data']

    logger.info('Dataset initialized')

    return batch_generator(x_data=x_data,
                           x_data_adv=x_data_adv,
                           y_data=y_data,
                           batch_size=batch_size,
                           preprocess=preprocess,
                           resize=resize)


def adjust_size(batch, shape):
    if batch.shape[1] != shape[0]:
        images = np.array([imresize(img, shape) for img in batch])
        return images
    return batch


def get_config(name):
    if name == 'inception':
        preprocess = preprocess_xcept
        network = load_model('./inception.h5')
        resize = partial(adjust_size, shape=(299, 299, 3))
    elif name == 'xception':
        preprocess = preprocess_xcept
        network = load_model('./xception.h5')
        resize = partial(adjust_size, shape=(299, 299, 3))
    elif name == 'resnet':
        preprocess = preprocess_xcept
        network = load_model('./resnet.h5')
        resize = partial(adjust_size, shape=(224, 224, 3))
    elif name == 'densenet':
        preprocess = preprocess_xcept
        network = load_model('./densenet.h5')
        resize = partial(adjust_size, shape=(224, 224, 3))
    else:
        raise ValueError(f'{name} is not recognized')

    return preprocess, network, resize


def main(network,
         dataset_train='../data/new_test.h5',
         dataset_test='../data/new_test.h5',
         batch_size=32):
    preprocess, model, resize = get_config(network)

    gen = create_generator(dataset_train, batch_size, preprocess=preprocess, resize=resize)
    val_gen = create_generator(dataset_test, batch_size, preprocess=preprocess, resize=resize)

    model.compile(optimizer=SGD(momentum=0.9, nesterov=True),
                  loss=categorical_crossentropy,
                  metrics=[top_k_categorical_accuracy, categorical_accuracy])

    logger.info('Model compiled')

    model_checkpoint = ModelCheckpoint(f'./{network}_adv.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-5, verbose=1, factor=0.1, patience=4)
    callbacks = [model_checkpoint, es, reducer]

    model.fit_generator(generator=gen,
                        epochs=500,
                        steps_per_epoch=100,
                        validation_data=val_gen,
                        workers=4,
                        validation_steps=20,
                        callbacks=callbacks,
                        verbose=1,
                        use_multiprocessing=False
                        )


if __name__ == '__main__':
    Fire(main)
