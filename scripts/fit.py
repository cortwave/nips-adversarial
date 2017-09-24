import logging
from os import environ
import warnings
from threading import Lock

from fire import Fire
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import h5py
from keras.metrics import categorical_crossentropy
from keras.models import load_model as _load_model
from keras.applications.inception_v3 import preprocess_input
from keras.utils
import numpy as np

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['CUDA_VISIBLE_DEVICES'] = ''

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

BATCH_SIZE = 64


def load_model(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _load_model(model)


def form_batch(X, y, batch_size):
    idx = np.random.randint(0, X.shape[0], int(batch_size))
    idx = sorted(idx.tolist())
    # when h5 dataset is not loaded into the memory, it accepts only lists as indices
    try:
        return X[idx], y[idx]
    except:
        logger.exception(f'Something went wrong with {idx}')
        raise ValueError()


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


def smooth(arr, temperature=2):
    arr = (np.log(arr) + 1e-6) / temperature
    return np.exp(arr) / np.sum(np.exp(arr), axis=1).reshape(-1, 1)


@threadsafe_generator
def batch_generator(x_data, x_data_adv, y_data, batch_size):
    while True:
        x_batch1, y_batch1 = form_batch(x_data, y_data, batch_size=batch_size / 2)
        x_batch2, y_batch2 = form_batch(x_data_adv, y_data, batch_size=batch_size / 2)

        x = preprocess_input(np.vstack((x_batch1, x_batch2)).astype('float32'))
        y = smooth(np.vstack((y_batch1, y_batch2)), temperature=2)
        logger.info('Batch is ready')
        yield x, y


def create_generator(dataset):
    dataset = h5py.File(dataset, 'r')

    x_data = dataset['x_data']
    x_data_adv = dataset['x_data_adv']
    y_data = dataset['y_data']

    logger.info('Dataset initialized')

    return batch_generator(x_data=x_data,
                           x_data_adv=x_data_adv,
                           y_data=y_data,
                           batch_size=BATCH_SIZE)


def main(network, dataset_train, dataset_test):
    model = load_model(network)

    gen = create_generator(dataset_train)
    val_gen = create_generator(dataset_test)

    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy)

    logger.info('Model compiled')

    model_checkpoint = ModelCheckpoint('./updated.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=4)
    callbacks = [model_checkpoint, es, reducer]

    model.fit_generator(generator=gen,
                        epochs=500,
                        steps_per_epoch=100,
                        validation_data=val_gen,
                        workers=4,
                        validation_steps=20,
                        callbacks=callbacks,
                        verbose=2,
                        use_multiprocessing=False
                        )


if __name__ == '__main__':
    Fire(main)
