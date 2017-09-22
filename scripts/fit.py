import logging
from os import environ
import warnings
from threading import Lock

from sklearn.model_selection import KFold
from fire import Fire
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import h5py
from keras.metrics import categorical_crossentropy
from keras.models import load_model as _load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np

environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
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


def smooth(arr, temperature=2):
    arr = np.log(arr) / temperature
    return np.exp(arr) / np.sum(np.exp(arr), axis=1)


@threadsafe_generator
def batch_generator(x_data, x_data_adv, y_data, batch_size):
    while True:
        x_batch1, y_batch1 = form_batch(x_data, y_data, batch_size=batch_size)
        x_batch2, y_batch2 = form_batch(x_data_adv, y_data, batch_size=batch_size)

        x = preprocess_input(np.vstack((x_batch1, x_batch2)))
        y = smooth(np.vstack((y_batch1, y_batch2)), temperature=2)
        yield x, y


def main(network, dataset):
    model = load_model(network)
    dataset = h5py.File(dataset, 'r')

    x_data = dataset['x_data']
    x_data_adv = dataset['x_data_adv']
    y_data = dataset['y_data']

    folder = KFold(n_splits=5, shuffle=True)
    train_index, test_index = map(lambda x: sorted(x.tolist()), next(folder.split(x_data)))

    gen = batch_generator(x_data=x_data[train_index],
                          x_data_adv=x_data_adv[train_index],
                          y_data=y_data[train_index],
                          batch_size=BATCH_SIZE)

    val_gen = batch_generator(x_data=x_data[test_index],
                              x_data_adv=x_data_adv[test_index],
                              y_data=y_data[test_index],
                              batch_size=BATCH_SIZE)

    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy)

    model_checkpoint = ModelCheckpoint('./updated.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=4)
    callbacks = [model_checkpoint, es, reducer]

    model.fit_generator(generator=gen,
                        epochs=500,
                        steps_per_epoch=500,
                        validation_data=val_gen,
                        workers=4,
                        validation_steps=100,
                        callbacks=callbacks,
                        )


if __name__ == '__main__':
    Fire(main)
