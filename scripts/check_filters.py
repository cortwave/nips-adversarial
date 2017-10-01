import logging
from functools import partial
from os import path

from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_wavelet
from skimage.restoration import denoise_nl_means
from skimage.filters import median, gaussian
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from joblib import Parallel, delayed
import numpy as np
from h5py import File
from fire import Fire

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def apply_filter(batch, filter_func):
    data = Parallel(-1)(delayed(filter_func)(img.astype('uint8')) for img in batch)
    return (np.array(data) * 255).astype('uint8')


def filter_median(img):
    res = []
    for c in range(img.shape[-1]):
        res.append(median(img[..., c]))
    return np.stack(res, axis=2)


def nofilter(x):
    return x


def main(data_dir='../data', scripts_dir='./', use_adv=True):
    file = File(path.join(data_dir, 'sample.h5'))
    x_data = file['x_data_adv'][:] if int(use_adv) else file['x_data'][:]
    y_data = file['y_data'][:]

    inc = InceptionV3()
    inc.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    xc = Xception()
    xc.compile(loss='categorical_crossentropy',
               optimizer='sgd',
               metrics=['accuracy'])

    inc_adv = load_model(path.join(scripts_dir, 'inception_adv.h5'))
    inc_adv.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
    xc_adv = load_model(path.join(scripts_dir, 'xception_adv.h5'))
    xc_adv.compile(loss='categorical_crossentropy',
                   optimizer='sgd',
                   metrics=['accuracy'])

    for net_name, net in (('inception', inc),
                          ('inception_adv', inc_adv),
                          ('xception', xc),
                          ('xception_adv', xc_adv)
                          ):
        for f in (filter_median,
                  partial(denoise_nl_means, multichannel=True),
                  partial(denoise_wavelet, multichannel=True),
                  partial(denoise_bilateral, multichannel=True),
                  partial(denoise_tv_chambolle, multichannel=True),
                  partial(gaussian, multichannel=True),
                  nofilter):
            try:
                imgs = apply_filter(x_data, f)
                imgs = preprocess_input(imgs.astype('float32'))
                loss, acc = net.evaluate(imgs, y_data, batch_size=64, verbose=0)
                fname = f.func.__name__ if hasattr(f, 'func') else f.__name__
                logger.info(f'{net_name} with {fname}: loss {loss:.3f}, accuracy {acc:.3f}')
            except:
                logger.exception(f'{f} failed')


if __name__ == '__main__':
    Fire(main())
