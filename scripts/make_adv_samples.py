import logging
from glob import glob
from os import environ
from sys import exit
from time import time
import warnings

from fire import Fire
import h5py
import joblib as jl
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.applications.resnet50 import preprocess_input as preprocess_input_rn
from keras.models import load_model as _load_model
from keras import backend as K
from scipy.misc import imresize
import numpy as np

K.set_learning_phase(0)
environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def load_model(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _load_model(model)


def adjust_size(batch, shape):
    if batch.shape[1] != shape[0]:
        images = np.array([imresize(img, shape) for img in batch])
        return images
    return batch


class Metamodel:
    def __init__(self, model, preprocess, need_resize):
        self.model = model
        self.preprocess = preprocess
        self.need_resize = need_resize


class Attacker:
    def __init__(self, eps):
        self.model1 = Metamodel(model=load_model('./inception.h5'),
                                preprocess=preprocess_input_inc,
                                need_resize=False)
        self.model2 = Metamodel(model=load_model('./resnet.h5'),
                                preprocess=preprocess_input_rn,
                                need_resize=True)
        self.eps = eps

    def do_gsa(self, raw, target_idx):
        adv = np.copy(raw)

        i = 0
        if self.eps is None:
            eps = np.random.randint(4, 12)
        else:
            eps = self.eps

        while np.abs(adv - raw).max() < eps and i < eps * 3:
            i += 1
            grad = self.get_gradient_signs(np.copy(adv), target_idx)
            adv += grad
            adv = np.clip(adv, 0, 255)
        return adv

    def get_gradient_signs(self, raw, target_idx):
        metamodel = np.random.choice((self.model1, self.model2))

        raw = adjust_size(raw, (224, 224, 3)) if metamodel.need_resize else raw

        original_array = metamodel.preprocess(raw.astype('float32'))
        model = metamodel.model
        target = to_categorical(target_idx, 1000)
        target_variable = K.variable(target)
        loss = categorical_crossentropy(model.output, target_variable)
        gradients = K.gradients(loss, model.input)
        get_grad_values = K.function([model.input], gradients)
        grad_values = get_grad_values([original_array])[0]
        grad_values = adjust_size(grad_values, (299, 299, 3))

        grad_signs = np.sign(grad_values)
        return grad_signs


class Ensemble:
    def __init__(self):
        self.models = (Metamodel(model=load_model('./inception.h5'),
                                 preprocess=preprocess_input_inc,
                                 need_resize=False),
                       Metamodel(model=load_model('./xception.h5'),
                                 preprocess=preprocess_input_inc,
                                 need_resize=False),
                       Metamodel(model=load_model('./resnet.h5'),
                                 preprocess=preprocess_input_rn,
                                 need_resize=True))

    def predict(self, raw):
        result = []
        for m in self.models:
            if m.need_resize:
                img = adjust_size(np.copy(raw), (224, 224, 3))
            else:
                img = np.copy(raw)
            img = m.preprocess(img.astype('float32'))
            result.append(m.model.predict(img))
        return np.mean(result, axis=0)


def make_adv_samples(raw_images_dir, cache_dir, cuda_device=None):
    if cuda_device:
        environ['CUDA_VISIBLE_DEVICES'] = f'{cuda_device}'

    processor = Attacker(eps=None)
    ensemble = Ensemble()

    queue_path = '/tmp/img_list.h5'
    try:
        queue = jl.load(queue_path)
    except FileNotFoundError:
        logger.info('Making images list from scratch')
        queue = glob(f'{raw_images_dir}/*.jpg') + glob(f'{raw_images_dir}/*.png')
        jl.dump(queue, queue_path)

    batch_size = 64

    counter = 0
    while len(queue) >= batch_size:
        counter += 1
        if counter >= 3:
            exit(3)

        logger.info(f'There are {len(queue)} images left')
        t = time()
        names = queue[:batch_size]
        queue = queue[batch_size:]
        jl.dump(queue, queue_path)

        batch = []
        for name in names:
            try:
                img = load_img(name)
            except OSError:
                logger.info(f'image {name} is broken')
                img = None
                for j in range(len(names)):
                    try:
                        img = load_img(names[j])
                        logger.info(f'using {names[j]} instead')
                        continue
                    except:
                        pass

            batch.append(img_to_array(img))
        batch = np.array(batch)

        try:
            targets = ensemble.predict(batch).astype('float32')
            processed = processor.do_gsa(batch, np.argmax(targets, axis=1)).astype('uint8')
        except:
            logger.exception('Something went wrong')
            continue

        file = h5py.File(f'{cache_dir}/{int(time())}.h5', 'w')

        x_data = file.create_dataset('x_data', shape=(batch_size, 299, 299, 3), dtype=np.uint8)
        x_data_adv = file.create_dataset('x_data_adv', shape=(batch_size, 299, 299, 3), dtype=np.uint8)
        y_data = file.create_dataset('y_data', shape=(batch_size, 1000), dtype=np.float32)

        x_data[...] = batch.astype('uint8')
        x_data_adv[...] = processed
        y_data[...] = targets

        file.close()
        logger.info('{} batches processed during {:.2f}'.format(counter, time() - t))

    logger.info('Done!')


if __name__ == '__main__':
    Fire(make_adv_samples)
