import logging
from glob import glob
from os import environ

import h5py
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.applications.resnet50 import preprocess_input as preprocess_input_rn
from keras.models import load_model
from keras import backend as K
from scipy.misc import imresize
import numpy as np

K.set_learning_phase(0)
environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


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
        while np.abs(adv - raw).max() < self.eps and i < self.eps * 3:
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


def make_adv_samples(raw_images_dir, cache_dir):
    processor = Attacker(eps=4)
    ensemble = Ensemble()

    images = glob(f'{raw_images_dir}/*')
    batch_size = 64
    for i in range(0, len(images), batch_size):
        names = images[i:i + batch_size]
        batch = np.array([img_to_array(load_img(name)) for name in names])

        try:
            targets = ensemble.predict(batch).astype('float32')
            processed = processor.do_gsa(batch, np.argmax(targets, axis=1)).astype('uint8')
        except:
            logger.exception('Something went wrong')
            continue

        file = h5py.File(f'{cache_dir}/{i/batch_size}.h5', 'w')

        x_data = file.create_dataset('x_data', shape=(batch_size, 299, 299, 3), dtype=np.uint8)
        y_data = file.create_dataset('y_data', shape=(batch_size, 1000), dtype=np.float32)

        x_data[...] = processed
        y_data[...] = targets

        file.close()
        logger.info('{} images processed'.format(i))

    logger.info('Done!')


if __name__ == '__main__':
    make_adv_samples(raw_images_dir='/home/arseny/imgnet/images',
                     cache_dir='../data/binimg')