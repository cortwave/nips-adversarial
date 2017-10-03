from glob import glob
from os import environ, path
import logging

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from keras.metrics import categorical_crossentropy
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.applications.resnet50 import preprocess_input as preprocess_input_rn
from keras.models import load_model
from keras import backend as K
from scipy.misc import imresize
import numpy as np
from fire import Fire

K.set_learning_phase(0)

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

logger.info('Start!')


def postprocess_inc(x):
    x /= 2
    x += .5
    x *= 255.
    x = np.clip(x, 0, 255)
    return x


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


class ImageProcessor:
    def __init__(self, eps):
        self.model1 = Metamodel(model=load_model('./inception.h5'),
                                preprocess=preprocess_input_inc,
                                need_resize=False)
        self.model2 = Metamodel(model=load_model('./resnet.h5'),
                                preprocess=preprocess_input_rn,
                                need_resize=True)
        self.model3 = Metamodel(model=load_model('./xception.h5'),
                                preprocess=preprocess_input_inc,
                                need_resize=False)

        self.eps = eps

    def do_gsa(self, raw):
        adv = np.copy(raw)

        i = 0
        while np.abs(adv - raw).max() < self.eps and i < self.eps * 3:
            i += 1
            grad = self.get_gradient_signs(np.copy(adv))
            adv += grad
            adv = np.clip(adv, 0, 255)
        return adv

    def get_gradient_signs(self, raw):
        metamodel = np.random.choice((self.model1, self.model2, self.model3))

        raw = adjust_size(raw, (224, 224, 3)) if metamodel.need_resize else raw

        original_array = metamodel.preprocess(raw.astype('float32'))
        model = metamodel.model
        target_idx = model.predict(original_array).argmax(axis=1)
        target = to_categorical(target_idx, 1000)
        target_variable = K.variable(target)
        loss = categorical_crossentropy(model.output, target_variable)
        gradients = K.gradients(loss, model.input)
        get_grad_values = K.function([model.input], gradients)
        grad_values = get_grad_values([original_array])[0]
        grad_values = adjust_size(grad_values, (299, 299, 3))

        grad_signs = np.sign(grad_values)
        return grad_signs


def main(input_dir, output_dir, max_epsilon):
    processor = ImageProcessor(eps=max_epsilon)

    images = glob(input_dir + '/*')
    batch_size = 50
    for i in range(0, len(images), batch_size):
        names = images[i:i + batch_size]
        batch = np.array([img_to_array(load_img(name)) for name in names])
        output_names = [path.join(output_dir, fname.split('/')[-1]) for fname in names]
        processed = processor.do_gsa(batch)

        for out_name, adv_img in zip(output_names, processed):
            array_to_img(adv_img).save(out_name)
        logger.info('{} images processed'.format(i))

    logger.info('Done!')

if __name__ == '__main__':
    Fire(main)
