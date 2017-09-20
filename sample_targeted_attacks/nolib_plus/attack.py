from glob import glob
from os import environ, path
import logging

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from keras.metrics import categorical_crossentropy
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras import backend as K
from scipy.misc import imresize
import numpy as np
import pandas as pd
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


def trick(img, model, obj_to_fake=None):
    true_label = np.argmax(model.predict(img))

    if obj_to_fake is None:
        obj_to_fake = true_label
        while true_label == obj_to_fake:
            obj_to_fake = np.random.randint(0, 999)

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output
    cost_function = model_output_layer[0, obj_to_fake]

    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])

    max_change_above = img * 1.01
    max_change_below = img * 0.99

    hacked_image = np.copy(img)
    base_lr = .5
    i = 0

    while i <= 5000:
        i += 1
        if not i % 10:
            is_success = describe(hacked_image, model, i, obj_to_fake, true_label)
            if is_success:
                return hacked_image, true_label

        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
        lr = base_lr * (1000 - i / 4) / 1000
        hacked_image += gradients * lr

        # ToDo: clip on L0 norm
        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

    return hacked_image, true_label


class Metamodel:
    def __init__(self, model, preprocess, need_resize):
        self.model = model
        self.preprocess = preprocess
        self.need_resize = need_resize


class ImageProcessor:
    def __init__(self, eps):
        self.model = Metamodel(model=load_model('./inception.h5'),
                               preprocess=preprocess_input,
                               need_resize=False)
        self.eps = eps

    def do_gsa(self, raw, batch_targets):
        img = preprocess_input(raw)
        adv = np.copy(raw)

        i = 0
        while np.abs(adv - raw).max() < self.eps and i < self.eps * 2:
            i += 1
            grad = self.get_gradient_signs(img, batch_targets)
            adv += grad
            adv = np.clip(adv, 0, 255)
            img = preprocess_input(np.copy(adv))
        return adv

    def get_gradient_signs(self, original_array, target_idx):
        model = self.model.model
        target = to_categorical(target_idx, 1000)
        target_variable = K.variable(target)
        loss = categorical_crossentropy(model.output, target_variable)
        gradients = K.gradients(loss, model.input)
        get_grad_values = K.function([model.input], gradients)
        grad_values = get_grad_values([original_array])[0]
        grad_signs = np.sign(grad_values)
        return grad_signs


def main(input_dir, output_dir, max_epsilon):
    processor = ImageProcessor(eps=max_epsilon)

    images = [x for x in glob(input_dir + '/*') if not x.endswith('.csv')]
    metadata = pd.read_csv(path.join(input_dir, 'target_class.csv'), names=['ImageId', 'TargetClass'])
    targets = {k: (v - 1) for k, v in zip(metadata['ImageId'], metadata['TargetClass'])}

    batch_size = 50
    for i in range(0, len(images), batch_size):
        names = images[i:i + batch_size]
        batch = np.array([img_to_array(load_img(name)) for name in names])
        batch_targets = [targets[name.split('.')[0].split('/')[-1]] for name in names]
        output_names = [path.join(output_dir, fname.split('/')[-1]) for fname in names]
        processed = processor.do_gsa(batch, batch_targets)

        for out_name, adv_img in zip(output_names, processed):
            array_to_img(adv_img).save(out_name)
        logger.info('{} images processed'.format(i))

    logger.info('Done!')


if __name__ == '__main__':
    Fire(main)
