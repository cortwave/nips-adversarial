from glob import glob
from os import environ, path
import logging

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from keras.metrics import categorical_crossentropy
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras import backend as K
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


class ImageProcessor:
    def __init__(self, eps):
        self.models = (load_model('./inception.h5'),
                       load_model('./inception_adv.h5'),
                       load_model('./xception.h5'))

        self.eps = eps

    def get_gradient_signs(self, original_array, target_idx=None):
        if target_idx is None:
            model = self.models[0]
            target_idx = model.predict(preprocess_input(np.copy(original_array))).argmax(axis=-1)
        else:
            model = np.random.choice(self.models)

        target = to_categorical(target_idx, 1000)
        target_variable = K.variable(target)
        loss = categorical_crossentropy(model.output, target_variable)
        gradients = K.gradients(loss, model.input)
        get_grad_values = K.function([model.input], gradients)
        grad_values = get_grad_values([preprocess_input(original_array)])[0]
        grad_signs = np.sign(grad_values)
        return grad_signs

    def do_gsa(self, raw, batch_targets):
        adv = np.copy(raw)
        original = np.copy(raw)
        img = preprocess_input(raw)

        neutralize = self.get_gradient_signs(np.copy(img), target_idx=None)
        adv += neutralize

        i = 0
        while np.abs(adv - original).max() < self.eps and i < self.eps * 2:
            i += 1
            grad = self.get_gradient_signs(img, batch_targets)
            adv -= grad
            adv = np.clip(adv, 0, 255)
            img = np.copy(adv)
        return adv


def main(input_dir, output_dir, max_epsilon):
    processor = ImageProcessor(eps=max_epsilon)

    images = [x for x in glob(input_dir + '/*') if not x.endswith('.csv')]
    metadata = pd.read_csv(path.join(input_dir, 'target_class.csv'), names=['ImageId', 'TargetClass'])
    targets = {k: (v - 1) for k, v in zip(metadata['ImageId'], metadata['TargetClass'])}

    batch_size = 50
    for i in range(0, len(images), batch_size):
        names = images[i:i + batch_size]
        batch = np.array([img_to_array(load_img(name)) for name in names])
        batch_targets = [targets[name.split('/')[-1]] for name in names]
        output_names = [path.join(output_dir, fname.split('/')[-1]) for fname in names]
        processed = processor.do_gsa(batch, batch_targets)

        for out_name, adv_img in zip(output_names, processed):
            array_to_img(adv_img).save(out_name)
        logger.info('{} images processed'.format(i))

    logger.info('Done!')


if __name__ == '__main__':
    Fire(main)
