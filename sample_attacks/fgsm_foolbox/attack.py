from glob import glob
from os import environ, path
import logging

import foolbox
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import numpy as np
from fire import Fire

from keras import backend as K
K.set_learning_phase(0)

environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

logger.info('Start!')


def postprocess(x):
    x /= 2
    x += .5
    x *= 255.
    x = np.clip(x, 0, 255).astype('int16')
    return x


def clip_epsilon(raw, adversarial, eps):
    diff = np.clip(raw - adversarial, -eps, eps)
    adversarial = np.clip(raw - diff, 0, 255)
    return adversarial


def get_img(fname):
    raw = load_img(fname, target_size=(299, 299))
    raw = img_to_array(raw).astype('float16')
    img = preprocess_input(np.copy(raw))
    return raw, img


def describe(model, input_image, verbose=False):
    # Run the image through the neural network
    input_image = np.expand_dims(input_image, 0)
    predictions = model.predict(input_image)

    # Convert the predictions into text and print them
    if verbose:
        predicted_classes = decode_predictions(predictions, top=1)
        imagenet_id, name, confidence = predicted_classes[0][0]
        print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
    return np.argmax(predictions)


def process_image(kmodel, fmodel, img_path, eps):
    raw, img = get_img(img_path)
    label = describe(kmodel, img)

    attack = foolbox.attacks.FGSM(fmodel, criterion=foolbox.criteria.Misclassification())
    adversarial = attack(img, label)

    if adversarial is not None:
        adversarial = postprocess(adversarial)
        return clip_epsilon(raw, adversarial, eps=eps)
    return raw


def main(input_dir, output_dir, max_epsilon):
    kmodel = load_model('./inception.h5')
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(-1, 1), preprocessing=(0, 1))

    for i, fname in enumerate(glob(input_dir + '/*')):
        hacked = process_image(kmodel=kmodel, fmodel=fmodel, eps=max_epsilon, img_path=fname)
        out_name = path.join(output_dir, fname.split('/')[-1])
        array_to_img(hacked).save(out_name)
        if not i % 100:
            logger.info('{} images processed'.format(i))

    logger.info('Done!')


if __name__ == '__main__':
    Fire(main)
