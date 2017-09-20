from glob import glob
from os import environ
import logging

from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.xception import preprocess_input as preprocess_xcept
from keras.preprocessing import image
import numpy as np
from fire import Fire

environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

logger.info('Start!')


def main(input_dir, output_file):
    files = ('./resnet.h5', './vgg.h5', 'xception.h5')
    networks = [load_model(f) for f in files]

    images_224, images_299, names = [], [], []
    for fname in glob(input_dir + '/*'):
        img224 = image.load_img(fname, target_size=(224, 224))
        img299 = image.load_img(fname, target_size=(299, 299))
        img224, img299 = map(lambda img: image.img_to_array(img),
                             (img224, img299))

        images_224.append(img224)
        images_299.append(img299)
        names.append(fname.split('/')[-1])

    images_224 = preprocess_input(np.array(images_224).astype('float32'))
    images_299 = preprocess_xcept(np.array(images_299).astype('float32'))

    result = np.zeros((len(names), 1000))
    for i, net in enumerate(networks):
        logger.info('{}'.format(i))
        if i > 1:
            pred = net.predict(images_299, batch_size=50)
        else:
            pred = net.predict(images_224, batch_size=50)
        result += pred

    with open(output_file, 'w') as out:
        for i in range(result.shape[0]):
            out.write('{},{}\n'.format(names[i], np.argmax(result[i]) + 1))

    logger.info('Done!')


if __name__ == '__main__':
    Fire(main)
