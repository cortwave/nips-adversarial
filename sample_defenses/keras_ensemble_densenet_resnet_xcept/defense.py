from glob import glob
from os import environ
import logging

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.xception import preprocess_input as preprocess_xcept
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import numpy as np
from fire import Fire

environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

logger.info('Start!')


def preprocess_input_densenet(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017  # scale values
    return x


def process_batch(fnames):
    images_resnet, images_xc, names = [], [], []
    for fname in fnames:
        img224 = image.load_img(fname, target_size=(224, 224))
        img299 = image.load_img(fname, target_size=(299, 299))
        img224, img299 = map(lambda img: image.img_to_array(img),
                             (img224, img299))

        images_resnet.append(img224)
        images_xc.append(img299)
        names.append(fname.split('/')[-1])

    images_dense = preprocess_input_densenet(np.array(images_resnet).astype('float32'))
    images_resnet = preprocess_input(np.array(images_resnet).astype('float32'))
    images_xc = preprocess_xcept(np.array(images_xc).astype('float32'))

    return names, images_dense, images_resnet, images_xc


def main(input_dir, output_file):
    files = ('./resnet.h5', './densenet161.h5', './xception.h5')
    networks = [load_model(f) for f in files]

    fnames = glob(input_dir + '/*')
    final = []

    for batch_num, batch in enumerate(np.array_split(fnames, 10)):
        logger.info('Working with batch {}'.format('batch_num'))

        names, images_dense, images_resnet, images_xc = process_batch(batch)

        pred = np.zeros((len(names), 1000))
        for i, net in enumerate(networks):
            if i == 0:
                pred += net.predict(images_resnet, batch_size=50)
            elif i == 1:
                pred += net.predict(images_dense, batch_size=50)
            else:
                pred += net.predict(images_xc, batch_size=50)

        for i in range(pred.shape[0]):
            final.append('{},{}\n'.format(names[i], np.argmax(pred[i]) + 1))

    with open(output_file, 'w') as out:
        for line in final:
            out.write(line)

    logger.info('Done!')


if __name__ == '__main__':
    Fire(main)
