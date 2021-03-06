from glob import glob
from functools import partial
from os import environ, cpu_count
import logging
from multiprocessing.dummy import Pool

from keras.applications.xception import preprocess_input as preprocess_xcept
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
from fire import Fire

environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)

logger.info('Start!')


def apply_filter(batch, filter_func):
    data = Pool(cpu_count()).map(filter_func, batch.astype('uint8'))
    return (np.array(data) * 255).astype('float32')


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

    return names, np.array(images_resnet), np.array(images_xc)


def main(input_dir, output_file):
    files = ('./resnet_adv.h5', './xception_adv.h5')
    networks = [load_model(f) for f in files]

    fnames = glob(input_dir + '/*')
    final = []

    for batch_num, batch in enumerate(np.array_split(fnames, 10)):
        logger.info('Working with batch {}'.format(batch_num))

        names, images_resnet, images_xc = process_batch(batch)

        pred = np.zeros((len(names), 1000))
        for filter_func in (partial(denoise_tv_chambolle, multichannel=True),
                            partial(gaussian, multichannel=True),
                            partial(denoise_bilateral, multichannel=True)):
            images_resnet_filtered = apply_filter(images_resnet, filter_func)
            images_xc_filtered = apply_filter(images_xc, filter_func)
            for i, net in enumerate(networks):
                if i == 0:
                    processed = preprocess_xcept(images_resnet_filtered.astype('float32'))
                    # xception-style preprocessing was used due to bug in adversarial training
                    # see scripts/fit.py
                    pred += net.predict(processed, batch_size=50)
                elif i == 1:
                    processed = preprocess_xcept(images_xc_filtered.astype('float32'))
                    pred += net.predict(processed, batch_size=50)

        for i in range(pred.shape[0]):
            final.append('{},{}\n'.format(names[i], np.argmax(pred[i]) + 1))

    with open(output_file, 'w') as out:
        for line in final:
            out.write(line)

    logger.info('Done!')


if __name__ == '__main__':
    Fire(main)
