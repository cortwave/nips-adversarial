from io import BytesIO
from glob import glob
import logging
from sys import exit

from fire import Fire
import numpy as np
import requests
from PIL import Image
from joblib import Parallel, delayed


logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def get_crop_coords(width, height):
    new_width = np.clip(min(width, height), 0, 1000)
    new_height = new_width

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return left, top, right, bottom


def crop_image(img):
    img = img.crop(get_crop_coords(*img.size))
    return img.resize((299, 299))


def download(url, out_dir):
    image_name = url.split('/')[-1]
    files = glob(f'{out_dir}/*')
    if image_name in files:
        return

    try:
        logger.info(f'Getting {url}')
        r = requests.get(url)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            if 10 < np.array(img).mean() < 250:
                # too black or too white images are probably broken
                crop_image(img).save(f'{out_dir}/{image_name.lower()}')
            logger.info(f'{image_name} saved')
    except KeyboardInterrupt:
        exit()
    except:
        logger.exception(f'Can not download {url}')


def main(img_list, out_dir):
    with open(img_list) as lst:
        urls = [url[:-1] for url in lst.readlines()]
        np.random.shuffle(urls)
        Parallel(n_jobs=32, backend='threading')(delayed(download)(url, out_dir) for url in urls)


if __name__ == '__main__':
    Fire(main)
