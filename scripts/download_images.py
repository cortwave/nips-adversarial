from io import BytesIO
from glob import glob
import logging

import numpy as np
import requests
from PIL import Image
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def download(url):
    image_name = url.split('/')[-1]
    files = glob('images/*')
    if image_name in files:
        return

    try:
        logger.info(f'Getting {url}')
        r = requests.get(url)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            if 10 < np.array(img).mean() < 250:
                # too black or too white images are probably broken
                img.resize((299, 299)).save(f'images/{image_name}')
            logger.info(f'{image_name} saved')

    except:
        logger.exception(f'Can not download {url}')


if __name__ == '__main__'
    with open('img.txt') as lst:
        urls = np.random.shuffle([url[:-1] for url in lst.readlines()])
        Parallel(n_jobs=32, backend='threading')(delayed(download)(url) for url in urls)
