from glob import glob
import logging

from h5py import File
from fire import Fire
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def unite(batches, out_dir, out_name, batch_size):
    num_batches = len(batches)

    file = File(f'{out_dir}/{out_name}.h5', 'w')

    x_data = file.create_dataset('x_data', shape=(batch_size * num_batches, 299, 299, 3), dtype=np.uint8)
    x_data_adv = file.create_dataset('x_data_adv', shape=(batch_size * num_batches, 299, 299, 3), dtype=np.uint8)
    y_data = file.create_dataset('y_data', shape=(batch_size * num_batches, 1000), dtype=np.float32)

    for i, batch in enumerate(batches):
        batch = File(batch, 'r')

        batch_x, batch_x_adv, batch_y = map(lambda x: batch[x], ('x_data', 'x_data_adv', 'y_data'))

        offset = i * batch_size
        x_data[offset: offset + batch_size] = batch_x
        x_data_adv[offset: offset + batch_size] = batch_x_adv
        y_data[offset: offset + batch_size] = batch_y

        batch.close()
        logger.info(f'{i+1}/{num_batches} batches for {out_name} repacked')

    file.close()


def main(batch_dir, out_dir, batch_size=64):
    batches = glob(f'{batch_dir}/*.h5')
    logger.info(f'There are {len(batches)} batches')
    train_batches, test_batches = train_test_split(batches, test_size=.2)

    Parallel(n_jobs=2)(delayed(unite)(batches_list, out_dir, name, batch_size)
                       for batches_list, name in ((train_batches, 'train'),
                                                  (test_batches, 'test')))


if __name__ == '__main__':
    Fire(main)
