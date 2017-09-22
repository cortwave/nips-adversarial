from glob import glob
import logging

from h5py import File
from fire import Fire
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__file__)


def main(batch_dir, out_dir, batch_size=64):
    batches = glob(f'{batch_dir}/*.h5')
    num_batches = len(batches)

    file = File(f'{out_dir}/{batch_dir.split("/")[0]}.h5', 'w')

    x_data = file.create_dataset('x_data', shape=(batch_size * num_batches, 299, 299, 3), dtype=np.uint8,
                                 compression="gzip")
    x_data_adv = file.create_dataset('x_data_adv', shape=(batch_size * num_batches, 299, 299, 3), dtype=np.uint8,
                                     compression="gzip")
    y_data = file.create_dataset('y_data', shape=(batch_size * num_batches, 1000), dtype=np.float32,
                                 compression="gzip")

    for i, batch in enumerate(batches):
        batch = File(batch, 'r')

        batch_x, batch_x_adv, batch_y = map(lambda x: batch[x], ('x_data', 'x_data_adv', 'y_data'))

        offset = i * batch_size
        x_data[offset: offset + batch_size] = batch_x
        x_data_adv[offset: offset + batch_size] = batch_x_adv
        y_data[offset: offset + batch_size] = batch_y

        batch.close()
        logger.info(f'{i+1}/{num_batches} batches repacked')

    file.close()


if __name__ == '__main__':
    Fire(main)
