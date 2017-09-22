from h5py import File
from glob import glob
from fire import Fire
import numpy as np


def main(batch_dir, out_dir, batch_size=64):
    batches = glob(f'batch_dir/*.h5')

    file = File(f'{out_dir}/{pics)}.h5', 'w')

    x_data = file.create_dataset('x_data', shape=(batch_size, 299, 299, 3), dtype=np.uint8)
    y_data = file.create_dataset('y_data', shape=(batch_size, 1000), dtype=np.float32)



if __name__ == '__main__':
    Fire(main)


