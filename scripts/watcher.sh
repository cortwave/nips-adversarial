#!/usr/bin/env bash

until python make_adv_samples.py --raw_images_dir ../data/images --cache_dir ../data/images_bin; do
  echo 'time to restart'
done