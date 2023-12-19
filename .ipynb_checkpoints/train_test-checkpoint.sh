#!/bin/bash

for i in {1..5}
do
  python3 train_semi.py TRAIN TEST --subset 'experimentset'

  python -m val

done