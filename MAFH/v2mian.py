import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from training.v2MESDCH import  train
from loguru import logger

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    # mirflickr25k , coco2014 , nus wide , tc12
    datasets = ['nus wide']
    bits = [32]
    for ds in datasets:
        for bit in bits:
            train(ds, bit, batch_size=64, issave=True, max_epoch=150)

    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

