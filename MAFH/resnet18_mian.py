import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from training.MESDCH import train
from loguru import logger

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    # mirflickr25k , coco2014 , nus wide , iapr tc-12
    datasets = ['coco2014']
    bits = [64]
    for ds in datasets:
        for bit in bits:
            train(ds, bit, batch_size=128, issave=True, max_epoch=150)

    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

