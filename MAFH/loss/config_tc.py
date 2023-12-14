import warnings


class DefaultConfig(object):
    load_img_path = None  
    load_txt_path = None

    # data parameters
    default_seed = 8
    training_size = 10000
    query_size = 2000
    database_size = 17999
    batch_size = 128
    tag_length = 1251
    label_length = 275
    dataset_name = 'iapr tc-12'
    #D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/coco2014
    img_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/iapr tc-12/'
    imgname_mat_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/iapr tc-12/imgList.mat'
    tag_mat_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/iapr tc-12/tagList.mat'
    label_mat_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/iapr tc-12/labelList.mat'

    # hyper-parameters
    max_epoch = 150
    alpha=1
    beta=1.4
    gamma = 0.1
    bit = 64
    y_dim=1251
    label_dim=275
    lr = 10 ** (-1.5)  

    use_gpu = True
    valid = True


    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
