import warnings


class DefaultConfig(object):
    load_img_path = None  
    load_txt_path = None

    # data parameters
    default_seed = 7
    training_size = 10500
    query_size = 2100
    database_size = 188321
    batch_size = 128
    tag_length = 1000
    label_length = 21
    dataset_name = 'nus wide'
    #D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/coco2014
    img_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/nus wide/'
    imgname_mat_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/nus wide/imgList21.mat'
    tag_mat_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/nus wide/tagList21.mat'
    label_mat_dir = 'D:/lixue/cross-modal/20220812DCMH/MESDCH-master/dataset/data/nus wide/labelList21.mat'

    # hyper-parameters
    max_epoch = 150
    alpha=1
    beta=1.4
    gamma = 0.1
    bit = 32
    y_dim=1000
    label_dim=21
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
