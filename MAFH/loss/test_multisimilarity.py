import torch
from utils.utils import calc_map_k
from utils.meters import AverageMeter
def multilabelsimilarityloss_KL1(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    KLloss = torch.sum(torch.mul(labelsSimilarity - hashrepresentationsSimilarity,
                                 torch.log((1e-5 + labelsSimilarity) / (1e-5 + hashrepresentationsSimilarity)))) / (
                     num_train * batch_size)
    # KLloss2 = torch.sum(torch.relu(labelsSimilarity - hashrepresentationsSimilarity)) / (num_train * batch_size)
    # KLloss3 = torch.sum(torch.relu(hashrepresentationsSimilarity - labelsSimilarity)) / (num_train * batch_size)
    # KLloss = KLloss1 + 0.5 * KLloss2 + 0.5 * KLloss3
    # print('KLloss1 = %4.4f, KLloss2 = %4.4f'%(KLloss1 , KLloss2))
    return KLloss


def multilabelsimilarityloss_KL(labels_batchsize, labels_train, hashrepresentations_batchsizes,
                                hashrepresentations__train):
    for index, hashrepresentations_batchsize in enumerate(hashrepresentations_batchsizes):
        batch_size = labels_batchsize.shape[0]
        num_train = labels_train.shape[0]
        labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
        labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
        hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
            torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
        hashrepresentations__train = hashrepresentations__train / torch.sqrt(
            torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
        labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
        hashrepresentationsSimilarity = torch.relu(
            torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
        KLloss = torch.sum(torch.mul(labelsSimilarity - hashrepresentationsSimilarity,
                                    torch.log((1e-5 + labelsSimilarity) / (1e-5 + hashrepresentationsSimilarity)))) / (
                        num_train * batch_size)
        KLloss += KLloss
    return KLloss/len(hashrepresentations_batchsizes)


def multilabelsimilarityloss_MSE(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    MSEloss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (num_train * batch_size)

    return MSEloss

def _loss_store_init(loss_store):
    """
    initialize loss store, transform list to dict by (loss name -> loss register)
    :param loss_store: the list with name of loss
    :return: the dict of loss store
    """

    dict_store = {}
    for loss_name in loss_store:
        dict_store[loss_name] = AverageMeter()
    loss_store = dict_store
    return loss_store
def remark_loss(*args):
    """
    store loss into loss store by order
    :param args: loss to store
    :return:
    """

    loss_store = ['intra loss','inter loss', 'quantization loss','loss']
    loss_store = _loss_store_init(loss_store)
    for i, loss_name in enumerate(loss_store.keys()):
        if isinstance(args[i], torch.Tensor):
            loss_store[loss_name].update(args[i].item())
        else:
            loss_store[loss_name].update(args[i])
    return loss_store
def reset_loss(loss_store=None):
    if loss_store is None:
        loss_store = loss_store
    for store in loss_store.values():
        store.reset()


def bit_scalable(img_model, txt_model, qB_img, qB_txt, rB_img, rB_txt, dataset, to_bit=[12,8]):#to_bit=[64, 32, 16]
    def get_rank(img_net, txt_net):
        w_img = img_net.weight.weight
        w_txt = txt_net.weight.weight
        # w_img = F.softmax(w_img, dim=0)
        # w_txt = F.softmax(w_txt, dim=0)
        w = torch.cat([w_img, w_txt], dim=0)
        w = torch.sum(w, dim=0)
        # _, ind = torch.sort(w)
        _, ind = torch.sort(w, descending=True)  # 临时降序
        return ind

    hash_length = qB_img.size(1)
    rank_index = get_rank(img_model, txt_model)
    dataset.query()
    query_label = dataset.get_all_label()
    dataset.retrieval()
    retrieval_label = dataset.get_all_label()

    def calc_map(ind):
        qB_img_ind = qB_img[:, ind]
        qB_txt_ind = qB_txt[:, ind]
        rB_img_ind = rB_img[:, ind]
        rB_txt_ind = rB_txt[:, ind]
        mAPi2t = calc_map_k(qB_img_ind, rB_txt_ind, query_label, retrieval_label)
        mAPt2i = calc_map_k(qB_txt_ind, rB_img_ind, query_label, retrieval_label)
        return mAPi2t, mAPt2i

    print("bit scalable from 128 bit:")
    for bit in to_bit:
        if bit >= hash_length:
            continue

        #bit_ind = rank_index[hash_length - bit: hash_length]
        bit_ind = rank_index[: bit]
        mAPi2t, mAPt2i = calc_map(bit_ind)
        print("%3d: i->t %4.4f| t->i %4.4f" % (bit, mAPi2t, mAPt2i))

