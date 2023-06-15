import numpy as np
from torch import nn
from utils.utils import calc_map_k
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def to_cuda(*args):
    """
    chagne all tensor from cpu tensor to cuda tensor
    :param args: tensors or models
    :return: the tensors or models in cuda by input order
    """
    cuda_args = []
    for arg in args:
        cuda_args.append(arg.cuda())
    return cuda_args

def get_codes(img_model, txt_model, dataset, bit, batch_size, cuda=True):
    dataset.both_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    #dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True, pin_memory=True)
    img_buffer = txt_buffer = torch.empty(len(dataset), bit, dtype=torch.float)
    if cuda:
        img_buffer, txt_buffer = to_cuda(img_buffer, txt_buffer)
    img_model.eval()
    txt_model.eval()
    for data in tqdm(dataloader):
        index = data['index'].numpy()
        img = data['img']
        txt = data['txt']
        if cuda:
            img, txt = to_cuda(img, txt)
        img_hash = torch.tanh(img_model(img))
        txt_hash = torch.tanh(txt_model(txt))
        img_buffer[index, :] = img_hash.data
        txt_buffer[index, :] = txt_hash.data
    return img_buffer, txt_buffer

def calc_hammingDist1(B1, B2):
    q = B2.shape[1]  # length of bit, e.g. 64
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))  # 返回18015个数，返回值越小，表示相似性越大
    return distH

def calc_map_k1(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist1(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map

def valid_calc(use_gpu,dataset,batch_size, bit,  img_model, txt_model, return_hash=True,cuda=True):
    """
    get valid data hash code and calculate mAP
    :param img_model: the image model
    :param txt_model: the txt model
    :param dataset: the valid dataset
    :param bit: the length of hash code
    :param batch_size: the batch size of valid
    :param drop_integer: if true, the excrescent data will be drop
    :param return_hash: if true, the hash codes will be returned
    :param cuda: if use cuda
    :return: mAP and hash codes(if return_hash = True)
    """
    # get query img and txt binary code
    dataset.query()
    qB_img, qB_txt = get_codes(img_model, txt_model, dataset, bit, batch_size, cuda=cuda)
    #qB_img = get_img_code(batch_size, bit, use_gpu, img_model,  dataset)
    #qB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    query_label = dataset.get_all_label()
    # get retrieval img and txt binary code
    dataset.retrieval()
    rB_img, rB_txt = get_codes(img_model, txt_model, dataset, bit, batch_size, cuda=cuda)
    #rB_img = get_img_code(batch_size, bit, use_gpu, img_model, dataset)
    #rB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    retrieval_label = dataset.get_all_label()
    mAPi2t = calc_map_k1(qB_img, rB_txt, query_label, retrieval_label)
    mAPt2i = calc_map_k1(qB_txt, rB_img, query_label, retrieval_label)
    if return_hash:
        return qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu(), query_label, retrieval_label
    return mAPi2t, mAPt2i

def valid(batch_size, bit, use_gpu, img_model: nn.Module, txt_model: nn.Module, dataset, return_hash=True):
    # get query img and txt binary code
    dataset.query()
    qB_img = get_img_code(batch_size, bit, use_gpu, img_model,  dataset)
    qB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    query_label = dataset.get_all_label()
    # get retrieval img and txt binary code
    dataset.retrieval()
    rB_img = get_img_code(batch_size, bit, use_gpu, img_model, dataset)
    rB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    retrieval_label = dataset.get_all_label()
    mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
    mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)

    if return_hash:
        return qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu(), query_label, retrieval_label
    return mAPi2t, mAPt2i

################################################################
def pr_curve1(query_hash, retrieval_hash, query_label, retrieval_label):
    #trn_binary = trn_binary.numpy()
    retrieval_hash = np.asarray(retrieval_hash, np.int32)
    retrieval_label = retrieval_label.numpy()
    query_hash = np.asarray(query_hash, np.int32)
    query_label = query_label.numpy()
    query_times = query_hash.shape[0]
    trainset_len = retrieval_hash.shape[0]
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)

    sum_p = np.zeros(trainset_len)
    sum_r = np.zeros(trainset_len)

    for i in range(query_times):
        #print('Query ', i+1)
        query_labal = query_label[i]
        query_binary = query_hash[i,:]
        query_result = np.count_nonzero(query_binary != retrieval_hash, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        #print(sort_indices[:11])
        #buffer_yes= np.equal(query_labal, retrieval_label[sort_indices]).astype(int)
        buffer_yes = ((query_labal @ retrieval_label[sort_indices].transpose())>0).astype(float)
        P = np.cumsum(buffer_yes) / Ns
        R = np.cumsum(buffer_yes)/np.sum(buffer_yes)#(trainset_len)*10
        sum_p = sum_p+P
        sum_r = sum_r+R

    return sum_p/query_times,sum_r/query_times
################################################################

def get_img_code(batch_size, bit, use_gpu, img_model: nn.Module, dataset, isPrint=False):
    dataset.img_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True, pin_memory=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    B_img = torch.zeros(len(dataset), bit, dtype=torch.float)
    if use_gpu:
        B_img = B_img.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        img = data['img']  # type: torch.Tensor
        if use_gpu:
            img = img.cuda()
        f = img_model(img)
        B_img[index, :] = f.data
        if isPrint:
            print(B_img[index, :])
    B_img = torch.sign(B_img)
    return B_img.cpu()


def get_txt_code(batch_size, bit, use_gpu, txt_model: nn.Module, dataset, isPrint=False):
    dataset.txt_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, drop_last=True, pin_memory=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    B_txt = torch.zeros(len(dataset), bit, dtype=torch.float)
    if use_gpu:
        B_txt = B_txt.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        txt = data['txt']  # type: torch.Tensor
        txt = txt.float()

        if use_gpu:
            txt = txt.cuda()
        g = txt_model(txt)
        B_txt[index, :] = g.data
        if isPrint:
            print(B_txt[index, :])
    B_txt = torch.sign(B_txt)
    return B_txt.cpu()




