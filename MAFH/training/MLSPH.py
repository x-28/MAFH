import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from torch.utils.data import DataLoader
#from config_mir import opt
from config_nus import opt
#from config_coco import opt
#from config_tc import opt
#from data_handler import *
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import Txt_net
from models.resnet import resnet18
from utils.valid import valid , pr_curve1
from utils.utils import calc_map_k
import torch.nn.functional as Funtional1
#from dataset.data.dataset import DatasetMirflckr25KTrain, DatasetMirflckr25KValid
from visdom import Visdom
import scipy.io as sio

def train():
    print('datasetname = %s; bit = %d' % (opt.dataset_name, opt.bit))

    #train_data = DatasetMirflckr25KTrain(opt.img_dir, opt.imgname_mat_dir,opt.tag_mat_dir,opt.label_mat_dir, batch_size=opt.batch_size, train_num=opt.training_size, query_num=opt.query_size)
    #valid_data = DatasetMirflckr25KValid(opt.img_dir,opt.imgname_mat_dir,opt.tag_mat_dir,opt.label_mat_dir, query_num=opt.query_size)

    # if opt.dataset_name.lower() == 'nus wide':
    #     from dataset.nus_wide import get_single_datasets
    #     tag_length = 1000
    #     label_length = 21
    # elif opt.dataset_name.lower() in ['coco2014', 'coco', 'mscoco', 'ms coco']:
    #     from dataset.coco2014 import get_single_datasets
    #     tag_length = 2026
    #     label_length = 80
    # elif opt.dataset_name.lower() == 'mirflickr25k':
    #     from dataset.mirflckr25k import get_single_datasets
    #     tag_length = 1386
    #     label_length = 24
    # elif opt.dataset_name.lower() in ['tc12', 'iapr tc-12', 'iaprtc-12']:
    #     from dataset.tc12 import get_single_datasets
    #     tag_length = 2885
    #     label_length = 275
    # else:
    #     raise ValueError("there is no dataset name is %s" % opt.dataset_name)
    from dataset.nus_wide import get_single_datasets

    train_data, valid_data = get_single_datasets(img_dir=opt.img_dir, img_mat_url= opt.imgname_mat_dir, tag_mat_url= opt.tag_mat_dir, label_mat_url= opt.label_mat_dir,
                                                 batch_size=opt.batch_size, train_num=opt.training_size, query_num=opt.query_size, seed=opt.default_seed)
    img_model = resnet18(opt.bit)
    txt_model = Txt_net(opt.tag_length, opt.bit)
    label_model = Txt_net(opt.label_length, opt.bit)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()


    num_train = len(train_data)
    train_L = train_data.get_all_label()

    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)

    if opt.use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    
    B = torch.sign(F_buffer + G_buffer)

    batch_size = opt.batch_size  # 128

    lr = opt.lr
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)

    learning_rate = np.linspace(lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)  
    ones_ = torch.ones(num_train - batch_size, 1)  
    unupdated_size = num_train - batch_size  

    max_mapi2t = max_mapt2i = 0.
    lossResult=np.zeros([2,4,opt.max_epoch])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=True)
    for epoch in range(opt.max_epoch):
        # train image net
        train_data.img_load()
        train_data.re_random_item()
        for data in tqdm(train_loader): 
            ind = data['index'].numpy()
            unupdated_ind = np.setdiff1d(range(num_train), ind)  

            sample_L = data['label']
            image = data['img']
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()  
                ones_ = ones_.cuda()  

           
            S = calc_neighbor(sample_L, train_L)  
            cur_f = img_model(image)  
            F_buffer[ind, :] = cur_f.data 
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            theta_x =calc_inner(cur_f,G)
            logloss_x=opt.alpha*torch.sum(torch.pow(S-theta_x,2))/(num_train * batch_size)
            theta_xx=calc_inner(cur_f,F)
            logloss_xx=opt.beta*torch.sum(torch.pow(S-theta_xx,2))/(num_train * batch_size)
            quantization_x = opt.gamma * torch.sum(torch.pow(B[ind, :] - cur_f, 2))/(batch_size * opt.bit)
            loss_x = logloss_x + logloss_xx + quantization_x
            loss_x=10*loss_x

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

        # train txt net
        train_data.txt_load()
        train_data.re_random_item()
        for data in tqdm(train_loader): 
            ind = data['index'].numpy()
            unupdated_ind = np.setdiff1d(range(num_train), ind)  

            sample_L = data['label']
            text = data['txt']
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            
            S = calc_neighbor(sample_L, train_L) 
            cur_g= txt_model(text)  
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            # calculate loss
            theta_y = calc_inner(cur_g,F)
            logloss_y=opt.alpha*torch.sum(torch.pow(S-theta_y,2))/(num_train * batch_size)
            theta_yy=calc_inner(cur_g,G)
            logloss_yy=opt.beta*torch.sum(torch.pow(S-theta_yy,2))/(num_train * batch_size)
            quantization_y = opt.gamma *torch.sum(torch.pow(B[ind, :] - cur_g, 2))/(batch_size*opt.bit)
            loss_y = logloss_y + logloss_yy +  quantization_y
            loss_y = 10*loss_y


            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

        print('current epoch: %1d, ImgLoss:%3.3f,TxtLoss:%3.3f,lr: %f' % (epoch + 1, loss_x, loss_y, lr))
        lossResult[0,:,epoch]=[logloss_x,logloss_xx,quantization_x,loss_x]
        lossResult[1,:,epoch]=[logloss_y,logloss_yy,quantization_y,loss_y]
        # update B
        B = torch.sign(F_buffer + G_buffer)

        if opt.valid:
            best_epoch = epoch
            #opt.batch_size,opt.bit,opt.use_gpu,
            mapi2t, mapt2i = valid(opt.batch_size,opt.bit,opt.use_gpu, img_model, txt_model, valid_data)
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:  
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                best_epoch = epoch+1
            #print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f;MAX_MAP(i->t): %3.4f, MAX_MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i,max_mapi2t,max_mapt2i))
            print('epoch: %1d,/%d,valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f;MAX_MAP(i->t): %3.4f, MAX_MAP(t->i): %3.4f,best_epoch:%1d' % (epoch + 1, opt.max_epoch,mapi2t, mapt2i, max_mapi2t, max_mapt2i,best_epoch))
        lr = learning_rate[epoch + 1]

        # set learning rate
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr
        ##################### 保存P R结果 ###########################
        if opt.bit in [16,32,64,128]:
            method = 'MLSPH'
            bit = opt.bit
            dataName = opt.dataset_name
            if not os.path.isdir('result/'+dataName+'/'+method+'/'+str(bit)+'/'):
                os.makedirs('result/'+dataName+'/'+method+'/'+str(bit)+'/')
            qB_img, qB_txt, rB_img, rB_txt, query_label, retrieval_label= \
                valid(opt.batch_size, opt.bit, opt.use_gpu, img_model, txt_model, valid_data, return_hash=True)
            P1, R1 = pr_curve1(
                qB_img,
                rB_txt,
                query_label,
                retrieval_label,
            )
            np.savetxt ('result/'+dataName+'/'+method+'/'+str(bit)+'/'+'/'+'(i2t)p.txt',P1,fmt='%3.5f')
            np.savetxt ('result/'+dataName+'/'+method+'/'+str(bit)+'/'+'/'+'(i2t)r.txt',R1,fmt='%3.5f')
            P2, R2 = pr_curve1(
                qB_txt,
                rB_img,
                query_label,
                retrieval_label,
            )
            np.savetxt ('result/'+dataName+'/'+method+'/'+str(bit)+'/'+'/'+'(t2i)p.txt',P2,fmt='%3.5f')
            np.savetxt ('result/'+dataName+'/'+method+'/'+str(bit)+'/'+'/'+'(t2i)r.txt',R2,fmt='%3.5f')
    ##################################################################################################

    print('...training procedure finish')
    if opt.valid:  #
        print('max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
    else:
        #opt.batch_size, opt.bit, opt.use_gpu
        mapi2t, mapt2i = valid(opt.batch_size, opt.bit, opt.use_gpu, img_model, txt_model, valid_data)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i

    viz=Visdom(env='my_loss')
    viz.line(X=np.arange(opt.max_epoch)+1,
             Y=np.column_stack((lossResult[0,0,:],lossResult[0,1,:],lossResult[0,2,:],lossResult[0,3,:])),
             opts=dict(
                 showlegend=True,
                 legend=['logloss_x','logloss_xx','quantity_x','loss_x'],
                 title='image loss',
                 xlabel='epoch number',
                 ylabel='loss value',
             ))
    viz.line(X=np.arange(opt.max_epoch) + 1,
             Y=np.column_stack((lossResult[1, 0, :], lossResult[1, 1, :], lossResult[1, 2, :], lossResult[1, 3, :])),
             opts=dict(
                 showlegend=True,
                 legend=['logloss_y', 'logloss_yy', 'quantity_y', 'loss_y'],
                 title='text loss',
                 xlabel='epoch number',
                 ylabel='loss value',
             ))

def calc_neighbor(label1, label2):
    # calculate the similar matrix
    if opt.use_gpu:
        label1 = label1.float()
        label2 = label2.float()
        Sim = label1.matmul(label2.transpose(0, 1)).type(torch.cuda.FloatTensor)
    else:
        Sim = label1.matmul(label2.transpose(0, 1)).type(torch.FloatTensor)

    numLabel_label1 = torch.sum(label1, 1)
    numLabel_label2 = torch.sum(label2, 1)

    x = numLabel_label1.unsqueeze(1) + numLabel_label2.unsqueeze(0) - Sim
    Sim = 2 * Sim / x  # [0,2]

    # cosine similarity
    # label1 = myNormalization(label1)
    # label2 = myNormalization(label2)
    # Sim = (label1.unsqueeze(1) * label2.unsqueeze(0)).sum(dim=2)

    # print(torch.max(Sim))
    # print(torch.min(Sim))
    return Sim


def myNormalization(X):
    x1=torch.sqrt(torch.sum(torch.pow(X, 2),1)).unsqueeze(1)
    return X/x1

def calc_inner(X1,X2):
    X1=myNormalization(X1)
    X2=myNormalization(X2)
    X=torch.matmul(X1,X2.t())  # [-1,1]
   
    return X


if __name__ == '__main__':
    train()
