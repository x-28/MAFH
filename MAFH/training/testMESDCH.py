import os

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import Txt_net
from new_models import resnet18,resnet34, get_MS_Text
from models.alexnet import alexnet
from utils.valid import valid , pr_curve1,valid_calc
from loss.test_multisimilarity import multilabelsimilarityloss_KL1, multilabelsimilarityloss_KL,remark_loss,reset_loss,bit_scalable
from loss.quantizationloss import quantizationLoss
from utils.save_results import save_hashcodes
from loguru import logger

def train(dataset_name: str, bit: int, issave=False, batch_size=64, use_gpu=True, max_epoch=150, lr=10 ** (-1.5), isvalid=True, alpha=0.3, beta=1.9, gamma=0.9):
    print('datasetname = %s; bit = %d' % (dataset_name, bit))

    if dataset_name.lower() == 'nus wide':
        from dataset.nus_wide import get_single_datasets
        tag_length = 1000
        label_length = 21
    elif dataset_name.lower() in ['coco2014', 'coco', 'mscoco', 'ms coco']:
        from dataset.coco2014 import get_single_datasets
        tag_length = 2026
        label_length = 80
    elif dataset_name.lower() == 'mirflickr25k':
        from dataset.mirflckr25k import get_single_datasets
        tag_length = 1386
        label_length = 24
    elif dataset_name.lower() in ['tc12', 'iapr tc-12', 'iaprtc-12']:
        from dataset.tc12 import get_single_datasets
        tag_length = 2885
        label_length = 275
    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)
    loss_store = ['intra loss','inter loss' , 'quantization loss', 'loss']
    train_data, valid_data = get_single_datasets(batch_size=batch_size)

    #img_model = alexnet(bit)
    img_model = resnet34(bit, 4)
    txt_model = get_MS_Text(tag_length, bit,4)
    label_model = Txt_net(label_length, bit)

    if use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        label_model = label_model.cuda()

    num_train = len(train_data)
    train_L = train_data.get_all_label()

    F_buffer = torch.randn(num_train, bit)
    G_buffer = torch.randn(num_train, bit)
    L_buffer = torch.randn(num_train, bit)

    if use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()
        L_buffer = L_buffer.cuda()

    B = torch.sign(F_buffer + G_buffer + L_buffer)

    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)
    optimizer_label = SGD(label_model.parameters(), lr=lr)

    learning_rate = np.linspace(lr, np.power(10, -6.), max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)  # 128*1
    ones_ = torch.ones(num_train - batch_size, 1)  # (10000-128)*1
    # unupdated_size = num_train - batch_size  # 10000-128

    max_mapi2t = max_mapt2i = 0.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    for epoch in range(max_epoch):
        # train label net
        train_data.img_load()
        train_data.re_random_item()
        for data in tqdm(train_loader):  # tqdm用来显示进度条的
            ind = data['index'].numpy()

            sample_L = data['label']
            sample_L_train = sample_L.unsqueeze(1).unsqueeze(-1).type(torch.float)
            if use_gpu:
                sample_L = sample_L.cuda()
                ones = ones.cuda()  # 128*1
                ones_ = ones_.cuda()  # (10000-128)*1
                sample_L_train = sample_L_train.cuda()

            cur_l = label_model(sample_L_train)  # cur_f: (batch_size, bit)
            L_buffer[ind, :] = cur_l.data  # F_buffer:10000*64
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            L = Variable(L_buffer)

            KLloss_ll = beta * multilabelsimilarityloss_KL1(sample_L, train_L, cur_l, L)
            KLloss_lx = multilabelsimilarityloss_KL1(sample_L, train_L, cur_l, F)
            KLloss_ly = alpha * multilabelsimilarityloss_KL1(sample_L, train_L, cur_l, G)
            quantization_l = gamma * quantizationLoss(cur_l, B[ind, :])

            loss_l = KLloss_ll + KLloss_lx + KLloss_ly + quantization_l

            optimizer_label.zero_grad()
            loss_l.backward()
            optimizer_label.step()

        # train image net
        img_model.train()
        #img_model.eval()
        train_data.img_load()
        train_data.re_random_item()
        for data in tqdm(train_loader):  # tqdm用来显示进度条的
            ind = data['index'].numpy()
            sample_L = data['label']
            image = data['img']
            if use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()  # 128*1
                ones_ = ones_.cuda()  # (10000-128)*1
            #print(img_model(image))
            middle_hash,cur_f = img_model(image)
            hash_layers = middle_hash
            hash = cur_f
            hash_layers.append(hash)
            F_buffer[ind, :] = hash.data
            F = Variable(F_buffer)
            L = Variable(L_buffer)

            KLloss_xx = beta * multilabelsimilarityloss_KL(sample_L, train_L, hash_layers, F)
            KLloss_xl = multilabelsimilarityloss_KL(sample_L, train_L, hash_layers, L)
            quantization_x = gamma * quantizationLoss(hash_layers, B[ind, :])

            loss_x = KLloss_xx + KLloss_xl + quantization_x

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()
            loss_store = remark_loss(KLloss_xx,KLloss_xl,quantization_x,loss_x)
        reset_loss(loss_store)
        weight = img_model.weight.weight
        weight = torch.mean(weight,dim=1)

        # train txt net
        txt_model.train()
        train_data.txt_load()
        train_data.re_random_item()
        for data in tqdm(train_loader):  # tqdm用来显示进度条的
            ind = data['index'].numpy()

            sample_L = data['label']
            text = data['txt']
            if use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            middle_hash,cur_g = txt_model(text)
            hash_layers = middle_hash
            hash = cur_g
            hash_layers.append(hash)
            G_buffer[ind, :] = hash.data
            L = Variable(L_buffer)
            G = Variable(G_buffer)

            KLloss_yy = beta * multilabelsimilarityloss_KL(sample_L, train_L, hash_layers, G)
            KLloss_yl = multilabelsimilarityloss_KL(sample_L, train_L, hash_layers, L)
            #yx =
            quantization_y = gamma * quantizationLoss(hash_layers, B[ind, :])

            loss_y = KLloss_yy + KLloss_yl + quantization_y

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            loss_store = remark_loss(KLloss_yy,KLloss_yl,quantization_y,loss_y)
        reset_loss(loss_store )
        weight = txt_model.weight.weight
        weight = torch.mean(weight,dim=1)

        print('current epoch: %1d, LabelLoss: %3.3f, ImgLoss:%3.3f,TxtLoss:%3.3f,lr: %f' % (
        epoch + 1, loss_l, loss_x, loss_y, lr))

        # update B
        B = torch.sign(F_buffer + G_buffer + L_buffer)

        if isvalid:
            best_epoch = epoch

            mapi2t, mapt2i = valid_calc(use_gpu,valid_data,batch_size, bit,  img_model, txt_model, return_hash=False,cuda=True)
            if mapt2i + mapi2t >= max_mapt2i + max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                best_epoch = epoch+1
            print('epoch: %1d, /:%1d,valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f;MAX_MAP(i->t): %3.4f, MAX_MAP(t->i): %3.4f,best_epoch:%1d' % (epoch + 1, max_epoch,mapi2t, mapt2i, max_mapi2t, max_mapt2i,best_epoch))

        lr = learning_rate[epoch + 1]
        # set learning rate
        for param in optimizer_label.param_groups:
            param['lr'] = lr
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr
##################### 保存P R结果 ###########################
        if bit in [16,32,64,128]:
            method = 'youhuaMESDCH'
            bit = bit
            dataName = dataset_name
            if not os.path.isdir('result/'+dataName+'/'+method+'/'+str(bit)+'/'):
                os.makedirs('result/'+dataName+'/'+method+'/'+str(bit)+'/')
            qB_img, qB_txt, rB_img, rB_txt, query_label, retrieval_label= \
                valid_calc(use_gpu,valid_data,batch_size, bit,  img_model, txt_model, return_hash=True,cuda=True)

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
    logger.add('logs/testMESDCH_%s_{time}.log')
    logger.info(str(bit)+'max_mapi2t: {:.4f}'.format(max_mapi2t)+'max_mapi2t: {:.4f}'.format(max_mapt2i))

    print('...training procedure finish')

    if isvalid:
        print('max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        if issave:
            save_hashcodes(batch_size, use_gpu, bit, img_model, txt_model, dataset_name, valid_data, 'testMESDCH')
    else:
        mapi2t, mapt2i = valid_calc(batch_size, bit, use_gpu, img_model, txt_model, valid_data)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))


    if mapt2i + mapi2t >= max_mapi2t + max_mapt2i:
        max_mapi2t = mapi2t
        max_mapt2i = mapt2i
        best_epoch = epoch
        img_model.save_dict(
            os.path.join("pth", str(bit) + '-' + img_model.module_name + '.pth'))
        txt_model.save_dict(
            os.path.join("pth", str(bit) + '-' + txt_model.module_name + '.pth'))
        qB_img = qB_img.cpu()
        qB_txt = qB_txt.cpu()
        rB_img = rB_img.cpu()
        rB_txt = rB_txt.cpu()
        if (epoch + 1) > 1:
            bit_scalable(img_model, txt_model, qB_img, qB_txt, rB_img, rB_txt,
                              valid_data)
        # self.best_train_img, self.best_train_txt = self.get_train_hash()
    print(
        'epoch: [%3d/%3d], valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f in epoch %d' %
        (epoch + 1, max_epoch, mapi2t, mapt2i, max_mapi2t, max_mapt2i, best_epoch + 1))
