import numpy as np
import os
import shutil
import dataset
import utils
import torch
from torch.autograd import Variable
import lmdb
import six
import sys
from PIL import Image
import torchvision.models as models
from torch import nn
from mini_net import Net

# 需要更改的参数
root = "/home/chenhan/read_picture/LMDB/test/"
model_path = "/home/chenhan/read_picture/net_params.pkl"
model = Net()
is_cuda = True

pic_count = 0
right_count = 0

wrong_folder = './'
wrong_folder = 'wrong'
filename = wrong_folder + '/' + 'wrong.txt'
if os.path.exists(wrong_folder):
    shutil.rmtree(wrong_folder)
os.mkdir(wrong_folder)
with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    f.write("Wrong prediction:\n")

if is_cuda:
    model.cuda()
    model = nn.DataParallel(model)
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))



env = lmdb.open(
    root,
    max_readers=1,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print(nSamples)
        
with open(filename,'a') as f: # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）

    for Index in range(1,nSamples+1):

        pic_count += 1

        img_key = 'image-%09d' % Index
        label_key = 'label-%09d' % Index

        with env.begin(write=False) as txn:
            label_key = 'label-%09d' % Index
            label = txn.get(label_key.encode()).decode()
            label = int(label)

            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            image_tosave = Image.open(buf).convert('RGB')

        transformer = dataset.resizeNormalize((32, 32))
        image = transformer(image)

        if is_cuda:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        model.eval()
        preds = model(image)

        _, predicted = torch.max(preds, 1)

        if (predicted.cpu() == label):
            right_count+=1
        else:
            wrong_img_name = wrong_folder + '/' + str(img_key) + '_' + str(label) + '.jpg'
            # 如果中间是‘:’而不是‘_’就出不来图片...玄学
            #image_tosave.save(wrong_img_name)
            print('predicted: %-20s,real: %-20s,key: %s ' 
            % (predicted.cpu(), label, img_key))
            f.write('predicted: %-20s,real: %-20s,key: %s \n' 
            % (predicted.cpu(), label, img_key))


print ("number of tested: %d" % pic_count)
print ("number of right: %d" % right_count)
print ("Accuracy : %f" % (right_count/pic_count))