import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random

# 重写数据加载，基类

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    # 重新调整图片大小并裁剪
    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1*random.randint(0,4)
        osize = [int(400*zoom), int(600*zoom)]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    #训练时使用的是crop
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    # elif opt.resize_or_crop == 'no':
    #     osize = [384, 512]
    #     transform_list.append(transforms.Scale(osize, Image.BICUBIC))

    if opt.isTrain and not opt.no_flip:
        # 训练的时候有这一项
        transform_list.append(transforms.RandomHorizontalFlip())

    # 将灰度值范围从 0~255 变为 -1~1
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    # 融合多个步骤
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
