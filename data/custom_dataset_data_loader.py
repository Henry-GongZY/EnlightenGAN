

import torch.utils.data
from data.base_data_loader import BaseDataLoader

# 使用不同的数据加载方式(挺多没有用到的),训练时使用的是 Unaligned 进行加载
def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    # 默认的方式
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'unaligned_random_crop':
        from data.unaligned_random_crop import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'pair':
        from data.pair_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'syn':
        from data.syn_dataset import PairDataset
        dataset = PairDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    # 读取图片在这里处理
    dataset.initialize(opt)
    return dataset

# 数据集加载器
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        # 对数据集进行处理并加载
        self.dataset = CreateDataset(opt)
        """使用dataloader加载数据
        dataset: 加载的数据集
        batchsize: 一次处理的数据量
        shuffle: 是否打乱
        num_workers: 工作线程数
        """
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle = not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
