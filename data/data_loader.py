
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    # 训练时使用的是 custom 加载方法
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
