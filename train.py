import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

#加载命令行选项
opt = TrainOptions().parse()
#加载配置选项
config = get_config(opt.config)

#加载数据 (unaligned loader)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#创建模型 (singleGANModel)
model = create_model(opt)
#创建可视化
visualizer = Visualizer(opt)

total_steps = 0

# 正如文章所说，200次训练，前100次学习率为1e-4，后100次学习率线性降低至0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters(epoch)

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        # 每隔数个steps（本文中是20000） 保存一次模型
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    # 每隔数个epoch （本文中是5个） 保存一次模型
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    # 统计时间
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if opt.new_lr:
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        '''
        算法使用的是这个分支，学习率线性降低
        '''
        if epoch > opt.niter:
            model.update_learning_rate()
