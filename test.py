import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.MambaLiteUNet import MambaLiteUNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')

    # 2017
    #resume_model = os.path.join('./pre_trained_models/ISIC2017/best-epoch51-loss0.1905.pth')
    
    # 2018
    resume_model = os.path.join('./pre_trained_models/ISIC2018/best-epoch53-loss0.2218.pth')
    
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing Models----------#')
    model_cfg = config.model_config
    model = MambaLiteUNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'])

    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Preparing dataset----------#')
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True, 
                             num_workers=config.num_workers,
                             drop_last=True)

    print('#----------Preparing loss, optimizer, scheduler, and scaler----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Setting other parameters----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    print('#----------Testing----------#')
    checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

    # Check if checkpoint has a "model_state_dict" key
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    # Load model weights with strict=False to handle minor mismatches
    model.module.load_state_dict(model_state_dict, strict=True)

    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )

if __name__ == '__main__':
    config = setting_config
    main(config)