
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ast
import torch
import random
import argparse
import numpy as np
from data_loader.loader import Loader
from core import train, test, build_model
from tools import make_dirs, Logger, os_walk, time_now
import warnings
warnings.filterwarnings("ignore")
from solver import build_optimizer, build_lr_scheduler
from config.config_rn import get_args
from tools.utils import save_train_configs, load_train_configs, time_now
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy


best_mAP_text = 0
best_rank1_text = 0
best_mINP_text = 0
best_mAP_ir = 0
best_rank1_ir = 0
best_mINP_ir = 0
best_mAP_fusion = 0
best_rank1_fusion = 0
best_mINP_fusion = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else "cpu")

    global best_mAP_text
    global best_rank1_text
    global best_mINP_text
    global best_mAP_ir
    global best_rank1_ir
    global best_mINP_ir
    global best_mAP_fusion
    global best_rank1_fusion
    global best_mINP_fusion

    print("=================Constructing output dir=================")
    if config.DEBUG:
        config.output_path = config.DEBUG_DIR
        print(f"Debug [{config.mode}] mode, dir: {config.output_path}")
    elif (config.auto_resume_training_from_lastest_step or config.resume_train_epoch>=0) and config.mode == 'train':
        print(f"Resume training from the latest step, dir: {config.output_path}")
    elif config.mode == 'test':
        print(f"Start testing with trained model, dir: {config.output_path}")
    else:
        config.output_path += f'{config.dataset}/'
        FV = config.Fix_Visual
        FF = config.Feat_Filter
        QBN = config.uni_BN
        if not FV and not FF and not QBN:
            config.output_path += 'Base/'

        if QBN and not FV and not FF:
            config.output_path += 'QBN/'

        if FV and not FF and not QBN:
            config.output_path += 'FV/'
        if FF and not FV and not QBN:
            config.output_path += 'Filter/'
        if FV and FF and not QBN:
            config.output_path += 'FV_Filter/'

        if FV and QBN and not FF:
            config.output_path += 'FV_QBN/'
        if FF and QBN and not FV:
            config.output_path += 'Filter_QBN/'
        if FV and FF and QBN:
            config.output_path += 'FV_Filter_QBN/'




        config.output_path += 'Baseline'
        if config.dataset == 'regdb':
            config.output_path += f'_{config.trial}'
        config.output_path += '_' + f'train[{config.training_mode}]' 
        if len(config.training_mode.split('_')) == 3:
            config.output_path += '_' + f'joint[{config.joint_mode}]'
        if "Text" in config.training_mode:
            config.output_path += '_' + config.captioner_name
            if "IR" in config.training_mode:
                config.output_path += '_' + config.fusion_way
            if config.llm_aug:
                config.output_path += '_' + 'LLM' + '_' + str(config.llm_aug_prob)
        if config.loss_names:
            config.output_path += '_' + config.loss_names
        if config.Return_B4_BN:
            config.output_path += '_' + 'Return_B4_BN'
        if config.uni_BN:
            config.output_path += '_' + 'uni_BN'
        if config.Fix_Visual:
            config.output_path += '_' + 'Fix_Visual'
        if config.Feat_Filter:
            config.output_path += '_' + 'Filtered'
        # config.output_path += '_' + time_now()
        print(f"start training from zero, dir {config.output_path}, training mode: {config.training_mode}")


    print("=================Preparing data=================")
    if config.dataset == 'sysu':
        print(f"Dataset: {config.dataset}, dir: {config.sysu_data_path}")
        config.pid_num = 395
    elif config.dataset == 'regdb':
        print(f"Dataset: {config.dataset}, dir: {config.regdb_data_path}")
        config.pid_num = 206
    elif config.dataset == 'llcm':
        print(f"Dataset: {config.dataset}, dir: {config.llcm_data_path}")
        config.pid_num = 713
    loaders = Loader(config)
    

    print("=================Preparing model=================")
    model = build_model(config)
    if config.training_weight_init and config.Fix_Visual:
        model.load_state_dict(torch.load(config.training_weight_init,map_location=device), strict=False)
        model.backup_pool = deepcopy(model.base_model.visual.__getattr__(config.pooling))
        model.backup_classifier = deepcopy(model.classifier)
        print(f"Successfully load model from {config.training_weight_init}")
    model = model.to(device)

    # ===== 这里统计参数量 =====
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Params: {total_params / 1e6:.2f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f} M")

    if config.mode == 'train':
        make_dirs(model.output_path)
        make_dirs(model.save_model_path)
        make_dirs(model.save_logs_path)
        # make_dirs(os.path.join(model.output_path, 'checkpoint/'))
        # check_point_path = os.path.join(model.output_path, 'checkpoint/checkpoint_latest.pth')

        logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.log'))
        logger('\n' * 3)
        logger(config)


        performance_writer = SummaryWriter(os.path.join(model.output_path,'vis_logs/performance'))
        loss_writer = SummaryWriter(os.path.join(model.output_path,'vis_logs/loss'))
        save_train_configs(config.output_path, config)

        print("=================preparing optimizer=================")

        optimizer = build_optimizer(config, model)
        scheduler = build_lr_scheduler(config, optimizer)
            
        start_train_epoch = 0

        # 在训练循环开始前添加早停参数
        no_improve_epochs = 0  # 记录连续未提升的轮数
        early_stop_patience = 10  # 容忍轮数，可调整
        best_overall_metric = 0.0  # 用于早停监控的最佳指标
        start_monitor_epoch = 15   # 从第15轮开始监控，避免过早停止

        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            scheduler.step(current_epoch)

            if current_epoch < config.total_train_epoch:
                result_vals, result = train(model, loaders, config, optimizer)
                # visual log
                for key, value in zip(*result_vals):
                    loss_writer.add_scalar(key, value, current_epoch)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))

            # testing while training
            if current_epoch + 1 >= config.eval_start_epoch and (current_epoch + 1) % config.eval_epoch == 0:
                result_dict = test(model, loaders, config, device)

                # 选择监控指标（这里以Text模式的Rank1为例，你可以根据需求调整）
                current_metric = 0
                monitor_mode = "Text"  # 可以选择 "Text", "IR", 或 "Fusion"
                
                if monitor_mode in config.test_modality:
                    if monitor_mode == "Text":
                        mINP, mAP, cmc = result_dict['Text']
                        current_metric = cmc[0]  # 使用Rank-1作为监控指标
                    elif monitor_mode == "IR":
                        mINP, mAP, cmc = result_dict['IR']
                        current_metric = cmc[0]
                    elif monitor_mode == "Fusion":
                        mINP, mAP, cmc = result_dict['Fusion']
                        current_metric = cmc[0]
                
                # 早停逻辑：只在达到开始监控的轮数后生效
                if current_epoch + 1 >= start_monitor_epoch:
                    if current_metric > best_overall_metric:
                        best_overall_metric = current_metric
                        no_improve_epochs = 0
                        print(f"Epoch {current_epoch}: 最佳指标提升至 {best_overall_metric:.4f}")
                    else:
                        no_improve_epochs += 1
                        print(f"Epoch {current_epoch}: 指标未提升，计数器 {no_improve_epochs}/{early_stop_patience}")
                    
                    # 检查是否触发早停
                    if no_improve_epochs >= early_stop_patience:
                        print(f"🎯 早停触发！在 epoch {current_epoch} 停止训练")
                        print(f"最佳指标: {best_overall_metric:.4f}")
                        break

                if 'IR' in config.test_modality:
                    mINP_ir, mAP_ir, cmc_ir = result_dict['IR']
                    is_best_rank_ir = (cmc_ir[0] >= best_rank1_ir)
                    # visual log
                    performance_writer.add_scalar(f'R1_IR', cmc_ir[0], current_epoch)
                    performance_writer.add_scalar(f'mAP_IR', mAP_ir, current_epoch)
                    performance_writer.add_scalar(f'mINP_IR', mINP_ir, current_epoch)
                    # new add
                    if is_best_rank_ir:
                        logger(f"New Best IR_RGB!!!")
                        best_rank1_ir = max(cmc_ir[0], best_rank1_ir)
                        best_mAP_ir = mAP_ir
                        best_mINP_ir = mINP_ir
                    logger(f"Best IR_RGB mINP: {best_mINP_ir}, Best mAP: {best_mAP_ir}, Best Rank1: {best_rank1_ir}")
                    # new add
                    model.save_model(current_epoch, is_best_rank_ir, mode="IR")
                    logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"IR_RGB",
                                                                                    mINP_ir, mAP_ir, cmc_ir))
                if 'Fusion' in config.test_modality:
                    mINP_fusion, mAP_fusion, cmc_fusion = result_dict['Fusion']
                    is_best_rank_fusion = (cmc_fusion[0] >= best_rank1_fusion)
                    # visual log
                    performance_writer.add_scalar(f'R1_Fusion', cmc_fusion[0], current_epoch)
                    performance_writer.add_scalar(f'mAP_Fusion', mAP_fusion, current_epoch)
                    performance_writer.add_scalar(f'mINP_Fusion', mINP_fusion, current_epoch)
                    # new add
                    if is_best_rank_fusion:
                        logger(f"New Best Fusion_RGB!!!")
                        best_rank1_fusion = max(cmc_fusion[0], best_rank1_fusion)
                        best_mAP_fusion = mAP_fusion
                        best_mINP_fusion = mINP_fusion
                    logger(f"Best Fusion_RGB mINP: {best_mINP_fusion}, Best mAP: {best_mAP_fusion}, Best Rank1: {best_rank1_fusion}")
                    # new add
                    model.save_model(current_epoch, is_best_rank_fusion, mode='Fusion')
                    logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Fusion_RGB",
                                                                                    mINP_fusion, mAP_fusion, cmc_fusion))
                if 'Text' in config.test_modality:
                    mINP_text, mAP_text, cmc_text = result_dict['Text']
                    is_best_rank_text = (cmc_text[0] >= best_rank1_text)
                    # visual log
                    performance_writer.add_scalar(f'R1_Text', cmc_text[0], current_epoch)
                    performance_writer.add_scalar(f'mAP_Text', mAP_text, current_epoch)
                    performance_writer.add_scalar(f'mINP_Text', mINP_text, current_epoch)
                    # new add
                    if is_best_rank_text:
                        logger(f"New Best Text_RGB!!!")
                        best_rank1_text = max(cmc_text[0], best_rank1_text)
                        best_mAP_text = mAP_text
                        best_mINP_text = mINP_text
                    logger(f"Best Text_RGB mINP: {best_mINP_text}, Best mAP: {best_mAP_text}, Best Rank1: {best_rank1_text}")
                    # new add
                    model.save_model(current_epoch, is_best_rank_text, mode='Text')
                    logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Text_RGB",
                                                                                    mINP_text, mAP_text, cmc_text))

        performance_writer.close()
        loss_writer.close()
        
    elif config.mode == 'test':
        make_dirs(model.output_path)
        make_dirs(model.save_model_path)
        make_dirs(model.save_logs_path)
        logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'test.log'))
        logger('\n' * 3)
        logger(config)
        print('Testing Modality Mode:{}'.format(config.test_modality))
        print('Testing Model Type:{}'.format(config.test_model_type))
        if config.Fix_Visual:
            model.load_state_dict(torch.load(config.training_weight_init,map_location=device),strict=False)
            model.backup_pool = deepcopy(model.base_model.visual.__getattr__(config.pooling))
            model.backup_classifier = deepcopy(model.classifier)
        if 'checkpoint' in config.test_model_path:
            model.load_state_dict(torch.load(config.test_model_path,map_location=device)['model_state_dict'], strict=False)
        else:
            model.load_state_dict(torch.load(config.test_model_path,map_location=device), strict=False)
        print('Successfully resume model from {}'.format(config.test_model_path))
        result_dict = test(model, loaders, config, device)
        if "IR" in config.test_modality:
            mINP_ir, mAP_ir, cmc_ir = result_dict['IR']
            if config.LOG4TEST:
                logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"IR_RGB",
                                                                                    mINP_ir, mAP_ir, cmc_ir))
            else:
                print('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"IR_RGB",
                                                                                    mINP_ir, mAP_ir, cmc_ir))
            
        if "Fusion" in config.test_modality:
            mINP_fusion, mAP_fusion, cmc_fusion = result_dict['Fusion']
            if config.LOG4TEST:
                if config.CAT_EVAL:
                    logger('===================Test with CAT FEAT===================')
                else:
                    logger('===================Test without CAT FEAT===================')
                logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Fusion_RGB",
                                                                                    mINP_fusion, mAP_fusion, cmc_fusion))
            else:
                print('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Fusion_RGB",
                                                                                    mINP_fusion, mAP_fusion, cmc_fusion))
            
        if "Text" in config.test_modality:
            mINP_text, mAP_text, cmc_text = result_dict['Text']
            if config.LOG4TEST:
                logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Text_RGB",
                                                                                    mINP_text, mAP_text, cmc_text))
            else:
                print('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Text_RGB",
                                                                                    mINP_text, mAP_text, cmc_text))
            

if __name__ == '__main__':
    config = get_args()
    if config.config_select == 'default':
        pass
    else:
        config = load_train_configs(config.config_select)
    seed_torch(config.seed)
    main(config)
