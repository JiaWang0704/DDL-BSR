import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
import data.util as util2
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group

def get_parser2(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str,default=path, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(path, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    return opt


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="/userHome/guest/wangjia/blindsr/DDL-BSR/codes/config/DDLBSR/options/train/train_setting2_stage3_x4.yml", help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    if opt["net2_opt"] is not None:
        opt2 = get_parser2(opt["net2_opt"])

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    # seed = opt["train"]["manual_seed"]
    # if seed is None:
    #     seed = random.randint(1, 10000)

    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        opt["pca_matrix_path"], map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        util.set_random_seed(opt['train']['manual_seed'])

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            
            # for key, path in opt["path"].items():
            #     print('paths',key, path)
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            train_ker = opt["train"]["kernel_criterion"]
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    if opt["net2_opt"] is not None:
        model2 = create_model(opt2)
        model = create_model(opt, model2.netG)  # load pretrained model of SFTMD
    else:
        model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    prepro = util.SRMDPreprocessing(
        scale=opt["scale"], pca_matrix=pca_matrix, cuda=True, **opt["degradation"]
    )
    kernel_size = opt["degradation"]["ksize"]
    padding = kernel_size // 2
    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    # if rank <= 0:
    prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
    for epoch in range(start_epoch, total_epochs + 1):
        # print('train-------------', epoch)
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            # print(current_step, 1)
            if current_step > total_iters:
                break
            LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(train_data["GT"], True, return_blur=True)
            # print(LR_img.shape, ker_map.shape, kernels.shape, lr_blured_t.shape, lr_t.shape)
            # zzzzz
            LR_img = (LR_img * 255).round() / 255
            # print(current_step, 2)
            model.feed_data(
                LR_img, GT_img=train_data["GT"], ker_map=ker_map, kernel=kernels, lr_blured=lr_blured_t, lr=lr_t
            )
            # print(current_step, 3)
            model.optimize_parameters(current_step)
            # print(current_step, 4)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            # print(current_step, 5)
            # visuals = model.get_current_visuals()
            # print(current_step, 2)
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    # if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    #     if rank <= 0:
                    #         tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation, to produce ker_map_list(fake)
            if (current_step % opt["train"]["val_freq"] == 0 or current_step==5 or current_step==500) and rank <= 0:
                avg_psnr = 0.0
                avg_psnr_k = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):

                    # LR_img, ker_map = prepro(val_data['GT'])
                    LR_img = val_data["LQ"]
                    lr_img = util.tensor2img(LR_img)  # save LR image for reference

                    # valid Predictor
                    model.feed_data(LR_img, val_data["GT"])
                    model.test()
                    visuals = model.get_current_visuals()

                    # Save images for reference
                    img_name = val_data["LQ_path"][0].split('/')[-1]
                    kernel_path = os.path.join(val_data['GT_path'][0].replace('/HR/', '/kernel/'))
                    img_dir = os.path.join(opt["path"]["val_images"], str(current_step))
                    # img_dir = os.path.join(opt['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)
                    # save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))
                    # util.save_img(lr_img, save_lr_path)
                    sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                    save_img_path = os.path.join(img_dir,img_name)
                    save_ker_path=os.path.join(img_dir, 'ker'+img_name)
                    
                    kernel = (
                        visuals["ker"][0]
                        .numpy()
                        .reshape(
                            opt["degradation"]["ksize"], opt["degradation"]["ksize"]
                        )
                    )

                    gt_ker = cv2.imread(kernel_path, cv2.IMREAD_UNCHANGED)
                    kernel = 1 / (np.max(kernel) + 1e-4) * 255 * kernel
                    # print(gt_ker, kernel)
                    avg_psnr_k += util.calculate_psnr(gt_ker,kernel)
                    
                    cv2.imwrite(save_ker_path, kernel)
                    util.save_img(sr_img, save_img_path)
                    # gtsave_img_path = os.path.join(
                    #     img_dir, "{:s}_GT.png".format(img_name, current_step)
                    # )
                    # util.save_img(gt_img, gtsave_img_path)

                    # calculate PSNR
                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]

                    cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                    cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)

                    # print(val_data["GT"].shape, gt_img.shape, sr_img.shape)
                    # print(cropped_gt_img_y.shape, cropped_sr_img_y.shape)

                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    idx += 1

                avg_psnr = avg_psnr / idx
                avg_psnr_k = avg_psnr_k/idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info("# Validation # PSNR: {:.6f}, PNSR_k: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, avg_psnr_k, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, PNSR_k: {:.6f},".format(
                        epoch, current_step, avg_psnr, avg_psnr_k
                    )
                )
                # tensorboard logger
                # if opt["use_tb_logger"] and "debug" not in opt["name"]:
                #     tb_logger.add_scalar("psnr", avg_psnr, current_step)

                if avg_psnr > 20:
                    # if rank <= 0:
                    prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
                        # torch.save(prev_state_dict, opt["name"]+".pth")
                else:
                    logger.info("# Validation crashed, use previous state_dict...\n")
                    model.netG.module.load_state_dict(copy.deepcopy(prev_state_dict), strict=True)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    # tb_logger.close()


if __name__ == "__main__":
    main()
