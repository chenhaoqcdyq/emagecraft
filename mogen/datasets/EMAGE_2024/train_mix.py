import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

import copy
try:
    from mmcv import Config, DictAction
except:
    import mmengine
    from mmengine import Config, DictAction
try:
    from mmcv.runner import get_dist_info, init_dist
except:
    from mmengine.dist.utils import get_dist_info, init_dist
    
from mogen.apis import set_random_seed, train_model
from mogen.datasets import build_dataset
from mogen.datasets import build_dataloader, build_dataset
from mogen.datasets.EMAGE_2024.dataloaders.build_vocab import Vocab
from dataloaders.data_tools import joints_list
from mmcv.parallel import DataContainer
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
# from torch.utils.data import DataLoader

class TrainTmp:
    def __init__(self, args):
        self.args = args
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")

# def custom_collate(batch):
#     # 提取 batch 中的每个元素，检查它是否是 DataContainer
#     batch = [x.data if isinstance(x, DataContainer) else x for x in batch]
#     return torch.utils.data.dataloader.default_collate(batch)

def custom_collate(batch):
    # for i, data in enumerate(batch):
    #     for key, value in data.items():
    #         if isinstance(value, DataContainer):
    #             print(f"DataContainer found in batch {i}, key: {key}")
    #             print(f"Data: {value.data}")
    batch = [{key: (d[key].data if isinstance(d[key], DataContainer) else d[key]) 
              for key in d} for d in batch]
    return torch.utils.data.dataloader.default_collate(batch)

        
class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        # self.rank = dist.get_rank()
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        if self.rank==0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
                wandb.init(project=args.project, entity="liu1997", dir=args.out_path, name=args.name[12:] + args.notes)
                wandb.config.update(args)
                self.writer = None  
        #self.test_demo = args.data_path + args.test_data_path + "bvh_full/"    
        # args.motioncraft_config = "configs/EMGAE/T2M_motionx_align_Finedance_Beats2_face_no_loss_0_25b.py"
        cfg = Config.fromfile(args.motioncraft_config)
        if "gpu_ids" in args:
            cfg.gpu_ids = args.gpu_ids
        else:
            cfg.gpu_ids = range(1) if "gpus" not in args else args.gpus
        # set random seeds
        if args.random_seed is not None:
            logger.info(f'Set random seed to {args.random_seed}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(args.random_seed, deterministic=args.deterministic)
        cfg.seed = args.random_seed
        # meta['seed'] = args.seed
        if cfg.data.train.get('base', None) is None:
            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))
        else:
            # Multi-modailty (Text2Motion) Training
            datasets_dict = []
            if 'speech' in cfg.data.train:
                beats_datasets = build_dataset(cfg.data.train.speech)
                datasets_dict.append(beats_datasets)
            if 'music' in cfg.data.train:
                finedance_datasets = build_dataset(cfg.data.train.music)
                datasets_dict.append(finedance_datasets)
            if 'text' in cfg.data.train:
                text_datasets = build_dataset(cfg.data.train.text)
                datasets_dict.append(text_datasets)
            # logger.info(f"text_datasets: {len(text_datasets)}; finedance_datasets: {len(finedance_datasets)}; beats_datasets: {len(beats_datasets)}")
            datasets = build_dataset(cfg.data.train.base)
            datasets.merge_datasets(
                datasets_dict
            )
            datasets = [datasets]
            logger.info(datasets[0].pipelines)   
        
        datasets= datasets if isinstance(datasets, (list, tuple)) else [datasets]
        # self.train_loader = [
        #     build_dataloader(
        #         ds,
        #         cfg.data.samples_per_gpu,
        #         cfg.data.workers_per_gpu,
        #         # cfg.gpus will be ignored if distributed
        #         num_gpus=len(cfg.gpu_ids),
        #         dist=args.ddp,
        #         round_up=True,
        #         seed=cfg.seed) for ds in datasets
        # ]
        # self.train_loader = [
        #     torch.utils.data.DataLoader(
        #     ds, 
        #     batch_size=args.batch_size,  
        #     shuffle=False if args.ddp else True,  
        #     num_workers=args.loader_workers,
        #     drop_last=True,
        #     sampler=torch.utils.data.distributed.DistributedSampler(ds) if args.ddp else None, 
        # ) for ds in datasets
        # ]
        
        # self.train_loader = self.train_loader[0]
        # self.world_size = len(cfg.gpu_ids)
        if args.ddp:
            # 用 DistributedSampler 替换 SequentialSampler
            sampler = DistributedSampler(datasets[0], num_replicas=self.world_size, rank=self.rank)
        else:
            # 否则可以使用 SequentialSampler
            sampler = SequentialSampler(datasets[0])
        self.train_loader = DataLoader(datasets[0], batch_size=args.batch_size, sampler=sampler, collate_fn=custom_collate)
        self.train_data = TrainTmp(args)
        self.train_length = len(self.train_loader)

        
        if args.ddp:
            # if "RVQVAE" not in args.g_name:
                # self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            if "RVQVAE" in args.g_name:
                from mogen.datasets.EMAGE_2024.models.vq.model import RVQVAE
                self.model = RVQVAE(
                    args, args.pose_dims, args.nb_code, args.code_dim, args.code_dim,
                    args.down_t, args.stride_t, args.width, args.depth,
                    args.dilation_growth_rate, args.vq_act, args.vq_norm
                ).to(self.rank)
            else:
                model_module = __import__(f"models.{args.model}", fromlist=["something"])
                self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            # self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=True)
        else: 
            if "RVQVAE" in args.g_name:
                from mogen.datasets.EMAGE_2024.models.vq.model import RVQVAE
                models_ = RVQVAE(
                    args, args.pose_dims, args.nb_code, args.code_dim, args.code_dim,
                    args.down_t, args.stride_t, args.width, args.depth,
                    args.dilation_growth_rate, args.vq_act, args.vq_norm
                )
                self.model = torch.nn.DataParallel(models_, args.gpus).cuda()
            else:
                model_module = __import__(f"models.{args.model}", fromlist=["something"])
                self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        
        if args.resume_path:
            checkpoint = torch.load(args.resume_path)
            self.model.load_state_dict(checkpoint['model_state'])
            if 'opt_state' in checkpoint and 'lrs' in checkpoint:
                self.opt.load_state_dict(checkpoint['opt_state'])
                self.opt_s.load_state_dict(checkpoint['lrs'])
            # start_epoch = checkpoint['epoch']
        # else:
        #     start_epoch = 0
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)
        
        # if args.d_name is not None:
        #     if args.ddp:
        #         self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
        #         self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model, process_group)   
        #         self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
        #                            broadcast_buffers=False, find_unused_parameters=False)
        #         # process_group = torch.distributed.new_group()
        #         # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
        #         # self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
        #         #                 broadcast_buffers=False, find_unused_parameters=False)
        #     else:    
        #         self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
        #     if self.rank == 0:
        #         logger.info(self.d_model)
        #         logger.info(f"init {args.d_name} success")
        #         if args.stat == "wandb":
        #             wandb.watch(self.d_model)
        #     self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
        #     self.opt_d_s = create_scheduler(args, self.opt_d)
           
        # if args.e_name is not None:
        #     """
        #     bugs on DDP training using eval_model, using additional eval_copy for evaluation 
        #     """
        #     eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
        #     # eval copy is for single card evaluation
        #     if self.args.ddp:
        #         self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.rank)
        #         self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank) 
        #     else:
        #         self.eval_model = getattr(eval_model_module, args.e_name)(args)
        #         self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                
        #     #if self.rank == 0:
        #     other_tools.load_checkpoints(self.eval_copy, args.data_path+args.e_path, args.e_name)
        #     other_tools.load_checkpoints(self.eval_model, args.data_path+args.e_path, args.e_name)
        #     if self.args.ddp:
        #         self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model, process_group)   
        #         self.eval_model = DDP(self.eval_model, device_ids=[self.rank], output_device=self.rank,
        #                               broadcast_buffers=False, find_unused_parameters=False)
        #     self.eval_model.eval()
        #     self.eval_copy.eval()
        #     if self.rank == 0:
        #         logger.info(self.eval_model)
        #         logger.info(f"init {args.e_name} success")  
        #         if args.stat == "wandb":
        #             wandb.watch(self.eval_model) 
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()
        
        self.alignmenter = metric.alignment(0.3, 7, self.train_data.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None
       
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    # def inverse_selection_6d(self, filtered_t, selection_array, n):
    #     original_shape_t = np.zeros((n, selection_array.size))
    #     selected_indices = np.where(selection_array == 1)[0]
    #     new_selected_indices = np.zeros((n, selected_indices.size*2))
    #     new_selected_indices[:, ::2] = selected_indices
    #     new_selected_indices[:, 1::2] = selected_indices 
    #     selected_indices = new_selected_indices.astype(np.bool)
    #     for i in range(n):
    #         original_shape_t[i, selected_indices] = filtered_t[i]
    #     return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 165)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 165), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def inverse_selection_tensor_6d(self, filtered_t, selection_array, n):
        new_selected_array = np.zeros((330))
        new_selected_array[::2] = selection_array
        new_selected_array[1::2] = selection_array 
        selection_array = new_selected_array
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 330)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 330), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  "%(epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += "{}: {:.3f}\t".format(name, metric.avg)
                self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({name: metric.avg}, step=epoch*self.train_length+its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'glr': lr_g}, step=epoch*self.train_length+its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'dlr': lr_d}, step=epoch*self.train_length+its)
        pstr += "dtime: %04d\t"%(t_data*1000)        
        pstr += "ntime: %04d\t"%(t_train*1000)
        pstr += "mem: {:.2f} ".format(mem_cost*len(self.args.gpus))
        logger.info(pstr)
     
    def val_recording(self, epoch):
        pstr_curr = "Curr info >>>>  "
        pstr_best = "Best info >>>>  "
        for name, states in self.tracker.loss_meters.items():
            metric = states['val']
            if metric.count > 0:
                pstr_curr += "{}: {:.3f}     \t".format(name, metric.avg)
                if epoch != 0:
                    if self.args.stat == "ts":
                        self.writer.add_scalars(f"val/{name}", {name+"_val":metric.avg, name+"_train":states['train'].avg}, epoch*self.train_length)
                    else:
                        wandb.log({name+"_val": metric.avg, name+"_train":states['train'].avg}, step=epoch*self.train_length)
                    new_best_train, new_best_val = self.tracker.update_and_plot(name, epoch, self.checkpoint_path+f"{name}_{self.args.name+self.args.notes}.png")
                    if new_best_val:
                        other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
        for k, v in self.tracker.values.items():
            metric = v['val']['best']
            if self.tracker.loss_meters[k]['val'].count > 0:
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, metric['value'], metric['epoch'])
        logger.info(pstr_curr)
        logger.info(pstr_best)
   
    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)

@logger.catch
def main_worker(rank, world_size, args):
    #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # rank = torch.distributed.get_rank()
    # world_size = torch.distributed.get_world_size()
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
    logger.info("Training from scratch ...")
    start_time = time.time()
    for epoch in range(args.epochs+1):
        # if args.ddp: trainer.val_loader.sampler.set_epoch(epoch)
        # trainer.val(epoch)
        # if (epoch) % args.test_period == 1: trainer.val(epoch)
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))
        if epoch != args.epochs:
            if args.ddp: trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            trainer.train(epoch)
        # if args.debug:
        #     other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
        #     other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), args.g_name)
        #     #other_tools.load_checkpoints(trainer.model, "/home/s24273/datasets/hub/pretrained_vq/last_140.bin", args.g_name)
        #     trainer.test(epoch)
        if (epoch) % args.test_period == 0 and epoch !=0:
            if rank == 0:
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
                # trainer.test(epoch)
       
    if rank == 0:
        for k, v in trainer.tracker.values.items():
            if trainer.tracker.loss_meters[k]['val'].count > 0:
                other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"{k}.bin"), args.g_name)
                logger.info(f"inference on ckpt {k}_val_{v['val']['best']['epoch']}:")
                trainer.test(v['val']['best']['epoch'])
        other_tools.record_trial(args, trainer.tracker)
        # wandb.log({"fid_test": trainer.tracker["fid"]["test"]["best"]})
        if args.stat == "ts":
            trainer.writer.close()
        else:
            wandb.finish()
    
            
if __name__ == "__main__":
    # os.environ["MASTER_ADDR"]='127.0.0.1'
    # port = int(random.random() * 100) + 8600
    # os.environ["MASTER_PORT"]=str(port)
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    args = config.parse_args()
    if args.ddp:
        # mp.set_start_method("spawn", force=True)
        # mp.spawn(
        #     main_worker,
        #     args=(len(args.gpus), args,),
        #     nprocs=len(args.gpus),
        #         )
        world_size = len(args.gpus)
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
        # main_worker(0, 1, args)
    else:
        main_worker(0, 1, args)
        