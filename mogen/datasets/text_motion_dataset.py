import copy
import os
import os.path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm 

from .base_dataset import BaseMotionDataset
from .builder import DATASETS
from ..models.utils.quaternion import ax_from_6v, ax_to_6v

@DATASETS.register_module()
class TextMotionDataset(BaseMotionDataset):
    """TextMotion dataset.

    Args:
        text_dir (str): Path to the directory containing the text files.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 text_dir: Optional[Union[str, None]] = None,
                 token_dir: Optional[Union[str, None]] = None,
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,
                 siamese_mode: Optional[bool] = False,
                 tcomb_mode: Optional[bool] = False):
        self.text_dir = os.path.join(data_prefix, 'datasets', dataset_name,
                                     text_dir)
        if token_dir is not None:
            self.token_dir = os.path.join(data_prefix, 'datasets',
                                          dataset_name, token_dir)
        else:
            self.token_dir = None
        if clip_feat_dir is not None:
            self.clip_feat_dir = os.path.join(data_prefix, 'datasets',
                                              dataset_name, clip_feat_dir)
        else:
            self.clip_feat_dir = None
        self.siamese_mode = siamese_mode
        self.tcomb_mode = tcomb_mode
        super(TextMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=pipeline,
                                                dataset_name=dataset_name,
                                                fixed_length=fixed_length,
                                                ann_file=ann_file,
                                                motion_dir=motion_dir,
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def load_anno(self, name):
        results = {}
        if self.siamese_mode:
            motion_path = os.path.join(self.motion_dir, name + '.npz')
            motion_data = np.load(motion_path)
            results['motion1'] = motion_data['motion1']
            results['motion2'] = motion_data['motion2']
            assert results['motion1'].shape == results['motion2'].shape
        else:
            motion_path = os.path.join(self.motion_dir, name + '.npy')
            motion_data = np.load(motion_path)
            results['motion'] = motion_data
        text_path = os.path.join(self.text_dir, name + '.txt')
        text_data = []
        for line in open(text_path, 'r'):
            text_data.append(line.strip())
        if len(text_data) == 0:
            print(text_path)
            text_data = [' ']
        results['text'] = text_data
        if self.token_dir is not None:
            token_path = os.path.join(self.token_dir, name + '.txt')
            token_data = []
            for line in open(token_path, 'r'):
                token_data.append(line.strip())
            results['token'] = token_data
        if self.clip_feat_dir is not None:
            clip_feat_path = os.path.join(self.clip_feat_dir, name + '.npy')
            clip_feat = torch.from_numpy(np.load(clip_feat_path))
            results['clip_feat'] = clip_feat
        results['dataset_name'] = self.dataset_name
        
        results['contact'] = self.contacts_emage(results)
        
        return results

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        text_list = results['text']
        idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[idx]
        if 'clip_feat' in results.keys():
            results['clip_feat'] = results['clip_feat'][idx]
        if 'token' in results.keys():
            results['token'] = results['token'][idx]
        results['dataset_name'] = self.dataset_name
        results = self.pipeline(results)
        return results

    def contacts_emage(self, results):
        motion = results['motion']
        device = self.smplx.pose_mean.device
        # dataset_name = results['dataset_name']
        import torch
        emage_motion = torch.zeros((motion.shape[0], 169),device=device)
        emage_motion[:, :3+63] = torch.Tensor(motion[:, :3+63]).to(device)
        emage_motion[:, 66+9:66+90+9] = torch.Tensor(motion[:, 66:66+90]).to(device)
        emage_motion[:, 66:66+3] = torch.Tensor(motion[:, 66+90:66+93]).to(device)
        # max_length = 128
        trans = torch.Tensor(motion[:, 309:309+3]).to(device)
        #print(n, s, r)
        exps = torch.Tensor(motion[:, 209:209+100]).to(device)
        all_tensor = []
        betas=torch.zeros((motion.shape[0], 300),device=device).float()
        # for i in range(s):
        batch_size = 4096  # 根据显存大小调整批量大小
        num_batches = (motion.shape[0] + batch_size - 1) // batch_size

        all_tensor = []
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, motion.shape[0])
                
                betas_batch = betas[start_idx:end_idx]
                trans_batch = trans[start_idx:end_idx].float()
                exps_batch = exps[start_idx:end_idx].float()
                emage_motion_batch = emage_motion[start_idx:end_idx]
                
                joints_batch = self.smplx(
                    betas=betas_batch,
                    transl=trans_batch, 
                    expression=exps_batch, 
                    jaw_pose=emage_motion_batch[..., 66:69].float(), 
                    global_orient=emage_motion_batch[...,:3].float(), 
                    body_pose=emage_motion_batch[...,3:21*3+3].float(), 
                    left_hand_pose=emage_motion_batch[...,25*3:40*3].float(), 
                    right_hand_pose=emage_motion_batch[...,40*3:55*3].float(), 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=torch.zeros_like(emage_motion_batch[..., 69:72]).float(), 
                    reye_pose=torch.zeros_like(emage_motion_batch[..., 72:75]).float(),
                )['joints'][:, (7,8,10,11), :].reshape(end_idx - start_idx, 4, 3).cpu()
                
                all_tensor.append(joints_batch)
        joints = torch.cat(all_tensor, axis=0) # all, 4, 3
        # print(joints.shape)
        feetv = torch.zeros(joints.shape[1], joints.shape[0])
        joints = joints.permute(1, 0, 2)
        #print(joints.shape, feetv.shape)
        feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
        #print(feetv.shape)
        contacts = (feetv < 0.01).numpy().astype(float)
        # print(contacts.shape, contacts)
        contacts = contacts.transpose(1, 0)
        return contacts
