import copy
import os
import os.path
from typing import Optional, Union

import numpy as np
import torch

from .base_dataset import BaseMotionDataset
from .builder import DATASETS

from .EMAGE_2024.dataloaders.beat_motionx import CustomDataset

import yaml
from addict import Dict
from tqdm import tqdm

@DATASETS.register_module()
class SpeechMotionDataset(BaseMotionDataset):
    """SpeechMotion dataset.

    Args:
        speech_dir (str): Path to the directory containing the speech files.
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
                 tcomb_mode: Optional[bool] = False,
                 
                 ann_config: Optional[Union[str, None]] = None,

                 ):


        
        self.ann_config = ann_config

        super(SpeechMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=pipeline,
                                                dataset_name=dataset_name,
                                                fixed_length=fixed_length,
                                                ann_file=ann_file,
                                                motion_dir=motion_dir,
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def load_anno(self, name):
        raise NotImplementedError

    def load_annotations(self):
        """Load annotations from ``ann_file`` to ``data_infos``"""
        
        
        with open(self.ann_config, 'r') as file:
            self.s2g_args = Dict(yaml.safe_load(file))
        
        temp_data_infos = CustomDataset(self.s2g_args, self.ann_file.split('/')[-1].split('.')[0])
        self.data_infos = []
        for i in tqdm(range(len(temp_data_infos))):
            results = {}
            
            results['text'] = [
                [temp_data_infos.lang_model.index2word[int(item)] for item in temp_data_infos[i]['word']]
            ]

            
            unique_list_str = []
            for item in results['text'][0]:
                if item not in unique_list_str and item != '':  
                    unique_list_str.append(item)

            
            results['text'][0] = 'A person is doing a speech, and the speech content is ' + \
                                    ' '.join(unique_list_str)

            results['motion'] = np.zeros((temp_data_infos[i]['pose'].shape[0], 322))
            results['motion'][:, :3+63] = temp_data_infos[i]['pose'][:, :3+63]             
            results['motion'][:, 66:66+90] = temp_data_infos[i]['pose'][:, 66+9:66+90+9]   
            results['motion'][:, 66+90:66+93] = temp_data_infos[i]['pose'][:, 66:66+3]     
            results['motion'][:, 209:209+100] = temp_data_infos[i]['facial']               
            results['motion'][:, 309:309+3] = temp_data_infos[i]['trans']

            # results['facial'] = temp_data_infos[i]['facial']
            # results['pose'] = temp_data_infos[i]['pose']
            # results['trans'] = temp_data_infos[i]['trans']
            results['contact'] = self.contacts_emage(results)
            results['c'] =  np.array(temp_data_infos[i]['audio'])
            results['dataset_name'] = self.dataset_name
            self.data_infos.append(results)
        del temp_data_infos

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        text_list = results['text']
        idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[idx]
        if 'clip_feat' in results.keys():
            results['clip_feat'] = results['clip_feat'][idx]
        results['dataset_name'] = self.dataset_name   
        results = self.pipeline(results)
        # results = self.trans_smplx322_emage(results)
        return results
    def prepare_evaluation(self, ):
        raise NotImplementedError

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
        batch_size = 1024  # 根据显存大小调整批量大小
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

    def trans_smplx322_emage(self, results):
        motion = results['motion']
        # dataset_name = results['dataset_name']
        contacts = results['contact']
        emage_motion = np.zeros((motion.shape[0], 169))
        emage_motion[:, :3+63] = motion[:, :3+63]             
        emage_motion[:, 66+9:66+90+9] = motion[:, 66:66+90]
        emage_motion[:, 66:66+3] = motion[:, 66+90:66+93]
        # max_length = 128
        trans = motion[:, 309:309+3]
        exps = motion[:, 209:209+100]
        emage_motion[:, :-4] = contacts
        results['pose'] = emage_motion
        results['trans'] = trans
        results['facial'] = exps
        
        return results