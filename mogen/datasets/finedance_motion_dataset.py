import copy
import os
import os.path
from typing import Optional, Union

import numpy as np
import torch

from .base_dataset import BaseMotionDataset
from .builder import DATASETS

import json
from tqdm import tqdm

def get_train_test_list(datasplit):
        all_list = []
        train_list = []
        for i in range(1,212):
            all_list.append(str(i).zfill(3))

        if datasplit == "cross_genre":
            test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
            ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]+["130"]
        elif datasplit == "cross_dancer":
            test_list = ['001','002','003','004','005','006','007','008','009','010','011','012','013','124','126','128','130','132']
            ignor_list = ['115','117','119','121','122','135','137','139','141','143','145','147'] + ["116", "118", "120", "123", "202", "159"]+["130"]       
        else:
            raise("error of data split!")
        for one in all_list:
            if one not in test_list:
                if one not in ignor_list:
                    train_list.append(one)
        test_list = [item for item in test_list if item not in ignor_list]
        return train_list, test_list, ignor_list

@DATASETS.register_module()
class FinedanceMotionDataset(BaseMotionDataset):
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
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,
                 
                 datasplit: Optional[Union[str, None]] = None,
                 music_dir: Optional[Union[str, None]] = None,
                 ):
        
        self.datasplit = datasplit
        self.music_dir = os.path.join(data_prefix, 'datasets', dataset_name,
                                     music_dir)
        self.text_dir = os.path.join(data_prefix, 'datasets', dataset_name,
                                     text_dir)
        if clip_feat_dir is not None:
            self.clip_feat_dir = os.path.join(data_prefix, 'datasets',
                                              dataset_name, clip_feat_dir)
        else:
            self.clip_feat_dir = None
        super(FinedanceMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=pipeline,
                                                dataset_name=dataset_name,
                                                fixed_length=fixed_length,
                                                ann_file=ann_file,
                                                motion_dir=motion_dir,
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def load_annotations(self):
        """Load annotations from ``ann_file`` to ``data_infos``"""
        mode = self.ann_file.split('/')[-1].split('.')[0]
        train_list, test_list, ignor_list = get_train_test_list(self.datasplit)
        if mode == 'train':
            datalist= train_list
        else:
            datalist = test_list

        self.data_infos = []
        for l_idx, line in tqdm(enumerate(datalist)):
            self.data_infos.append(self.load_anno(line))

    def load_anno(self, name):
        results = {}
        
        motion_path = os.path.join(self.motion_dir, name + '.npy')
        motion_data_ori = np.load(motion_path)
        motion_data = np.zeros((motion_data_ori.shape[0], 322))
        motion_data[:, :3+63] = motion_data_ori[:, 4+3:4+3+66]
        motion_data[:, 66:66+90] = motion_data_ori[:, 4+3+66:4+3+66+90]
        motion_data[:, 309:309+3] = motion_data_ori[:, 4:4+3]
        
        motion_data[:, 309+1] = motion_data[:, 309+1] + 1.3 

        
        music_path = os.path.join(self.music_dir, name + '.npy')
        music_data = np.load(music_path)

        
        before_offset = 360
        motion_data = motion_data[before_offset:]
        music_data = music_data[before_offset:]
        
        min_all_len = min(motion_data.shape[0], music_data.shape[0])
        results['motion'] = motion_data[:min_all_len]
        results['c'] =  music_data[:min_all_len]
        

        text_path = os.path.join(self.text_dir, name + '.json')
        text_data = []
        with open(text_path, 'r') as file:
            json_text = json.load(file)
            text_data.append(f"A dancer is performing a {json_text['style1']} dance in the {json_text['style2']} style to the rhythm of the {json_text['name']} song.")
        results['text'] = text_data

        if self.clip_feat_dir is not None:
            clip_feat_path = os.path.join(self.clip_feat_dir, name + '.npy')
            clip_feat = torch.from_numpy(np.load(clip_feat_path))
            results['clip_feat'] = clip_feat

        results['dataset_name'] = self.dataset_name
        
        # results = self.trans_smplx322_emage(results)
        results['contact'] = self.contacts_emage(results)
        # results['motion'] = np.hstack([results['motion'],results['contact']])
        return results

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
        # all_tensor = []
        betas=torch.zeros((motion.shape[0], 300),device=device).float()
        # for i in range(s):
        batch_size = 512  # 根据显存大小调整批量大小
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

        # joints = torch.cat(all_tensor, axis=0)
        # with torch.no_grad():
        #     joints = self.smplx(
        #         betas=betas,
        #         transl=trans.float(), 
        #         expression=exps.float(), 
        #         jaw_pose=emage_motion[..., 66:69].float(), 
        #         global_orient=emage_motion[...,:3].float(), 
        #         body_pose=emage_motion[...,3:21*3+3].float(), 
        #         left_hand_pose=emage_motion[...,25*3:40*3].float(), 
        #         right_hand_pose=emage_motion[...,40*3:55*3].float(), 
        #         return_verts=True,
        #         return_joints=True,
        #         leye_pose=torch.zeros_like(emage_motion[..., 69:72]).float(), 
        #         reye_pose=torch.zeros_like(emage_motion[..., 72:75]).float(),
        #     )['joints'][:, (7,8,10,11), :].reshape(motion.shape[0], 4, 3).cpu()
        # all_tensor.append(joints)
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
        import smplx
        smplx_data_path = "/workspace/motion_diffusion/EMAGE/EMAGE"
        smplx.create(
            smplx_data_path+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        emage_motion = np.zeros((motion.shape[0], 169))
        emage_motion[:, :3+63] = motion[:, :3+63]             
        emage_motion[:, 66+9:66+90+9] = motion[:, 66:66+90]
        emage_motion[:, 66:66+3] = motion[:, 66+90:66+93]
        # max_length = 128
        trans = motion[:, 309:309+3]
        #print(n, s, r)
        exps = motion[:, 209:209+100]
        all_tensor = []
        # for i in range(s):
        with torch.no_grad():
            joints = self.smplx(
                betas=torch.zeros(motion.shape[0], 300).cuda(),
                transl=trans, 
                expression=exps, 
                jaw_pose=emage_motion[..., 66:69], 
                global_orient=emage_motion[...,:3], 
                body_pose=emage_motion[...,3:21*3+3], 
                left_hand_pose=emage_motion[...,25*3:40*3], 
                right_hand_pose=emage_motion[...,40*3:55*3], 
                return_verts=True,
                return_joints=True,
                leye_pose=torch.zeros_like(emage_motion[..., 69:72]), 
                reye_pose=torch.zeros_like(emage_motion[..., 72:75]),
            )['joints'][:, (7,8,10,11), :].reshape(motion.shape[0], 4, 3).cpu()
        all_tensor.append(joints)
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
        # contact = 
        emage_motion[:, :-4] = contacts
        results['pose'] = emage_motion
        results['trans'] = trans
        results['facial'] = exps
        
        return results