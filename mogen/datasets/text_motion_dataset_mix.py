import copy
from typing import Optional, Union

import numpy as np

from .base_dataset import BaseMotionDataset
from .builder import DATASETS

@DATASETS.register_module()
class TextMixMotionDataset(BaseMotionDataset):
    """TextMixMotion dataset.
        Args:
    """

    def __init__(self,
                 data_prefix: Optional[Union[str, None]] = 'mix',
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,):

        super(TextMixMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=[],
                                                dataset_name='mix',
                                                fixed_length=None,
                                                ann_file='mix',
                                                motion_dir='mix',
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)
        # import smplx
        # smplx_data_path = "/workspace/motion_diffusion/EMAGE/EMAGE/"
        # self.smplx = smplx.create(
        #     smplx_data_path+"smplx_models/", 
        #     model_type='smplx',
        #     gender='NEUTRAL_2020', 
        #     use_face_contour=False,
        #     num_betas=300,
        #     num_expression_coeffs=100, 
        #     ext='npz',
        #     use_pca=False,
        # ).cpu().eval()

    def merge_datasets(self, merge_datasets):
        self.data_infos = []
        self.pipelines = {}
        for idx, item_dataset in enumerate(merge_datasets):
            try:
                self.pipelines[item_dataset.dataset.dataset_name] = item_dataset.dataset.pipeline
                self.data_infos += item_dataset.dataset.data_infos * item_dataset.times
            except Exception as e:
                print("No Repeated Dataset Wrapper, should only be used in testing!")
                self.pipelines[item_dataset.dataset_name] = item_dataset.pipeline
                self.data_infos += item_dataset.data_infos


    def load_anno(self, name):
        raise NotImplementedError

    def load_annotations(self):
        pass

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = {}
        results['text'] = copy.deepcopy(self.data_infos[idx]['text'])
        results['motion'] = copy.deepcopy(self.data_infos[idx]['motion'])
        results['dataset_name'] = copy.deepcopy(self.data_infos[idx]['dataset_name'])
        results['contact'] = copy.deepcopy(self.data_infos[idx]['contact'])
        text_list = results['text']
        idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[idx]
        # dataset_name = results['dataset_name']
        # if results['dataset_name'] == 'beat2_null':
        #     print(results)
            # results['motion'] = results['motion'][idx]
            # pass
        results = self.pipelines[results['dataset_name']](results)
        # contact = copy.deepcopy(self.data_infos[idx]['contact'])
        results = self.trans_smplx322_emage(results)
        return results
    
    def trans_smplx322_emage(self, results):
        motion = results['motion']
        import torch
        # dataset_name = results['dataset_name']
        # contacts = results['contact']
        # contact = torch.Tensor(contact).to(motion.device)
        contact = results['contact']
        emage_motion = np.zeros((motion.shape[0], 169))
        emage_motion[:, :3+63] = motion[:, :3+63]             
        emage_motion[:, 66+9:66+90+9] = motion[:, 66:66+90]
        emage_motion[:, 66:66+3] = motion[:, 66+90:66+93]
        # max_length = 128
        trans = motion[:, 309:309+3]
        exps = motion[:, 209:209+100]
        emage_motion[:, -4:] = contact
        results['pose'] = emage_motion
        results['trans'] = trans
        results['facial'] = exps
        results['beta'] = torch.zeros((motion.shape[0], 300)).to(motion.device).float()
        del results['motion']
        return results
    
    # def trans_smplx322_emage(self, results):
    #     motion = results['motion']
    #     device = motion.device
    #     # dataset_name = results['dataset_name']
    #     import torch
    #     emage_motion = torch.zeros((motion.shape[0], 169),device=device)
    #     emage_motion[:, :3+63] = motion[:, :3+63]
    #     emage_motion[:, 66+9:66+90+9] = motion[:, 66:66+90]
    #     emage_motion[:, 66:66+3] = motion[:, 66+90:66+93]
    #     # max_length = 128
    #     trans = motion[:, 309:309+3]
    #     #print(n, s, r)
    #     exps = motion[:, 209:209+100]
    #     all_tensor = []
    #     betas=torch.zeros((motion.shape[0], 300),device=device).float()
    #     # for i in range(s):
    #     with torch.no_grad():
    #         joints = self.smplx(
    #             betas=betas,
    #             transl=trans.float(), 
    #             expression=exps.float(), 
    #             jaw_pose=emage_motion[..., 66:69].float(), 
    #             global_orient=emage_motion[...,:3].float(), 
    #             body_pose=emage_motion[...,3:21*3+3].float(), 
    #             left_hand_pose=emage_motion[...,25*3:40*3].float(), 
    #             right_hand_pose=emage_motion[...,40*3:55*3].float(), 
    #             return_verts=True,
    #             return_joints=True,
    #             leye_pose=torch.zeros_like(emage_motion[..., 69:72]).float(), 
    #             reye_pose=torch.zeros_like(emage_motion[..., 72:75]).float(),
    #         )['joints'][:, (7,8,10,11), :].reshape(motion.shape[0], 4, 3).cpu()
    #     all_tensor.append(joints)
    #     joints = torch.cat(all_tensor, axis=0) # all, 4, 3
    #     # print(joints.shape)
    #     feetv = torch.zeros(joints.shape[1], joints.shape[0])
    #     joints = joints.permute(1, 0, 2)
    #     #print(joints.shape, feetv.shape)
    #     feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
    #     #print(feetv.shape)
    #     contacts = (feetv < 0.01).numpy().astype(float)
    #     # print(contacts.shape, contacts)
    #     contacts = contacts.transpose(1, 0)
    #     # contact = 
    #     emage_motion[:, -4:] = torch.from_numpy(contacts).float().to(device)
    #     results['pose'] = emage_motion
    #     results['trans'] = trans
    #     results['facial'] = exps
    #     results['beta'] = betas
    #     del results['motion']
    #     # results['dataset_name'] = dataset_name
    #     return results
    
