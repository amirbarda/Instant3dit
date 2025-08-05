import os
#import s3fs

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import json

#from flash_s3_dataloader.s3_io  import load_s3_image, load_s3_json, _get_parent_basename
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import imageio
from scipy.ndimage import gaussian_filter
import cv2
from pathlib import Path

class MaskeObjaverseDataset(Dataset):
    """
    A dataset to prepare the objaverse images and their prompts
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_root,
        # tokenizer,
        s3_dirs_filename,
        masks_root,
        blurring_sigma_range,
        batch_size,
        excluded_mask_types = [],
        captions_filename='filtered_cap3d.csv',
        view_type='offset0',
        size=1024,
        prompt_suffix="",
        prompt_prefix="",
        inconsistent_mask_mode = False,
        validation = False,
        validation_size = 100,
        debug_idx = -1,
    ):
        
        self.validation = validation
        self.size = size
        self.batch_size = batch_size
        self.prompt_suffix = prompt_suffix
        self.prompt_prefix = prompt_prefix
        self.inconsistent_mask_mode = inconsistent_mask_mode

        self.blurring_sigma_range = blurring_sigma_range 
        self.data_root = data_root
        self.mask_type_dict = {'type3': 0, 'type2' : 1, 'type1' : 2}

        print('Loading image dirs...')
        with open(os.path.join(data_root, s3_dirs_filename)) as f:
            self.paths = [x.strip() for x in f]
        print('Loaded image dirs.')

        caption_path = os.path.join(data_root, captions_filename)
        print(f'Loading captions from {caption_path}...')
        caption_dict = {}
        for line in open(caption_path, 'r', encoding='utf-8-sig'):
            line = line.strip().split(',')
            caption_dict[line[0]] = ','.join(line[1:])
        print('Loaded captions.')
        
        f = open(os.path.join(data_root, masks_root), 'r')
        self.mask_dict = json.load(f)

        print('Filtering valid paths...')
        self.paths = [x for x in self.paths if self.get_objaverse_id_from_s3_path(x) in caption_dict and self.get_objaverse_full_id_from_s3_path(x) in self.mask_dict] 

        if debug_idx < 0:
            if validation:
                self.paths = self.paths[:validation_size]
            else:
                self.paths = self.paths[validation_size:]
        else:
            self.paths = [self.paths[debug_idx]]

        self.captions = [caption_dict[self.get_objaverse_id_from_s3_path(x)] for x in self.paths]
        self.mask_dict = {k : v for (k,v) in self.mask_dict.items() if self.make_s3_path_from_id(k) in self.paths or k=='root'}
        
        print('Filtering done.')

        print(f'Number of objects: {len(self.paths)}')
        self.image_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.mask_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0], [1])]
        )
        mask_root = self.mask_dict['root']
        self.mask_paths = []
        for k,key_mask_list in self.mask_dict.items():
            if k == 'root':
                continue
            for key_mask in key_mask_list:
                masks_ids = key_mask['masks']
                if key_mask['mask_type'] in excluded_mask_types:
                    continue
                for i in masks_ids:
                    self.mask_paths.append(os.path.join(mask_root, key_mask['mask_type'], k, f'masks_{i}'))


        if validation:
            random.Random(42).shuffle(self.mask_paths)

        self.caption_dict = {}
        for i in range(len(self.captions)):
            model_name = self.get_objaverse_full_id_from_s3_path(self.paths[i])
            self.caption_dict[model_name] = self.captions[i]

        # check data integrity:
        for i,mask_path in enumerate(self.mask_paths):
            assert self.get_objaverse_full_id_from_mask_path(mask_path)[0] in self.caption_dict

        self.view_type = view_type
        assert self.view_type in ['random', 'offset0']
        self.num_views = 16

    def get_objaverse_id_from_s3_path(self, path):
        path = path.strip('/').split('/')
        return path[-2]
    
    def get_objaverse_full_id_from_s3_path(self,path: str):
        model_name = os.path.join(*path.split('/')[2:4])
        return model_name

    def make_s3_path_from_id(self,uid):
        return os.path.join(self.data_root, 'renders', uid, 'renderings')

        
    def get_objaverse_full_id_from_mask_path(self,path):
        split_path = path.split('/')
        model_name = os.path.join(*split_path[3:5])
        mask_type = path.split('/')[5]
        mask_index = split_path[-1].split('_')[1]
        return model_name, mask_type, mask_index
    
    def get_mask_type_from_mask_path(self, mask_path):
        mask_type = os.path.join(mask_path.split('/')[2])
        return mask_type

    def __len__(self):
        
        return len(self.mask_paths)

    def __getitem__(self, index):
        
        example = {}

        index = index % len(self.mask_paths)

        # go over all the masks 

        if self.view_type == 'random':
            starting_view_idx = float(np.random.randint(0, self.num_views))
            view_indices = [int(starting_view_idx + self.num_views // 4 * i) % self.num_views for i in range(4)]
        elif self.view_type == 'offset0':
            view_indices = [int(self.num_views // 4 * i) % self.num_views for i in range(4)]
        else:
            raise NotImplementedError

        model_name, mask_type, mask_index = self.get_objaverse_full_id_from_mask_path(self.mask_paths[index])
        mask_index = [mask_index]*4
                
        mask_root_path = Path(self.mask_paths[index]).parent
        mask_filenames = [os.path.join(f'masks_{mask_idx}', f'{view_idx:06d}.png') for (mask_idx,view_idx) in zip(mask_index,view_indices)]
        instance_mask_paths =  [os.path.join(mask_root_path, name) for name in mask_filenames]
        mask_instance_images = [imageio.imread(path)[:,:,:] for path in instance_mask_paths]

        image_filenames = [f'{view_idx:08d}_rgb.png' for view_idx in view_indices]
        instance_image_paths = [os.path.join(self.make_s3_path_from_id(model_name), name) for name in image_filenames]
        # s3 = s3fs.S3FileSystem(anon=True)
        instance_images = [Image.open(path) for path in instance_image_paths]
        #instance_images = [imageio.imread(path)[:,:,:] for path in instance_image_paths]
        for image in instance_images:
            if not image.mode == "RGB":
                assert False
        # tile        
        # instance_images = [np.array(image) for image in instance_images]
        instance_image = np.concatenate([np.concatenate(instance_images[:2], axis=1),
                                         np.concatenate(instance_images[2:], axis=1)], axis=0)
        assert instance_image.shape[-1] == 3
        instance_image = Image.fromarray(instance_image)

        if self.get_mask_type_from_mask_path(instance_mask_paths[0]) == 'volume':
            mask_instance_images = [np.where(gaussian_filter(np.array(image), sigma=np.random.uniform(*self.blurring_sigma_range))>0,255,0) for image in mask_instance_images]
            
        else:
            mask_instance_images = [np.array(image) for image in mask_instance_images]
            
        mask_instance_image = np.concatenate([np.concatenate(mask_instance_images[:2], axis=1),
                                         np.concatenate(mask_instance_images[2:], axis=1)], axis=0)

        mask_instance_image = Image.fromarray(mask_instance_image.astype(np.uint8))


        # resize
        instance_image = instance_image.resize((self.size, self.size), Image.LANCZOS)
        assert instance_image.height == instance_image.width == self.size
        # add to return dict
        example["jpg"] = self.image_transforms(instance_image)

        mask_instance_image = mask_instance_image.resize((self.size, self.size), Image.LANCZOS)
        assert mask_instance_image.height == mask_instance_image.width == self.size
        # add to return dict

        example["mask_type"] = torch.tensor(self.mask_type_dict[self.get_mask_type_from_mask_path(instance_mask_paths[0])])

        example["mask"] = self.mask_transforms(mask_instance_image)

        instance_prompt = self.caption_dict[model_name]#self.captions[index]

        if self.prompt_prefix:
            instance_prompt = self.prompt_prefix + ' ' + instance_prompt
        if self.prompt_suffix:
            instance_prompt = instance_prompt + ' ' + self.prompt_suffix

        example["txt"] = instance_prompt


        example['original_size_as_tuple'] = torch.tensor([instance_image.width, instance_image.height]).float()
        example['target_size_as_tuple'] = example['original_size_as_tuple'].clone()
        example['crop_coords_top_left'] = torch.tensor([0, 0]).float()
        if self.validation:
            example['mask_index'] = self.mask_paths[index]
        return example


def collate_fn(examples):


    batch = {}
    for key in examples[0].keys():
        if isinstance(examples[0][key], torch.Tensor):
            batch[key] = (torch.stack([example[key] for example in examples], dim=0))
        elif isinstance(examples[0][key], str):
            batch[key] = [example[key] for example in examples]
        elif isinstance(examples[0][key], int):
            batch[key] = [example[key] for example in examples]
        else:
            raise NotImplementedError
    return batch


def get_dataloader_from_config(config, debug_idx = -1):
    dataset = MaskeObjaverseDataset(batch_size=config['batch_size'], debug_idx = debug_idx, **config.dataset_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        # num_workers=0,
    )

    val_dataset = MaskeObjaverseDataset( batch_size=config['batch_size'],debug_idx = debug_idx, validation=True, **config.dataset_config)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    return dataloader,val_dataloader
