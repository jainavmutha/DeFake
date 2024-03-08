# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.
'''
import sys
sys.path.append('.')

import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from dataset.albu import IsotropicResize


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list = self.collect_img_and_label_for_one_dataset(one_data)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list
                    
        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        
    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans
       
    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []

        # Try to get the dataset information from the JSON file
        try:
            with open(os.path.join('/Users/jainavmutha/DeepfakeBench/datasets/FaceForensics++.json'), 'r') as f:
                dataset_info = json.load(f)
                
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        cp = None
        if dataset_name == 'FaceForensics++':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        # Get the information for the current dataset
        for tt in dataset_info[dataset_name]:
            for stream in dataset_info[dataset_name][tt]:
                sub_dataset_info = dataset_info[dataset_name][tt][stream][self.mode]
                #for video in dataset_info[dataset_name][label][stream][self.mode]:
                    #sub_dataset_info = dataset_info[dataset_name][label][stream][self.mode][video]

            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            #if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                #sub_dataset_info = sub_dataset_info[self.compression]
            #elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                #sub_dataset_info = sub_dataset_info['c40']
            # Iterate over the videos in the dataset
                for video_info_key, video_info_value in sub_dataset_info.items():
                        print(video_info_value )
                        label = self.config['label_dict'][tt]
                        frame_paths = video_info_value['paths']

                        # Select self.frame_num frames evenly distributed throughout the video
                        total_frames = len(frame_paths)
                        if self.frame_num < total_frames:
                            step = total_frames // self.frame_num
                            selected_frames = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                            # Append the label and frame paths to the lists according the number of frames
                            label_list.extend([label]*len(selected_frames))
                            frame_path_list.extend(selected_frames)
                        else:
                            label_list.extend([label]*total_frames)
                            frame_path_list.extend(frame_paths)
            
        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list))
        random.shuffle(shuffled)
        label_list, frame_path_list = zip(*shuffled)
        
        return frame_path_list, label_list

     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None: 
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
            mask = cv2.resize(mask, (size, size))/255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((size, size, 1))

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
            return np.float32(landmark)
        else:
            return np.zeros((81, 2))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        # Get the mask and landmark paths
        mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark
        
        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation
        
        # Load mask and landmark (if needed)
        if self.config['with_mask']:
            mask = self.load_mask(mask_path)
        else:
            mask = None
        if self.config['with_landmark']:
            landmarks = self.load_landmark(landmark_path)
        else:
            landmarks = None

        # Do Data Augmentation
        if self.mode=='train' and self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)
        
        return image_trans, label, landmarks_trans, mask_trans
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None
        
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/home/zhiyuanyan/disfin/deepfake_benchmark/training/config/detector/xception.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config = config,
                mode = 'train', 
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
        
'''
import os
import math
import yaml
import glob
import json
import pandas as pd

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class IsotropicResize(A.DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.resize(img, (size, int(h / w * size)), interpolation=interpolation_up)
    else:
        img = cv2.resize(img, (int(w / h * size), size), interpolation=interpolation_up)
    return cv2.resize(img, (size, size), interpolation=interpolation_down)

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, image_file_path, transform=None, max_side=256):
        self.root_dir = root_dir
        self.image_paths = pd.read_csv(image_file_path, header=None).squeeze("columns").tolist()
        self.transform = transform
        self.max_side = max_side
        self.image_count = 0  # Counter variable

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        label = 1 if 'real' in self.image_paths[idx] else 0
        image = Image.open(image_path).convert("RGB")

        # Increment the counter
        self.image_count += 1

        # Dynamically resize the image
        img_array = np.array(image)
        img_array_before_transform = img_array.copy()  # Make a copy for comparison later
        img_array = isotropically_resize_image(img_array, size=self.max_side)

        if self.transform:
            augmented = self.transform(image=img_array)
            img_array = augmented['image']

        return {'image': img_array, 'label': label}

class DeepfakeAbstractBaseDataset(data.Dataset):
    def __init__(self, root_dir, image_file_path, transform=None, config=None, mode='test'):
        self.root_dir = root_dir
        self.image_paths = pd.read_csv(image_file_path, header=None).squeeze("columns").tolist()
        self.transform = transform
        self.max_side = 256
        self.config = config
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def load_rgb(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.max_side, self.max_side), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        if file_path is None:
            return np.zeros((self.max_side, self.max_side, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((self.max_side, self.max_side))
            mask = cv2.resize(mask, (self.max_side, self.max_side))/255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((self.max_side, self.max_side, 1))

    def load_landmark(self, file_path):
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
            return np.float32(landmark)
        else:
            return np.zeros((81, 2))

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img):
        kwargs = {'image': img}
        
        transformed = self.transform(**kwargs)
        
        augmented_img = transformed['image']

        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_paths[index])
        label = 1 if 'real' in self.image_paths[index] else 0

        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)

        mask_path = image_path.replace('frames', 'masks')
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        if self.mode == 'test' and self.config['use_data_augmentation']:
            image_trans = self.data_aug(image)
        else:
            image_trans= deepcopy(image)

        image_trans = self.normalize(self.to_tensor(image_trans))
        
        return image_trans, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

       

        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        
        return data_dict

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

# Usage:
# Load the configuration file
config_path = '/Users/jainavmutha/DeepfakeBench/training/config/detector/ucf.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create Albumentations transformation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=[-10, 10], p=0.5),
    A.GaussianBlur(blur_limit=[3, 7], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1], p=0.5),
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
    ToTensorV2()
])

# Create custom dataset and dataloader
custom_dataset = DeepfakeAbstractBaseDataset(root_dir='/Users/jainavmutha/DeepfakeBench/datasets/', 
                                       image_file_path='/Users/jainavmutha/DeepfakeBench/training/copied_image_paths.csv', 
                                       transform=transform, 
                                       config=config,
                                       mode='train')

custom_data_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=8,
                                                 collate_fn=custom_dataset.collate_fn)

# Iterate through the DataLoader
for iteration, batch in enumerate(custom_data_loader):
    images = batch['image']
    labels = batch['label']
    # Process your batches here
print('DONEhhhhhhh')
    # ...
