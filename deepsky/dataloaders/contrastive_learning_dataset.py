from torchvision.transforms import transforms
from .gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from .view_generator  import ContrastiveLearningViewGenerator
from ..exceptions.exceptions import InvalidDatasetSelection

from glob import glob
from torch.utils.data import Dataset
import torch
import os
from skimage import io, transform
from PIL import Image

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
import torchvision

import cv2
import numpy as np
import os
from os.path import isfile, join
from os import listdir
from ..utils.ellab_utils import create_circular_mask

 

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)



def make_dataset(
    directory ,
    class_to_idx,
    extensions  = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"),
    is_valid_file = None
):    
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

   
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)



    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


def find_classes(directory: str)  :
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class DeepSkyDatasetFolder(Dataset):
    """A generic data loader.
    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root :str ,
        transform : torchvision.transforms = None
        
    ):
  
        self.root = root
        self.transform = transform
        classes, class_to_idx = self.find_classes(root)
        samples = self.make_dataset(root, class_to_idx )
        self.classes = classes
        print("LABELS:{}".format(classes))
        print("LABEL INDICES:{}".format(class_to_idx))
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.itercount = 0

    @staticmethod
    def make_dataset(
        directory,
        class_to_idx 
        
    )  :
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
  
        return make_dataset(directory, class_to_idx )

    def find_classes(self, directory: str)  :
        """Find the class folders in a dataset structured as follows::
            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        Args:
            directory(str): Root directory path, corresponding to ``self.root``
        Raises:
            FileNotFoundError: If ``dir`` has no class folders.
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int)  :
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_name, target = self.samples[index]

        # image = io.imread(img_name)
        # print("loading image{} of class {}".format(img_name, target))
        image = Image.open(img_name)
        # image.save("dataset_samples_runtime/in_test.jpg")
        img = np.array(image)
        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)
        # masked_img = image.copy()
        img[~mask] = 0
        image = Image.fromarray(img)
        self.itercount+=1
        # image = image.resize((256, 256), Image.BILINEAR)
        # image.show()
        sample = image
        if self.transform:
            sample = self.transform(image)
        if(self.itercount==100):
            self.itercount=0
            image.save("./tmp/test.jpg")
            # print(np.transpose(sample.numpy(),(1,2,0)).shape)
            # print(img.shape)
            Image.fromarray((255*np.transpose(sample.numpy(),(1,2,0))).astype(np.uint8)).save("./tmp/test_transform.jpg")
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self):
        return self.targets   



class GRSCD_DatasetFolder(Dataset):
    """A generic data loader.
    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root ,
        transform     :torchvision.transforms = None,

         
    ):

        self.root = root
        self.transform = transform
        classes, class_to_idx = self.find_classes(root)
        samples = self.make_dataset(root, class_to_idx )
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory,
        class_to_idx 
        
    )  :
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
  
        return make_dataset(directory, class_to_idx )

    def find_classes(self, directory: str)  :
        """Find the class folders in a dataset structured as follows::
            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        Args:
            directory(str): Root directory path, corresponding to ``self.root``
        Raises:
            FileNotFoundError: If ``dir`` has no class folders.
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int)  :
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_name, target = self.samples[index]

        # image = io.imread(img_name)
        # print("loading image{} of class {}".format(img_name, target))
        image = Image.open(img_name)
        # image.save("dataset_samples_runtime/in_test.jpg")
        img = np.array(image)
        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)
        # masked_img = image.copy()
        img[~mask] = 0
        image = Image.fromarray(img)
        # image.save("dataset_samples_runtime/test.jpg")
        # image = image.resize((256, 256), Image.BILINEAR)
        # image.show()
        sample = image
        if self.transform:
            sample = self.transform(image)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)



class DeepSky(Dataset):
    """DeepSky dataset."""

    def __init__(self,  root_dir, transform=None):
        """
        Args:
           root_dir: root folder with all images per year/month/day
           Warning: All images are returning a dummy "0" class label
        """
        self.root_dir = root_dir
        self.transform = transform
        print("Scanning folders")
        self.files= glob(root_dir +'/**/*.jpg', recursive=True)
        print("DeepSky dataset found {} images".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        # image = io.imread(img_name)
        print("loading image{}".format(img_name))
        image = Image.open(img_name)
        image.save("dataset_samples_runtime/in_test.jpg")
        img = np.array(image)
        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)
        # masked_img = image.copy()
        img[~mask] = 0
        image = Image.fromarray(img)
        image.save("dataset_samples_runtime/test.jpg")
        image = image.resize((224, 224), Image.BILINEAR)
        image.show()
   
        if self.transform is not None:
            sample = self.transform(image)
            trans = transforms.ToPILImage()
            im1 = out = trans(sample[0])
            im1.save("dataset_samples_runtime/in1.jpg")
            im2 = out = trans(sample[1])
            im2.save("dataset_samples_runtime/in2.jpg")

        
        return sample, 0



class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.05 * s)
        data_transforms = transforms.Compose([
                                              transforms.Resize(224),
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVertical(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'GRSCD': lambda: DeepSkyDatasetFolder(self.root_folder, 
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views)),          

                          'deepsky': lambda: DeepSky(self.root_folder, 
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views)),                                                                                                                  
                                                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
