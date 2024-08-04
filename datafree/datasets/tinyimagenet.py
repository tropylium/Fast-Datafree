from torchvision.datasets import VisionDataset
import torch
from typing import Optional, Callable, Any, Tuple, cast
import os
from torchvision.transforms import v2
import json

class TinyImageNet(VisionDataset):
    TRAIN_IMAGES = "train-images.pt"
    TRAIN_LABELS = "train-labels.pt"
    TEST_IMAGES = "val-images.pt"
    TEST_LABELS = "val-labels.pt"

    def __init__(
        self, 
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(
            root, 
            transform=transform,
            target_transform=target_transform,
        )

        self.train = train

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")
        
        self.images, self.targets = self._load_data()
        self.to_images = v2.ToPILImage()

        with open(os.path.join(root, "labels_info.json")) as f:
            self.idx_to_description = json.load(f)

    def _check_exists(self) -> bool:
        return all(
            os.path.isfile(os.path.join(self.root, file))
            for file in [self.TRAIN_IMAGES, self.TRAIN_LABELS, self.TEST_IMAGES, self.TEST_LABELS]
        )

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.load(os.path.join(self.root, 
            self.TRAIN_IMAGES if self.train else self.TEST_IMAGES))
        labels = torch.load(os.path.join(self.root, 
            self.TRAIN_LABELS if self.train else self.TEST_LABELS))

        assert(isinstance(images, torch.Tensor))
        assert(isinstance(labels, torch.Tensor))

        return images, labels

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        img = self.images[index, ...]
        img = self.to_images(img)
        target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target