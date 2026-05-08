
import numpy as np
import matplotlib.pyplot as plt
from deepbench.astro_object import StarObject, GalaxyObject 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from typing import TYPE_CHECKING
import torchvision 

if TYPE_CHECKING:
    from collections.abc import Callable

class SkyDataset(Dataset):
    """PyTorch Dataset for generating sky images."""
    def __init__(self, n_samples:int, image_size: int=28, shuffle:bool=False, seed:int=42):
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.image_size = image_size
        self.seed = seed
        if not hasattr(self, 'noise_level'):
            self.noise_level = 0.05  # Default noise level, can be overridden in subclasses

        # Generate all data upfront
        self.rng = np.random.default_rng(seed=seed)
        self.labels = self.decide_labels()
        self.images = self._generate_all_images()
    
    def _generate_all_images(self):
        """Generate all images once and store them."""
        images = []
        for label in self.labels:
            images.append(self.generate_image(label))
        return np.stack(images)
    
    def decide_labels(self):
        """Override in subclasses to change label distribution."""
        n_stars = (self.rng.integers(low=int(.45*self.n_samples), high=int(.65*self.n_samples)))
        n_galaxies = self.n_samples - n_stars
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)
    
    def generate_image(self, label:int):
        """Override in subclasses to change image generation."""
        radius = self.rng.integers(low=1, high=self.image_size // 2)
        center_x = self.rng.integers(low=1, high=self.image_size)
        center_y = self.rng.integers(low=1, high=self.image_size)
        
        if label == 0:
            image = StarObject(
                image_dimensions=(self.image_size, self.image_size),
                noise_level=self.noise_level,
                radius=radius
            ).create_object(
                center_x=center_x, center_y=center_y
            )
        else:
            image = GalaxyObject(
                image_dimensions=(self.image_size, self.image_size),
                noise_level=self.noise_level,
                radius=radius
            ).create_object(
                center_x=center_x, center_y=center_y
            )
        
        return image
    
    def get(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__getitem__(idx)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # Add channel dimension if needed (for grayscale images)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        return image, label


class SkyGenerator:
    """Factory class that returns PyTorch DataLoader for SkyDataset."""
    def __init__(self, n_samples: int, dataset, shuffle: bool=False, batch_size: int=64, seed: int=42, transform: "Callable | None"=None):
        self.dataset = dataset(n_samples, seed=seed, transform=transform, shuffle=shuffle)
        self.shuffle = shuffle
        self.batch_size = batch_size
    
    def get_dataloader(self):
        """Return a PyTorch DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
    
    def __len__(self):
        return len(self.dataset)

class TestDataset(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=64, seed=seed, shuffle=shuffle)


class SkyDataset01(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        
        self.size_options = [i for i in range(10, 30)]
        self.random_augment = torchvision.transforms.v2.RandomApply([
            torchvision.transforms.v2.CenterCrop(size=16),
            torchvision.transforms.v2.ColorJitter(brightness=0.5),
            torchvision.transforms.v2.ColorJitter(contrast=0.5),
            torchvision.transforms.v2.ColorJitter(saturation=0.5),
        ])
        super().__init__(n_samples, image_size=64, seed=seed, shuffle=shuffle)

    def generate_image(self, label):
        image_size = int(self.image_size + self.rng.choice(self.size_options))

        radius = self.rng.integers(low=1, high=image_size // 2)
        center_x = self.rng.integers(low=1, high=image_size)
        center_y = self.rng.integers(low=1, high=image_size)
        
        if label == 0:
            image = StarObject(
                image_dimensions=(image_size, image_size),
                noise_level=self.noise_level,
                radius=radius
            ).create_object(
                center_x=center_x, center_y=center_y
            )
        else:
            image = GalaxyObject(
                image_dimensions=(image_size, image_size),
                noise_level=self.noise_level,
                radius=radius
            ).create_object(
                center_x=center_x, center_y=center_y
            )
        img = torch.from_numpy(image).float().unsqueeze(dim=0)
        transformed_image = torchvision.transforms.v2.Resize(self.image_size)(self.random_augment(img))
        return transformed_image

class SkyDataset02(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=64, seed=seed, shuffle=shuffle)
    
    def decide_labels(self):
        n_stars = (self.rng.integers(low=int(.85*self.n_samples), high=int(.95*self.n_samples)))
        n_galaxies = self.n_samples - n_stars
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)


class SkyDataset03(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=28, seed=seed, shuffle=shuffle)

    def decide_labels(self):
        n_stars = int((self.rng.integers(low=int(.2*self.n_samples), high=int(.3*self.n_samples))))
        n_galaxies = int(.5 * (self.n_samples - n_stars))
        wild_card = self.n_samples - (n_stars + n_galaxies)
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)] + [2 for _ in range(wild_card)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)
    
    def generate_image(self, label):
        if label == 2:
            return torch.randn((self.image_size, self.image_size)) * self.noise_level  # Generate random noise for wildcard class
        else:
            return super().generate_image(label)
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        # Convert label 2 (wildcard) to label 0 (Star)
        if self.rng.random() < 0.5:
          convert = torch.tensor(0)
        else:
          convert = torch.tensor(1)
        label = torch.where(label == 2, convert, label)
        return image, label


class SkyDataset04(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=28, seed=seed, shuffle=shuffle)

    def decide_labels(self):
        n_stars = (self.rng.integers(low=int(.45*self.n_samples), high=int(.65*self.n_samples)))
        n_galaxies = (self.rng.integers(low=int(.6*(self.n_samples - n_stars)), high=int(.9*(self.n_samples - n_stars))))
        wild_card = self.n_samples - (n_stars + n_galaxies)
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)] + [2 for _ in range(wild_card)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image.numpy(), label.numpy()


class SkyDataset05(SkyDataset):
    def __init__(self, n_samples:int, image_size: int=28, shuffle:bool=False, seed:int=42):
        self.noise_level = 0.25
        self.regression_labels = torch.zeros(n_samples, 3, dtype=torch.float32)
        super().__init__(n_samples, image_size=image_size, shuffle=shuffle, seed=seed)

    def _generate_all_images(self):
        """Generate all images once and store them."""
        images = []
        for index, label in enumerate(self.labels):
            image, reg_label = self.generate_image(label)
            images.append(image)
            self.regression_labels[index] = reg_label
        return np.stack(images)
    
    def generate_image(self, label:int):
        """Override in subclasses to change image generation."""
        radius = self.rng.integers(low=1, high=self.image_size // 2)
        center_x = self.rng.integers(low=1, high=self.image_size)
        center_y = self.rng.integers(low=1, high=self.image_size)
        regression_label = torch.tensor([center_x, center_y, radius], dtype=torch.float32)

        if label == 0:
            image = StarObject(
                image_dimensions=(self.image_size, self.image_size),
                noise_level=self.noise_level,
                radius=radius
            ).create_object(
                center_x=center_x, center_y=center_y
            )
        else:
            image = GalaxyObject(
                image_dimensions=(self.image_size, self.image_size),
                noise_level=self.noise_level,
                radius=radius
            ).create_object(
                center_x=center_x, center_y=center_y
            )

        return image, regression_label
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.regression_labels[idx].to(torch.float32)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        
        # Add channel dimension if needed (for grayscale images)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        return image, label

class Eval: 
    @staticmethod
    def plot_loss_history(train_loss, val_loss): 
        epochs = range(len(train_loss))


        plt.plot(epochs, train_loss, label="Train", marker='o')
        plt.plot(epochs, val_loss, label='Validation', marker='x')

        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def ROC_curve(prediction_classes, labels): 
        score_fpr, score_tpr, _ = roc_curve(labels, prediction_classes)
        plt.plot(score_fpr, score_tpr, label='Your classifier')
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='black', label='Random Classifier')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC AUC Curve")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def confusion_matrix(prediction_classes, labels): 
        confusion = confusion_matrix(labels.ravel(), prediction_classes.ravel())
        plt.imshow(confusion)

        for true in range(confusion.shape[0]):
            for predicted in range(confusion.shape[1]):
                plt.text(predicted, true, confusion[true, predicted],
                            ha="center", va="center", fontdict={
                                "color":"white", 
                                "backgroundcolor":"black", 
                                "size": 5})

        plt.xticks([0, 1], labels=["Star", "Galaxy"])
        plt.yticks([0, 1], labels=["Star", "Galaxy"])  
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
