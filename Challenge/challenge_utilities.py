
import numpy as np
import matplotlib.pyplot as plt
from deepbench.astro_object import StarObject, GalaxyObject 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, confusion_matrix


class SkyDataset(Dataset):
    """PyTorch Dataset for generating sky images."""
    def __init__(self, n_samples, image_size=28, pre_processing=None, shuffle=False, seed=42):
        self.n_samples = n_samples
        self.pre_processing = pre_processing
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
    
    def generate_image(self, label):
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
        
        if self.pre_processing is not None:
            image = self.pre_processing(image)
        
        return image
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
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
    def __init__(self, n_samples, dataset, shuffle=False, batch_size=64, seed=42, transform=None):
        self.dataset = dataset(n_samples, seed=seed)
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


class SkyDataset01(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=64, pre_processing=None, seed=seed, shuffle=shuffle)


class SkyDataset02(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=28, pre_processing=None, seed=seed, shuffle=shuffle)
    
    def decide_labels(self):
        n_stars = (self.rng.integers(low=int(.85*self.n_samples), high=int(.95*self.n_samples)))
        n_galaxies = self.n_samples - n_stars
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)


class SkyDataset03(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=28, pre_processing=None, seed=seed, shuffle=shuffle)

    def decide_labels(self):
        n_stars = int((self.rng.integers(low=int(.15*self.n_samples), high=int(.25*self.n_samples))))
        n_galaxies = int(.5 * (self.n_samples - n_stars))
        wild_card = self.n_samples - (n_stars + n_galaxies)
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)] + [2 for _ in range(wild_card)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)
    
    def generate_image(self, label):
        if label == 2:
            return np.zeros((self.image_size, self.image_size))
        else:
            return super().generate_image(label)
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        # Convert label 2 (wildcard) to label 0 (Star)
        label = torch.where(label == 2, torch.tensor(0), label)
        return image, label


class SkyDataset04(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        super().__init__(n_samples, image_size=28, pre_processing=None, seed=seed, shuffle=shuffle)

    def decide_labels(self):
        n_stars = (self.rng.integers(low=int(.45*self.n_samples), high=int(.65*self.n_samples)))
        n_galaxies = (self.rng.integers(low=int(.6*(self.n_samples - n_stars)), high=int(.9*(self.n_samples - n_stars))))
        wild_card = self.n_samples - (n_stars + n_galaxies)
        labels = [0 for _ in range(n_stars)] + [1 for _ in range(n_galaxies)] + [2 for _ in range(wild_card)]
        
        if self.shuffle:
            self.rng.shuffle(labels)
        
        return np.asarray(labels)


class SkyDataset05(SkyDataset):
    def __init__(self, n_samples, seed=42, shuffle=False):
        self.noise_level = 0.25  # Increase noise level for this dataset
        super().__init__(n_samples, image_size=28, pre_processing=None, seed=seed, shuffle=shuffle)


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
