import json
import pandas as pd
from pathlib import Path
from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torchmetrics
from torchmetrics.functional import mean_absolute_error
import torchvision.models as models
from pytorch_lightning import LightningModule, LightningDataModule
from torch import nn, no_grad, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

root_path = Path(".")
frame_size = 224
original_image_size = 512

keypoints_train_transform = A.Compose(
    [
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.25, rotate_limit=15, p=0.75),
        # A.RandomBrightnessContrast(p=0.75),
        # A.SafeRotate(p=0.75),
        # A.Blur(blur_limit=2, p=0.25),
        # A.RandomBrightnessContrast(),
        A.Resize(frame_size, frame_size),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()],
    keypoint_params=A.KeypointParams(format='xy',
                                     remove_invisible=False)
)

keypoints_val_transform = A.Compose(
    [
        A.Resize(frame_size, frame_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()],
    keypoint_params=A.KeypointParams(format='xy',
                                     remove_invisible=False)
)

keypoints_predict_transform = A.Compose(
    [
        A.Resize(frame_size, frame_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()],
)


def read_rgb_image(filename: Path) -> np.ndarray:
    """
    Read RGB image from filesystem in RGB color order.
    Note: By default, OpenCV loads images in BGR memory order format.
    :param filename: Image file path
    :return: A numpy array with a loaded image in RGB format

    (c) https://github.com/BloodAxe/pytorch-toolbelt
    """
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Cannot read image '{filename}'")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image


def define_backbone(backbone_name):
    if backbone_name == "resnet50":
        backbone = models.resnet50(pretrained=True, progress=False)
    elif backbone_name == "resnet18":
        backbone = models.resnet18(pretrained=True, progress=False)
    elif backbone_name == "resnext50_32x4d":
        backbone = models.resnext50_32x4d(pretrained=True, progress=False)
    else:
        backbone = None
        print("Specify backbone name!")
        exit("Specify backbone name!")
    return backbone


class ISSDataModule(LightningDataModule):
    # DataModule will merge train and val datasets and split them into new train/validation/holdout datasets
    def __init__(self,
                 path_to_data: Path,
                 validation_share=0.2,
                 holdout_share=0.2,
                 shuffle_dataset=True,
                 random_seed=42,
                 batch_size=32):
        super().__init__()
        self.path_to_data = path_to_data
        self.validation_share = validation_share,
        self.holdout_share = holdout_share,
        self.shuffle_dataset = shuffle_dataset,
        self.random_seed = random_seed,
        self.batch_size = batch_size,
        self.image_paths = []
        self.labels = []
        self.train_image_paths = None
        self.validation_image_paths = None
        self.holdout_image_paths = None
        self.prediction_image_paths = None
        self.train_labels = None
        self.validation_labels = None
        self.holdout_labels = None

    def add_subset(self, subset_name):
        df = pd.read_csv(self.path_to_data / f"{subset_name}.csv")
        df["filenames"] = df['ImageID'].apply(lambda x: self.path_to_data / subset_name / f"{x}.jpg")

        df['parsed_location'] = df['location'].apply(lambda x: json.loads(x))
        temp = pd.DataFrame(df['parsed_location'].to_list(), columns=['X', 'Y'])
        df = pd.concat((df, temp), axis=1)

        labels = df[["distance", "X", "Y"]].values.tolist()

        return df["filenames"].values.tolist(), labels

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        oob_train_filenames, oob_train_labels = self.add_subset("train")
        self.image_paths += oob_train_filenames
        self.labels += oob_train_labels

        oob_valid_filenames, oob_valid_labels = self.add_subset("val")
        self.image_paths += oob_valid_filenames
        self.labels += oob_valid_labels

        path_to_test_images = root_path / "data" / "test"

        self.prediction_image_paths = [str(f) for f in path_to_test_images.glob("*.jpg")]

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset_size = len(self.labels)
        indices = list(range(dataset_size))
        validation_split = int(np.floor(self.validation_share[0] * dataset_size))
        holdout_split = int(np.floor((self.validation_share[0] + self.holdout_share[0]) * dataset_size))
        # [0 .. validation_indices ... *validation_split* ... holdout_indices ... *holdout_split* ... train_indices]
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        validation_indices = indices[:validation_split]
        holdout_indices = indices[validation_split:holdout_split]
        train_indices = indices[holdout_split:]
        image_paths = np.array(self.image_paths)
        labels = np.array(self.labels)
        self.train_image_paths = image_paths[train_indices]
        self.validation_image_paths = image_paths[validation_indices]
        self.holdout_image_paths = image_paths[holdout_indices]
        self.train_labels = labels[train_indices]
        self.validation_labels = labels[validation_indices]
        self.holdout_labels = labels[holdout_indices]

    def train_dataloader(self):
        train_split = ISSDataset(self.train_image_paths, self.train_labels, keypoints_train_transform)
        return DataLoader(train_split, batch_size=self.batch_size[0])

    def val_dataloader(self):
        val_split = ISSDataset(self.validation_image_paths, self.validation_labels, keypoints_val_transform)
        return DataLoader(val_split, batch_size=self.batch_size[0])

    def test_dataloader(self):
        test_split = ISSDataset(self.holdout_image_paths, self.holdout_labels, keypoints_val_transform)
        return DataLoader(test_split, batch_size=self.batch_size[0])

    def predict_dataloader(self):
        submission_split = ISSDataset(self.prediction_image_paths,
                                      labels=[],
                                      transform=keypoints_predict_transform,
                                      mode="predict")
        return DataLoader(submission_split, batch_size=self.batch_size[0])


class ISSDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 labels: List,
                 transform=None,
                 mode="train"  # train or predict
                 ):
        super().__init__()
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_rgb_image(self.image_paths[idx])

        if self.mode == "train":
            label = self.labels[idx]
            # label  = [distance, X, Y]
            # we should transform only X and Y
            transformed = self.transform(image=image,
                                         keypoints=[label[1:3]]
                                         )
            label = torch.LongTensor((label[0], transformed["keypoints"][0][0], transformed["keypoints"][0][1]))
            image = transformed["image"]
            return image, label
        elif self.mode == "predict":
            transformed = self.transform(image=image)
            image = transformed["image"]
            return image


class ISSDocker(LightningModule):
    def __init__(self,
                 backbone_name,
                 learning_rate=0.001,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr = learning_rate

        # init a pretrained NN
        backbone = define_backbone(backbone_name)
        num_filters = backbone.fc.in_features
        print(f"Number of filters is {num_filters}")
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.head = nn.Linear(num_filters, 3)  # distance and 2 coordinates

    def forward(self, x):
        self.feature_extractor.eval()
        with no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.head(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.long()
        features = self.feature_extractor(x).flatten(1)
        y_hat = self.head(features)
        loss = mean_absolute_error(y, y_hat)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = y.long()
        features = self.feature_extractor(x).flatten(1)
        y_hat = self.head(features)
        loss = mean_absolute_error(y, y_hat)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # y = y.long()
        features = self.feature_extractor(x).flatten(1)
        y_hat = self.head(features)
        loss = mean_absolute_error(y, y_hat)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, patience=10),
                'monitor': 'val_loss',
            }
        }


def prepare_submission(model):
    path_to_test_images = root_path / "data" / "test"

    model.eval()
    coefficient = original_image_size / frame_size
    with open("submission.csv", "w") as f:
        f.write("ImageID,distance,location\n")
        for file_number in range(0, 5000):
            image = read_rgb_image(path_to_test_images / f"{file_number}.jpg")
            image = keypoints_predict_transform(image=image)["image"]
            image = image.unsqueeze(0)
            # image = image.to(device)

            prediction = model(image)[0]
            # print(prediction)

            f.write(f'{file_number},{prediction[0]},"[coefficient * {prediction[1]}, {coefficient * prediction[2]}]"\n')
