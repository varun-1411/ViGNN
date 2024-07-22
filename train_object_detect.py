import torch
from torch import nn

import vig
# from your_vig_model import ViGModel

# Load the pretrained ViG model
# vig_model = ViGModel()
# vig_model.load_state_dict(torch.load('path/to/pretrained/vig_weights.pth'))

# Remove the prediction layer

gnn_model = vig.vig_ti_224_gelu()
# print(vig_model)
# vig_model.prediction = nn.Identity()
# print(model)
# input = torch.rand(5,3,224,224)
# print(vig_model(input))
#

import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    def __init__(self, backbone, in_feature, out_channels, scale_factors):
        super(SimpleFeaturePyramid, self).__init__()
        assert isinstance(backbone, nn.Module)
        self.backbone = backbone
        self.in_feature = in_feature
        self.scale_factors = scale_factors

        self.stages = nn.ModuleList()
        for scale in scale_factors:
            if scale == 1.0:
                continue
            self.stages.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.backbone(x)
        in_feature_map = features[self.in_feature]
        pyramid_features = [in_feature_map]

        for stage in self.stages:
            in_feature_map = self.downsample(in_feature_map)
            pyramid_features.append(stage(in_feature_map))

        for _ in range(len(pyramid_features), len(self.scale_factors)):
            in_feature_map = self.downsample(in_feature_map)
            pyramid_features.append(in_feature_map)

        return pyramid_features

import torch
import math
from torchvision.models import resnet50  # or any other backbone

# Define the backbone network
backbone_net = resnet50(pretrained=True)

# Define input feature name
in_feature = 'layer4'  # Assuming layer4 is the last feature map of the backbone network

# Number of output channels for the pyramid feature maps
out_channels = 256  # You can adjust this according to your needs

# Scale factors for upsampling or downsampling the input features
scale_factors = [4.0, 2.0, 1.0, 0.5]  # Adjust according to your requirements

# Initialize SimpleFeaturePyramid
feature_pyramid = SimpleFeaturePyramid(
    net=backbone_net,
    in_feature=in_feature,
    out_channels=out_channels,
    scale_factors=scale_factors
)

# Example input
input_tensor = torch.randn(5, 256, 50, 50)  # Batch size x Channels x Height x Width

# Forward pass
output_features = feature_pyramid(input_tensor)

# Access pyramid features
for key, value in output_features.items():
    print(f"Feature map '{key}' size: {value.shape}")

import torchvision.models.detection.mask_rcnn
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

class ModifiedMaskRCNN(nn.Module):
    def __init__(self, gnn_model):
        super(ModifiedMaskRCNN, self).__init__()
        # Load the pre-trained Mask R-CNN model
        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # Replace the backbone with your GNN model
        self.mask_rcnn = replace_backbone(self.mask_rcnn, gnn_model)
        # Modify other components of Mask R-CNN as needed

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")
        # Forward pass through Mask R-CNN
        if self.training:
            loss_dict = self.mask_rcnn(images, targets)
            return loss_dict
        else:
            return self.mask_rcnn(images)
#%%
def replace_backbone(model, gnn_model):
    model.backbone = gnn_model
    return model
#%%
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision

import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision

#

import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

def get_coco_dataloader(root, ann_file, batch_size=4, num_workers=2, train_size=0.8, seed=42):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load COCO dataset
    coco_dataset = torchvision.datasets.CocoDetection(root=root, annFile=ann_file, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(train_size * len(coco_dataset))
    valid_size = len(coco_dataset) - train_size
    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(coco_dataset, [train_size, valid_size])

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader


def train_mask_rcnn(data_loader, num_epochs=10):
    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one, specific to COCO
    num_classes = 91  # 91 classes in COCO dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = ModifiedMaskRCNN(in_features, num_classes)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set model to training mode
    model.train()

    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        running_loss = 0.0

        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Backward pass
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(data_loader)}")

        # Update learning rate
        lr_scheduler.step()

    print("Training complete")

# Example usage
root = '/path/to/coco_dataset'
ann_file = '/path/to/coco_annotation_file'
batch_size = 128
num_workers = 2
num_epochs = 50

data_loader,_ = get_coco_dataloader(root, ann_file, batch_size, num_workers)
train_mask_rcnn(data_loader, num_epochs)

