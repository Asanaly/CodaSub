import time

from torchvision.models import resnet50
import pickle
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import segmentation_models_pytorch_local as smp

class modelInter:
    """
    self.model_cls -> model for classification (number 1)
    self.model_seg -> model for segmentation (number 2)
    """
    def __init__(self):

        self.model_seg = smp.Linknet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=3)
        """
        @misc
        {Iakubovskii: 2019,
         Author = {Pavel
        Iakubovskii},
        Title = {Segmentation
        Models
        Pytorch},
        Year = {2019},
        Publisher = {GitHub},
        Journal = {GitHub
        repository},
        Howpublished = {url
        {https: // github.com / qubvel / segmentation_models.pytorch}}
        }
        """
        self.model_cls = resnet50(pretrained=True)
        num_ftrs = self.model_cls.fc.in_features
        self.model_cls.fc = nn.Linear(num_ftrs, 2)  # 替换最后一层全连接层，以适应二分类问题

        self.mean = None
        self.std = None

    def load(self, path="./"):
        """
        Use it before predict(X), when two pickle models are in folder.
        :param path:
        :return:
        """
        with open(path + "/Model1.pickle", 'rb') as f:
            state_dict = pickle.load(f)
        self.model_cls.load_state_dict(state_dict, strict=False)

        with open(path + "/Model2.pickle", 'rb') as f:
            state_dict = pickle.load(f)
        self.model_seg.load_state_dict(state_dict, strict=False)

        return self

    def predict(self, X, isLabeled):
        #X = torch.from_numpy(X)
        """
        X: numpy array of shape (3,512,512)
        isLabeled (bool): we only label one image in a video. True represents this frame is labeled.
        """

        image = X.to(torch.float32).unsqueeze(0)  # (1,3,512,512)
        # Classify the image using the classification model
        cls_logit = self.model_cls(image)  # Assuming model_cls outputs logits
        cls_logit = torch.softmax(cls_logit, dim=-1)  # Convert logits to probabilities
        cls = cls_logit[0][0] >= 0.75  # Example threshold for binary classification
        seg_flag = cls.item()  # Convert to Python boolean

        seg = None  # Initialize segmentation variable

        if isLabeled:
            # If labeled, use the segmentation model to predict segmentation
            seg = self.model_seg(image)  # Get segmentation output
            seg = seg.squeeze(0).argmax(dim=0).detach().numpy()
            #seg = seg.argmax(dim=0).detach().numpy()

            # Post processing stuff

            # Apply Bilateral Filter for edge-preserving smoothing
            seg = cv2.bilateralFilter(seg.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

            # Convert to binary image
            seg_binary = (seg > 0.5).astype(np.uint8)

            # Aggressive morphological operations
            kernel = np.ones((7, 7), np.uint8)  # Use a larger kernel for stronger smoothing
            seg_binary = cv2.morphologyEx(seg_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            seg_binary = cv2.morphologyEx(seg_binary, cv2.MORPH_OPEN, kernel, iterations=2)

            # Find all contours
            contours, _ = cv2.findContours(seg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area, descending order
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Create a mask to identify the pixels corresponding to the two largest contours
            mask = np.zeros_like(seg_binary)

            # Draw the two largest contours as ellipses
            for contour in contours[:2]:  # Keep only the top 2 largest contours
                # Fit an ellipse to the contour
                if len(contour) >= 5:  # Ellipse fitting requires at least 5 points
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(mask, ellipse, 1, thickness=cv2.FILLED)  # Draw filled ellipse

            # Smooth the elliptical mask using Gaussian blur for softer edges
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

            # Preserve original pixel values in the elliptical areas
            smooth_seg = np.where(mask == 1, seg, 0)  # Keep original values in the mask area, set others to 0

            # Convert back to original format if needed
            seg = smooth_seg.astype(np.float32)

        return cls_logit.detach().numpy(), seg, seg_flag

    def save(self, path="./"):
        """
           Use after training
           Save the state dictionaries of the classification and segmentation models to pickle from pth files.
        """
        # Save the classification model's state dict separately as Model1.pickle
        with open(path + "/Model1.pickle", 'wb') as f:
            pickle.dump(self.model_cls.state_dict(), f)

        # Save the segmentation model's state dict separately as Model2.pickle
        with open(path + "/Model2.pickle", 'wb') as f:
            pickle.dump(self.model_seg.state_dict(), f)

    def get_cls_model(self):
        """Returns classification model, used for training"""
        return self.model_cls

    def get_seg_model(self):
        """Returns segmentation model, used for training"""
        return self.model_seg