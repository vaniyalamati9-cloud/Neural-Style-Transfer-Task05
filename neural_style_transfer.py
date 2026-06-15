
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(filename, size=512):
    image = Image.open(filename).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(device)

def imshow(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def get_features(image, model):
    layers = {
        "0":  "conv1_1",
        "5":  "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "21": "conv4_2",
        "28": "conv5_1"
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_transfer(content_image, style_image, num_steps=100):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    content_features = get_features(content_image, vgg)
    style_features = get_features(style_image, vgg)
    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in style_features
    }

    style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.8,
        "conv3_1": 0.5,
        "conv4_1": 0.3,
        "conv5_1": 0.1
    }

    content_weight = 1e4
    style_weight = 1e2

    target = content_image.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    for step in range(num_steps):
        torch.cuda.empty_cache()
        target_features = get_features(target, vgg)

        content_loss = torch.mean(
            (target_features["conv4_2"] -
             content_features["conv4_2"]) ** 2
        )

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean(
                (target_gram - style_gram) ** 2
            )
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c * h * w)

        total_loss = (content_weight * content_loss +
                     style_weight * style_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"Step [{step+1}/{num_steps}] "
                  f"Total Loss: {total_loss.item():.2f}")

    return target
