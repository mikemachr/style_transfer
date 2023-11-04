from PIL import Image
from io import BytesIO
import numpy as np
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

import io

vgg = models.vgg19(weights='DEFAULT').features


for param in vgg.parameters():
    param.requires_grad_(False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)


def load_image(img, max_size=400, shape=None):

    
    image_stream = io.BytesIO(img)
    image = Image.open(image_stream)
    if max(image.size) > max_size:

        size = max_size

    else:

        size = max(image.size)
    

    if shape is not None:

        size = shape
        

    in_transform = transforms.Compose([

                        transforms.Resize(size),

                        transforms.ToTensor(),

                        transforms.Normalize((0.485, 0.456, 0.406), 

                                             (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image


def im_convert(tensor):    

    image = tensor.to("cpu").clone().detach()

    image = image.numpy().squeeze()

    image = image.transpose(1,2,0)

    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    image = image.clip(0, 1)


    return image


def gram_matrix(tensor):

    _, d, h, w = tensor.size()

    tensor = tensor.view(d, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram 


def get_features(image, model, layers=None):

    

    if layers is None:

        layers = {'0': 'conv1_1',

                  '5': 'conv2_1', 

                  '10': 'conv3_1', 

                  '19': 'conv4_1',

                  '30': 'conv5_2', #content

                  '28': 'conv5_1'}

        

    features = {}

    x = image

    for name, layer in model._modules.items():

        x = layer(x)

        if name in layers:

            features[layers[name]] = x
        
    return features






def main(content_image , style_image, steps: int):

    style_image = load_image(style_image).to(device)
    content_image  = load_image(content_image).to(device)


    content_features = get_features(content_image, vgg)

    style_features = get_features(style_image, vgg)


    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}


    target = content_image.clone().requires_grad_(True).to(device)



    style_weights = {'conv1_1': 1.5,

                    'conv2_1': 0.80,

                    'conv3_1': 0.25,

                    'conv4_1': 0.25,

                    'conv5_1': 0.25}


    content_weight = 1e-2  

    style_weight = 1e9  




    optimizer = optim.Adam([target], lr=0.01)

    steps = steps  


    for i in range(steps):

        

        target_features = get_features(target, vgg)

        
        
        content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1'])**2)

        

        style_loss = 0

        for layer in style_weights:

            target_feature = target_features[layer]

            target_gram = gram_matrix(target_feature)

            _, d, h, w = target_feature.shape

            style_gram = style_grams[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

            style_loss += layer_style_loss / (d * h * w)

            

        total_loss = content_weight * content_loss + style_weight * style_loss

        

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        

    return(im_convert(target))

