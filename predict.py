import argparse
import json

import numpy as np
import torch
from PIL import Image
from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, default="./flowers/test/1/image_06743.jpg")
parser.add_argument('checkpoint', action="store_true", default=False)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, default="./cat_to_name.json")
parser.add_argument('--save_dir', type=str, default="./checkpoint.pth")
parser.add_argument('--arch', type=str, default="resnet")
parser.add_argument('--gpu', action="store_true", default=False)

args, _ = parser.parse_known_args()


def load_and_rebuild_model(checkpoint, save_dir, arch, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if arch == 'resnet':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)

    for param in model.parameters():
        param.requires_grad = False

    if checkpoint:
        state = torch.load(save_dir)

        if state["arch"] == 'resnet':
            model = models.densenet121(pretrained=True)
        elif state["arch"] == 'vgg19':
            model = models.vgg19(pretrained=True)
        else:
            raise ValueError('Unexpected network architecture', arch)

        model.classifier = state['classifier']
        model.load_state_dict(state['state_dict'])
        model.class_to_idx = state['class_to_idx']

    model.to(device)

    return model


def process_image(image):
    width, height = image.size
    size = 256
    new_size = 224

    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)

    resized_image = image.resize((width, height))

    x0 = (width - new_size) / 2
    y0 = (height - new_size) / 2
    x1 = x0 + new_size
    y1 = y0 + new_size

    cropped_image = image.crop((x0, y0, x1, y1))

    np_image = np.array(cropped_image) / 255.

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image_array = (np_image - mean) / std
    np_image_array = np_image_array.transpose((2, 0, 1))

    return torch.from_numpy(np_image_array)


def predict(image_path, checkpoint=False, topk=5, category_names="./cat_to_name.json", save_dir="./checkpoint.pth",
            arch="resnet",
            gpu=False):
    model = load_and_rebuild_model(checkpoint, save_dir, arch, gpu)
    model.eval()

    image = Image.open(image_path)

    img = process_image(image).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img.cuda())

    probs, classes = torch.exp(logps).data.topk(topk)
    if gpu:
        probs = probs.cpu().numpy()[0]
        classes = classes.cpu().numpy()[0]
    else:
        probs = probs.numpy()[0]
        classes = classes.numpy()[0]

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        names = list(cat_to_name.values())
        labels = [names[x] for x in classes]

    return probs, labels


if args.image and args.checkpoint:
    _image = args.image
    _checkpoint = args.checkpoint
    _top_k = args.top_k
    _category_names = args.category_names
    _save_dir = args.save_dir
    _arch = args.arch
    _gpu = args.gpu

    _probs, _labels = predict(_image, _checkpoint, _top_k, _category_names, _save_dir, _arch, _gpu)
    print(_probs)
    print(_labels)
