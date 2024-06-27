from Harmony.data import CC3M, ImageNet, YFCC15M
from torchvision import datasets, transforms

dataset_classes = {
        "YFCC15M": YFCC15M,
        "CC3M": CC3M,
        "KINETICS700_FRAMES": "",
        "CLEVR_COUNTS": "",
        "CALTECH101": "",
        "MNIST": datasets.MNIST,
        "STL10": datasets.STL10,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "FOOD101": datasets.Food101,
        "EUROSAT": datasets.EuroSAT,
        "DTD": datasets.DTD,
        "GTSRB": datasets.GTSRB,
        "FER2013": datasets.FER2013,
        "COUNTRY211": "",
        "AIRCRAFT": datasets.FGVCAircraft,
        "PETS": "",
        "KITTI_DISTANCE": datasets.Kitti,
        "FLOWERS": datasets.Flowers102,
        "RENDERED_SST2": datasets.RenderedSST2,
        "CARS": datasets.StanfordCars,
        "CUB200": "",
        "SUN397": "",
        "RESISC45": "",
        "IMAGENET": ImageNet,
        "IMAGENET-A": "",
        "IMAGENET-R": "",
        "IMAGENET-O": ""
}


def get_dataset_from_string(string):
    keys = string.split(":")
    data = keys[0].upper()
    try:
        if data in dataset_classes.keys():
            return dataset_classes[data]
        else:
            return datasets.ImageFolder
    except:
        raise Exception(f"Dataset {data} is not available")
