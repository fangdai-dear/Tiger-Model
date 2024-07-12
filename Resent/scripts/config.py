from torchvision import transforms

THYROID_LABEL = ['Benign', 'Malignant']
THYROID_SUBGROUP = [['Papillary','Follicular','Medullary']]


def THYROID():
    return (THYROID_LABEL, THYROID_SUBGROUP[0])


def Transforms(name):
    # if name in ["Thyroid"]:
    data_transforms_ONE = {
        'valid': transforms.Compose([
                transforms.CenterCrop(512),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ]),
        'train': transforms.Compose([
            transforms.CenterCrop(512),
            transforms.RandomCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    }
    return data_transforms_ONE
        