import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


# dataset_root = '/shared_data/LP'

dataset_root = 'specify your dataset root path'

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(dataset_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(dataset_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(dataset_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(dataset_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = f"{dataset_root}/imagenet-r/train/"
        test_dir = f"{dataset_root}/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = f"{dataset_root}/imagenet-a/train/"
        test_dir = f"{dataset_root}/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = f"{dataset_root}/cub/train/"
        test_dir = f"{dataset_root}/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = f"{dataset_root}/objectnet/train/"
        test_dir = f"{dataset_root}/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        train_dir = f"{dataset_root}/omnibenchmark/train/"
        test_dir = f"{dataset_root}/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dir = f"{dataset_root}/vtab/train/"
        test_dir = f"{dataset_root}/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class cars(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(196).tolist()

    # def __init__(self):
    #     pass

    def download_data(self):

        # train_dataset = datasets.StanfordCars(dataset_root, split='train', download=True)
        # test_dataset = datasets.StanfordCars(dataset_root, split='test', download=True)
        # self.train_data, self.train_targets = train_dataset.data, np.array(
        #     train_dataset.targets
        # )
        # self.test_data, self.test_targets = test_dataset.data, np.array(
        #     test_dataset.targets
        # )
        #
        train_dir = f"{dataset_root}/cars/train/"
        test_dir = f"{dataset_root}/cars/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class core50(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def __init__(self, inc):
        self.inc = inc
    def download_data(self):
        # download from here: http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz
        train_dir = f"{dataset_root}/core50_128x128/" + self.inc + "/"
        # print(train_dir)
        test_dir = f"{dataset_root}/core50_128x128/test_3_7_10/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class core50_joint(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def __init__(self):
        self.dil_tasks = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']

    def get_training_data(self):
        images = []
        labels = []

        for task in self.dil_tasks:
            train_dir = f"{dataset_root}/core50_128x128/" + task + "/"
            train_dset = datasets.ImageFolder(train_dir)

            for item in train_dset.imgs:
                images.append(item[0])
                labels.append(item[1])

        return np.array(images), np.array(labels)
    def download_data(self):
        # download from here: http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz

        self.train_data, self.train_targets = self.get_training_data()

        train_dir = f"{dataset_root}/core50_128x128/" + self.inc + "/"
        # print(train_dir)
        test_dir = f"{dataset_root}/core50_128x128/test_3_7_10/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class cddb(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(2).tolist()

    def __init__(self, inc):
        self.inc = inc

    def download_data(self):
        # download from here: https://coral79.github.io/CDDB_web/
        train_dir = f"{dataset_root}/CDDB/" + self.inc + "/train/"
        # print(train_dir)
        test_dir = f"{dataset_root}/CDDB/CDDB-hard_val/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class cddb_joint(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(2).tolist()

    def __init__(self):
        self.incs = ['gaugan', 'biggan', 'wild', 'whichfaceisreal', 'san']

    def get_training_data(self):
        images = []
        labels = []

        for inc in self.incs:
            train_dir = f"{dataset_root}/CDDB/" + inc + "/train/"
            train_dset = datasets.ImageFolder(train_dir)

            for item in train_dset.imgs:
                images.append(item[0])
                labels.append(item[1])

        return np.array(images), np.array(labels)

    def download_data(self):
        # download from here: https://coral79.github.io/CDDB_web/
        self.train_data, self.train_targets = self.get_training_data()

        test_dir = f"{dataset_root}/CDDB/CDDB-hard_val/val/"

        test_dset = datasets.ImageFolder(test_dir)

        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class domainnet(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(345).tolist()

    def __init__(self, inc):
        self.inc = inc

    def download_data(self):
        # download from http://ai.bu.edu/M3SDA/#dataset (use "cleaned version")
        aa = np.loadtxt(dataset_root + '/DomainNet/' + self.inc + '_train.txt', dtype='str')
        self.train_data = np.array([dataset_root + '/DomainNet/' + x for x in aa[:, 0]])
        self.train_targets = np.array([int(x) for x in aa[:, 1]])

        dil_tasks = ['real', 'quickdraw', 'painting', 'sketch', 'infograph', 'clipart']
        files = []
        labels = []
        for task in dil_tasks:
            aa = np.loadtxt(dataset_root + '/DomainNet/' + task + '_test.txt', dtype='str')
            files += list(aa[:, 0])
            labels += list(aa[:, 1])
        self.test_data = np.array([dataset_root + '/DomainNet/' + x for x in files])
        self.test_targets = np.array([int(x) for x in labels])

class domainnet_joint(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(345).tolist()

    def download_data(self):
        # download from http://ai.bu.edu/M3SDA/#dataset (use "cleaned version")
        dil_tasks = ['real', 'quickdraw', 'painting', 'sketch', 'infograph', 'clipart']
        files = []
        labels = []
        for task in dil_tasks:
            aa = np.loadtxt(dataset_root + '/DomainNet/' + task + '_train.txt', dtype='str')
            files += list(aa[:, 0])
            labels += list(aa[:, 1])
        self.train_data = np.array([dataset_root + '/DomainNet/' + x for x in files])
        self.train_targets = np.array([int(x) for x in labels])

        files = []
        labels = []
        for task in dil_tasks:
            aa = np.loadtxt(dataset_root + '/DomainNet/' + task + '_test.txt', dtype='str')
            files += list(aa[:, 0])
            labels += list(aa[:, 1])
        self.test_data = np.array([dataset_root + '/DomainNet/' + x for x in files])
        self.test_targets = np.array([int(x) for x in labels])