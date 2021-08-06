
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import dataloaders.cfg



def make_data_loader(args, **kwargs):

    if args.dataset == 'oled_data':

        # 构建数据提取器，利用dataloader
        # 利用torchvision中的transforms进行图像预处理

        mean = [0.49139968, 0.48215841, 0.44653091]
        stdv = [0.24703223, 0.24348513, 0.26158784]

        train_transforms = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])

        ##ImageFolder对象可以将一个文件夹下的文件构造成一类
        # 所以数据集的存储格式为一个类的图片放置到一个文件夹下
        # 然后利用dataloader构建提取器，每次返回一个batch的数据，在很多情况下，利用num_worker参数
        # 设置多线程，来相对提升数据提取的速度

        """
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
        """
        num_class = cfg.NUM_CLASSES

        train_set = datasets.ImageFolder(cfg.TRAIN_DATASET_DIR, transform=train_transforms)
        val_set   = datasets.ImageFolder(cfg.VAL_DATASET_DIR, transform=val_transforms)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader   = DataLoader(val_set, batch_size=args.batch_size*2, shuffle=False, **kwargs)

        return train_loader, val_loader, num_class

    else :
        raise NotImplementedError

