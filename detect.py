import argparse
import os
import numpy as np
import time
import datetime
import torch
import torch.backends.cudnn as cudnn

from dataloaders import cfg
from models.densenet import *
from dataloaders.datasets.datasets import LoadStreams, LoadImages

def main():

    parser = argparse.ArgumentParser(description="PyTorch DenseNet Detecting")
    parser.add_argument('--in_path', type=str, default='inference/image/test', help='image to test')
    parser.add_argument('--out_path', type=str, default='inference/output', help='mask image to save')
    parser.add_argument('--backbone', type=str, default='net121', choices=['net121', 'net161', 'net169', 'net201'],
                                                                  help='backbone name (default: net121)')
    parser.add_argument('--compression', type=int, default=0.7, help='network output stride')
    parser.add_argument('--bottleneck', type=str, default=True, help='network output stride')
    parser.add_argument('--drop_rate', type=int, default=0.5, help='dropout rate')
    parser.add_argument('--training', type=str, default=True, help='')
    parser.add_argument('--weights', type=str, default='weights/model_best.pth.tar', help='saved model')
    parser.add_argument('--num_classes', type=int, default=2, help='')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='use which gpu to train, must be a \
                                                                 comma-separated list of integers only (default=0)')
    parser.add_argument('--img_size', type=int, default=(360, 360), help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--save_dir', action='store_true', default='inference/output', help='save results to *.txt')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())

    webcam = args.in_path.isnumeric() or args.in_path.endswith('.txt') or args.in_path.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    results = 'result' + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + '.txt'
    results_file = os.path.join(args.save_dir, results)

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    model_s_time = time.time()

    # Define network
    model = DenseNet(
        backbone    = args.backbone,
        compression = args.compression,
        num_classes = args.num_classes,
        bottleneck  = args.bottleneck,
        drop_rate   = args.drop_rate,
        sync_bn     = args.sync_bn,
        training    = args.training
    )

    labels2classes = cfg.labels_to_classes
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model           = model.cuda()
    model_u_time    = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}s".format(model_load_time))

    # for i, (image, target) in enumerate(test_loader) :
    #     s_time = time.time()
    #     print(image.path)
    #     print(target)
    #
    #     model.eval()
    #     if args.cuda:
    #         image = image.cuda()
    #
    #     with torch.no_grad():
    #         out        = model(image)
    #         prediction = torch.max(out, 1)[1]
    #
    #     u_time = time.time()
    #     img_time = u_time-s_time
    #     label = labels2classes[str(prediction.cpu().numpy()[0])]
    #
    #     #print("image:{} label:{} time: {} ".format(image, label, img_time))
    #     print("label:{} time: {} ".format(label, img_time))


    # composed_transforms = transforms.Compose([
    #     transforms.Resize(args.img_size),
    #     transforms.ToTensor()
    # ])
    #
    # for name in os.listdir(args.in_path):
    #     s_time = time.time()
    #     image  = Image.open(args.in_path+"/"+name).convert('RGB')
    #     target = Image.open(args.in_path + "/" + name).convert('L')
    #     sample = {'image': image, 'label': target}
    #
    #     tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
    #
    #     model.eval()
    #
    #     if args.cuda:
    #         tensor_in = tensor_in.cuda()
    #     with torch.no_grad():
    #         out = model(tensor_in)
    #         prediction = torch.max(out, 1)[1]
    #
    #     u_time = time.time()
    #     img_time = u_time-s_time
    #     label = labels2classes[str(prediction.cpu().numpy()[0])]
    #
    #     print("image:{} label:{} time: {} ".format(name, label, img_time))

    if args.cuda :
        model.half()

    if webcam :
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset         = LoadStreams(args.in_path, img_size=args.img_size)
    else:
        dataset = LoadImages(args.in_path, img_size=args.img_size)

    all_s_time = time.time()

    for path, image, im0s, vid_cap in dataset :
        image  = torch.from_numpy(image).cuda()
        image  = image.half() if args.cuda else image.float()  # uint8 to fp16/32
        image /= 255.0              # 0 - 255 to 0.0 - 1.0

        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        s_time = time.time()

        model.eval()

        with torch.no_grad():
            out        = model(image)
            prediction = torch.max(out, 1)[1]

        u_time   = time.time()
        img_time = u_time-s_time
        label    = labels2classes[str(prediction.cpu().numpy()[0])]

        print("label: {} time: {}s".format(label, img_time))

        with open(results_file, 'a') as f:
            f.write("%s %s \n" % (path, label))

    all_e_time = time.time()
    all_time   = all_e_time - all_s_time

    print("Total time is {}s".format(all_time))
    print("Detect is done.")


if __name__ == "__main__":
   main()