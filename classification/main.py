import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
## torch.distributed는 분산처리를 위한 라이브러리
import torch.optim
import torch.multiprocessing as mp
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
''' 멀티프로세스를 위한 라이브러리
멀티프로세스는 하나의 프로그램을 여러개의 프로세스로 구성하여 각 프로세스가 하나의 작업을 처리하도록 하는 것
장점 : 여러개의 자식 프로세스 중 하나에 문제가 발생하면 그 자식 프로세스만 죽는 것이상으로 다른 영향이 확산되지 않는다.
단점 : Context Switching 과정에서 캐쉬 메모리 초기화 등 무거운 작업이 진행되며 시간이 많이 걸리는 등 오버헤드가 발생
        프로세스간 공유하는 메모리가 없기 때문에 Context Swithcing이 발생하면 캐쉬에 있는 모든 데이터를 모두 리셋하고 다시 정보를 불러와야 한다.
        프로세스는 각각 독립된 메모리 영역을 할당받기 때문에 프로세스 사이에서 변수를 공유 할 수 없다.
        Context Switching이란?
            CPU에서 여러 프로세스를 돌아가면서 작업을 처리하는데 이 과정을 Context Switching이라 한다.
            구체적으로, 동작 중인 프로세스가 대기를 하면서 해당 프로세스의 상태를 보관하고 대기하고 있던 다음 순서의 프로세스가 동작하면서 
            이전에 보관했던 프로세스의 상태를 복구하는 작업을 말한다.
'''
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ImageFolder import ImageFolder
from model import resnet
from model import vgg
import cv2
from sklearn.manifold import TSNE
import seaborn as sns

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
# model_name은 torchvision에 있는 모델들의 이름을 하나의 리스트로 저장 모델이름이 __로 시작하는것은 제외


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18'),
                    # choices=model_names,
                    # help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=777, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-g','--gradcam',dest='gradcam',action='store_true',
                    help='See grad cam')
parser.add_argument('-t','--tsne',dest='tsne',action='store_true',
                    help='See tSNE')
parser.add_argument('--numclasses',default=None,type=int,
                    help='Classes')
parser.add_argument('--dim', default=2, type=int,
                    help='t-SNE dimension')

## argument 파싱하는 부분
## 학습할 것인지, 밸리데이션 할것인지, 모델이름, gpu사용  할것인지 등에 대한 정보를 입력해주는 부분

best_acc1 = 0
## 최고의 정확도를 측정하기 위해 정확도 초기화


def main():
    args = parser.parse_args()
    ## 위에서 파싱한 정보들을 가져오는 부분

    if args.seed is not None:
        ## 시드에 대한 정보가 있을 시 시드 고정
        ## 시드 고정시키는 이유는 항상 동일한 결과를 가지게 하기 위해서
        random.seed(args.seed)
        ## random은 torchvision에서 사용되는 난수들을 고정하기 위해서 random seed를 고정
        torch.manual_seed(args.seed)
        ## torch.manual_seed를 고정하는 이유는 torch라이브러리에서 사용되는 난수들을 고정시키기 위함
        cudnn.deterministic = True
        np.random.seed(args.seed)
        ## cudnn 역시 난수를 고정하기 위해서 True값을 사용.
        ## 부작용으로는 연산 속도가 줄어드는 현상이 발생 할 수 있다.
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        ## 추가적으로 numpy까지 시드를 고정시켜 줄 수 있다.


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        ## gpu 정보가 있을 경우 gpu를 활성화 시킨다.

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        ## 분산 처리를 위한 매개변수 초기화

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ## 참인것을 반환

    ngpus_per_node = torch.cuda.device_count()
    ## gpu의 개수를 반환한다. 반환형 : int
    if args.multiprocessing_distributed:
        ## multiprocessing_distributed에 대한 정보가 있을 시에 파라미터 초기화
        ## world_size는 작업에 사용될 프로세스 수
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        '''
        spawn은 멀티프로세스의 실행 및 추적하기 위한 함수
        fn은 메인 프로세스에서 실행될 함수 fn(i,*args)
            i는 프로세스 인덱스이며, args는 인자들의 튜플값
        args는 fn에 넘어갈 인자들
        nprocs는 spawn하기위한 프로세스 수
        join은 모든 프로세스들의 차단 join???
        deamon은 deamonic 프로세스가 생성하도록 하는 플래그??
        start_method는 spaw함수로 실행된다. 다른 시작 메소드를 사용하려면 start_process()를 사용하면된다.(사용되지 않는다.)
        '''

    else:
        # Simply call main_worker function
        ## 분산연산에 대한 정보를 사용하지 않을 경우 간단하게 main_worker만 사용
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    ## 앞서 선언한 best_acc1의 변수 값을 사용
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        '''
        분산 처리 프로세스 그룹을 초기화
        backend는 프로세스간의 통신 프로토콜
        init_method는 초기화 방법을 지정하는 URL, 기본값은 'env://'
        world_size는 작업에 참여하는 프로세스 수
        rank는 현재 프로세스의 순위
        store는 모든 worker에 접근할 수 있는 key/value.connection/address정보로 사용가능. init_method와 상호 배타적
        timeout은 프로세스 그룹의 작업시간초과. 기본값은 30분
        '''
    # create model
    if args.pretrained:
        ## 사전훈련된 모델을 사용하기 위한 코드
        print("=> using pre-trained model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](pretrained=True)
        if args.arch == 'vgg':
            model = vgg.vgg11_bn(pretrained=True, num_classes=args.numclasses)
        elif args.arch == 'resnet':
            model = resnet.resnet18(num_classes=args.numclasses)
    else:
        ## 일반 모델을 사용하기 위한 코드
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'vgg':
            model = vgg.vgg11_bn(num_classes=args.numclasses)
        elif args.arch == 'resnet':
            model = resnet.resnet18(num_classes=args.numclasses)
    # print(model)
    # print(model.state_dict().keys())
    # total_params = 0
    # for t in model.state_dict().keys():
    #     if 'weight' in t:
    #         # print(t,":",model.state_dict()[t].shape)
    #         if 'fc' in t or 'classifier' in t:
    #             a = model.state_dict()[t].shape[0]
    #             b = model.state_dict()[t].shape[1]
    #             params = a * b
    #             print(t, model.state_dict()[t].shape, params)
    #             total_params += a * b
    #         elif 'bn' in t or 'downsample.1' in t:
    #             total_params += model.state_dict()[t].shape[0]
    #             print(t, model.state_dict()[t].shape, params)
    #         else:
    #             a = model.state_dict()[t].shape[0]
    #             b = model.state_dict()[t].shape[1]
    #             c = model.state_dict()[t].shape[2]
    #             d = model.state_dict()[t].shape[3]
    #             params = a * b * c * d
    #             print(t, model.state_dict()[t].shape, params)
    #             total_params += a * b * c * d
    # print("total_params : ",total_params)
    # exit()
        # model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        ## gpu를 사용할 수 없으면 cpu를 사용하는 코드
        print('using CPU, this will be slow')

    elif args.distributed:
        '''
        멀티 프로세스 분산을 사용하기 위해서는 병렬분산 생성자를 항상 단일 장치 범위를 설정해야 한다.
        그렇지 않으면 모든 사용가능한 장치를 사용한다.
        '''
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            '''
            프로세스당 하나의 GPU와 분산병렬처리를 할 경우 배치 사이즈를 직접 분할 해야한다. 
            사용하려고 하는 GPU의 수로 나눠주어야 한다.
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        ## GPU설정이 없다면 모든 사용가능한 GPU에 나눠서 배치한다.
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    ## Loss와 optimizer를 정의한다.
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    '''
    교차 엔트로피 함수
    '''

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    '''
    최적화 함수로 SGD를 사용
    옵티마이저는 계산된 기울기를 기반으로 매개변수를 업데이트 한다.
    파라미터들
        params : 적용시킬 파라미터들 그룹
        lr : learning rate로 가중치 적용을 얼만큼 적용시킬지에 대한 비율 
        momentum : 경사 하강법에 관성을 더해주는 강도
        weight_decay : 오버피팅을 최소화하는 방법 특정 가중치값이 커지는 것을 방지하기 위해 손실함수에 특정값을 더해주는 것.
        dampening : bias에 특정값을 빼주는것
        nesterov : 이전 속도를 더해주는것 Nesterov Momentum
    '''

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
                ## map_location은 어떤 gpu에 로드 할 것인지
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            state_dict = checkpoint['state_dict']

            for k,v in state_dict.items():
                if 'module' in k:
                    r = k[7:]
                    new_state_dict[r] = v
            if new_state_dict:
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    ## 하드웨어 자동 튜너를 사용해 최적 알고리즘을 찾아준다.
    ## 입력크기가 변하지 않으면 런타임이 훨씬 빠르다.
    ## 입력크기가 자주 변경될 경우 런타임이 손상 될 수 있다.

    # Data loading code
    ## 데이터 가져오기 위한 패스지정
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ## 데이터셋 로드 하는 부분
    ## RandomResizedCrop은 랜덤하게 224x224 만큼 잘라내는 함수
    ## 파라미터로는 size, scale, ratio, interpolation이 있다.
    ## RandomHorizontalFlip은 랜덤하게 뒤집는 함수 파라미터로는 p가 있는데 뒤집힐 확률

    trans = transforms.Compose([
        transforms.ColorJitter(brightness=(0.75, 1), hue=(-0.1, 0.1)),
        transforms.RandomAffine(degrees=(0, 15), translate=(0.05, 0.05), shear=(10, 10)),
        transforms.RandomRotation((0, 20)),
        transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(
        traindir,
        transform= trans,
    )


    if args.distributed:
        '''
        데이터로드를 데이터 세트의 하위 집합으로 제한하는 샘플러
            dataset : 활용될 데이터 셋
            num_replicas : 분산 교육에 참여하는 프로세스 수
            rank : 프로세스 순위 
            shuffle : 인덱스를 섞을 것인지
            seed : 디폴트 0
            drop_last : 디폴트 False 
        '''
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    '''
    데이터 세트와 샘플러를 결합하고 주어진 데이터 세트를 로드
    파라미터
        dataset : 데이터를 로드할 데이터 세트
        batch_size : 로드할 배치 샘플 수
        shuffle : 매 epoch당 데이터를 다시 섞는 기능
        sampler : 데이터 세트ㅡ에서 샘플을 추출하는 전략 정의. 일부 경우 shuffle을 하면 안된다.
        batch_sampler : 한번에 배치의 indices를 반환????
        num_workers : 데이터로드에 사용할 하위 프로세스 수
        collate_fn : Tensor의 미니 배치 형성 map데이터세트 일괄 로드 시 사용
        pin_memory : True 일경우 Tensor를 반환하기 전에 CUDA 고정 메모리로 복사
        drop_last : 데이터세트 크기가 배치크기로 나눌 수 없는 경우 마지막 불완전 배치를 삭제
        timeout : 처리 하기 위한 제한 시간 값
        worker_init_fn : worker ID를 입력으로 사용
        perfetch_fator : 각 작업자가 미리로드 한 샘플 수
        persist_workers : 사용 된 후 프로세스를 종료하지 않음. 데이터 인스턴스를 계속 유지할 수 있다.
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.gradcam:
        # model = MNIST_model.MNIST_model()
        model.cpu()
        model.eval()

        val_loader = torch.utils.data.DataLoader(
            ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        for index, (img,_) in enumerate(val_loader):
            # img, _ = next(iter(val_loader))
            pred = model(img.cpu())
            if torch.argmax(pred) == _:
                print(torch.argmax(pred))
                print(_)
                print("=====================")
            continue

            pred[:, _.item()].backward()

            gradients = model.get_activations_gradient()

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            activations = model.get_activations(img).detach()


            for i in range(512):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()

            heatmap = np.maximum(heatmap, 0)

            # heatmap /= torch.max(heatmap)

            # plt.matshow(heatmap.squeeze())
            # plt.show()

            img = np.array(img.squeeze().cpu())

            backtorgb = img.transpose((1,2,0))

            backtorgb = cv2.resize(src=np.array(backtorgb), dsize=(224, 224))
            heatmap = cv2.resize(src=np.array(heatmap), dsize=(224, 224))
            heatmap = heatmap - np.min(heatmap)
            heatmap = heatmap / np.max(heatmap)
            # heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            backtorgb = backtorgb + 2
            backtorgb = (backtorgb)/np.max(backtorgb)
            backtorgb = np.uint8(125*backtorgb)


            superimposed_img = heatmap   + backtorgb
            # cv2.imshow('a', backtorgb)
            # cv2.waitKey(0)
            if not os.path.isdir('grad_img'):
                os.makedirs('grad_img')
            cv2.imwrite(f'grad_img/map{index}.jpg', superimposed_img)
        return

    if args.tsne:
        model.cpu()
        val_loader = torch.utils.data.DataLoader(
            ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        if args.dim == 2:
            modelt = TSNE(n_components=2, learning_rate=100)
        else:
            modelt = TSNE(n_components=3, learning_rate=100)

        fig = plt.figure(figsize=(10, 10))
        if args.dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        for index,(img,tar) in enumerate(val_loader):
            try:
                tar = tar.view(args.batch_size,1)
            except:
                tar = tar.view(len(tar),1)
            tar = tar.cpu().detach().numpy()

            if index == 0:
                feature = model(img)
                feature=torch.nn.functional.log_softmax(feature)
                feature = feature.cpu().detach().numpy()
                feature = np.concatenate([feature,tar],axis=1)
                print(feature.shape)
            else:
                feature_t = model(img)
                feature_t = torch.nn.functional.log_softmax(feature_t)
                feature_t = feature_t.cpu().detach().numpy()
                feature_t = np.concatenate([feature_t,tar],axis=1)
                feature=np.concatenate([feature,feature_t],axis=0)


        tar = feature[:,-1]
        feature = feature[:,:-1]
        tar = tar.reshape((819,1))

        transformed = modelt.fit_transform(feature)
        transformed = np.concatenate([transformed,tar],axis=1)

        tar_label = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        xs = transformed[:, 0]
        ys = transformed[:, 1]
        if args.dim == 3:
            zs = transformed[:, 2]
            ax.scatter(xs[:148], ys[:148],zs[:148], c='tab:blue', label=tar_label[0])
            ax.scatter(xs[148:349], ys[148:349],zs[148:349], c='tab:orange', label=tar_label[1])
            ax.scatter(xs[349:496], ys[349:496],zs[349:496], c='tab:green', label=tar_label[2])
            ax.scatter(xs[496:633], ys[496:633],zs[496:633], c='tab:red', label=tar_label[3])
            ax.scatter(xs[633:], ys[633:],zs[633:], c='#353038', label=tar_label[4])
        else:
            ax.scatter(xs[:148], ys[:148], c='tab:blue', label=tar_label[0])
            ax.scatter(xs[148:349], ys[148:349], c='tab:orange', label=tar_label[1])
            ax.scatter(xs[349:496], ys[349:496], c='tab:green', label=tar_label[2])
            ax.scatter(xs[496:633], ys[496:633], c='tab:red', label=tar_label[3])
            ax.scatter(xs[633:], ys[633:], c='#353038', label=tar_label[4])

        ax.legend()
        ax.grid(True)

        plt.show()
        return




    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        ## 30 epoch마다 10배 learning rate를 줄인다.
        adjust_learning_rate(optimizer, epoch, args)

        # 한 epoch 학습
        train(train_loader, model, criterion, optimizer, epoch, args)

        # validation set 정확도 측정
        acc1 = validate(val_loader, model, criterion, args)

        # best_acc를 checkpoint로 저장하는 부분
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            ## 체크포인트 저장
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,f'flower_{args.arch}_checkpoint.pth',f'flower_{args.arch}_best.pth')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        ## 데이터 로딩 시간을 측정한다.
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        ## 모델의 결과를 받아온다.
        output = model(images)
        # print(output)
        ## 위에 정의한 크로스 엔트로피 로스에 모델 결과와 GT 비교
        loss = criterion(output, target)
        # print(loss)

        ## 정확도를 측정하고 로스를 기록한다.
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        ## 역전파 시행 후 적용
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## 시간 측정
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    ## 상태 출력
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    ## 모델 측정 상태로 변경
    model.eval()
    ## no_grad로 기울기를 업데이트 하지 않음
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                ## val 이미지를 해당하는 gpu에 복사
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',best='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()