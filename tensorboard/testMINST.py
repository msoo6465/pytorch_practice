import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import argparse
from MNIST_model import MNIST_model
import os
import shutil


parser = argparse.ArgumentParser(description='Pytorch MNIST Training')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

best_acc = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main():
    global best_acc
    global device
    args = parser.parse_args()

    torch.manual_seed(777)

    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    mnist_train = dsets.MNIST(root='mnist_train',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
    mnist_test = dsets.MNIST(root='mnist_val',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)
    # print([type(d) for i,d in mnist_test])
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=False)
    # print(len(test_loader))
    #
    # print([type(d) for i, d in test_loader])
    model = MNIST_model().to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    total_batch = len(data_loader)
    print(f'총 배치의 수 : {total_batch}')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_acc.to(device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch{checkpoint['epoch']})")
        else:
            print(os.path.join(os.getcwd(),args.resume))
            print(f"No Checkpoint {args.resume}")


    if args.evaluate:
        validation(test_loader, model, criterion, args)
        return

    for epoch in range(args.epochs):
        avg_cost = 0

        for i,(X,Y) in enumerate(data_loader):

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            cost = criterion(pred,Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost/total_batch
            if i%1000 == 0:
                print(f'[{i}/{total_batch}] Training')
                break
        adjust_learning_rate(optimizer,epoch,args)
        acc1 = validation(test_loader,model,criterion,args)

        print('[Epoch : {:>4}] cost = {:>.9}'.format(epoch+1,avg_cost))

        is_best = acc1 > best_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'MNIST',
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

def validation(test_loader,model,criterion,args):
    global device
    accuracy = 0
    model.eval()
    for i, (inp,tar) in enumerate(test_loader):
        X_test = inp.to(device)
        Y_test = tar.to(device)

        prediction = model(X_test)
        # print(torch.argmax(prediction,1))
        # print(Y_test)
        correct_prediction = torch.argmax(prediction,1)==Y_test
        # print(correct_prediction.float().mean())
        accuracy += correct_prediction.float().mean()
    print('Accuracy : ',accuracy.item()/(i+1))
    return accuracy.item()/(i+1)



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state,is_best,filename = 'mnsitpoint.pth.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'mnist_best.pth.tar')

if __name__ == '__main__':
    main()