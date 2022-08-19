import time
import numpy as np
import torch
import cv2
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


'''Early Stopping'''
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def make_optimizer(args, network):
    trainable = filter(lambda x: x.requires_grad, network.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}

    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer):

    lambda1 = lambda epoch: (1 - epoch/args.epochs)**0.9
    scheduler = lrs.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)

    return scheduler


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def get_miou(preds, gt): #计算miou
    miou = 0
    pre_pic = torch.argmax(preds,1)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = gt[i]
        union = torch.logical_or(predict,mask).sum()
        inter = ((predict + mask)==2).sum()
        if union < 1e-5:
            return 0
        miou += inter / union
    return miou/batch


def get_boundary(pic,is_mask):
    if not is_mask:
        pic = torch.argmax(pic,1).cpu().numpy().astype('float64')
    else:
        pic = pic.cpu().numpy()
    batch, width, height = pic.shape
    new_pic = np.zeros([batch, width + 2, height + 2])
    mask_erode = np.zeros([batch, width, height])
    dil = int(round(0.02*np.sqrt(width ** 2 + height ** 2)))
    if dil < 1:
        dil = 1
    for i in range(batch):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(batch):
        pic_erode = cv2.erode(new_pic[j],kernel,iterations=dil)
        mask_erode[j] = pic_erode[1: width + 1, 1: height + 1]
    return torch.from_numpy(pic-mask_erode)


def get_biou(preds, gt):
    inter = 0
    union = 0
    pre_pic = get_boundary(preds, is_mask=False)
    real_pic = get_boundary(gt, is_mask=True)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        inter += ((predict * mask) > 0).sum()
        union += ((predict + mask) > 0).sum()
    if union < 1:
        return 0
    biou = (inter/union)
    return biou


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr
