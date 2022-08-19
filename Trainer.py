# -*- coding: UTF-8 -*-
import os
import torch
from tqdm import tqdm
from utils import make_optimizer, make_scheduler, get_miou, get_biou, Adder, Timer, check_lr


class Trainer(object):
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.optimizer = make_optimizer(args, self.model)
        self.scheduler = make_scheduler(args, self.optimizer)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):

        print('Start Training...')

        epoch_loss = Adder()
        mIoU = Adder()
        bIoU = Adder()
        timer = Timer('m')
        best_mIoU = 0.

        self.epoch = 1

        if self.args.resume:
            state = torch.load(self.args.resume)
            self.epoch = state['epoch']
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.model.load_state_dict(state['model'])
            print('Resume from %d epoch' % self.epoch)
            self.epoch += 1

        timer.tic()
        for epoch_idx in range(self.epoch, self.args.epochs + 1):

            self.epoch = epoch_idx

            self.model.train()
            for iter_idx, (inputs, masks) in enumerate(tqdm(self.train_loader)):

                inputs = inputs.to(self.args.device)
                masks = masks.to(self.args.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs,masks.long())

                epoch_loss(loss.item())
                mIoU(get_miou(outputs, masks))
                bIoU(get_biou(outputs, masks))

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            lr = check_lr(self.optimizer)
            print("Epoch:{}\t train: mIoU:{:.3f}\t boundary_IoU:{:.3f}\t loss:{:.4f}\tLR:{:.9f}\n".format(epoch_idx,
                                                                                                          mIoU.average(),
                                                                                                          bIoU.average(),
                                                                                                          epoch_loss.average(),
                                                                                                          lr))
            epoch_loss.reset()
            mIoU.reset()
            bIoU.reset()

            test_mIoU = self.test()
            if test_mIoU >= best_mIoU:
                save_name = os.path.join(self.args.model_save_dir, 'model_best.pkl')
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'epoch': epoch_idx}, save_name)

            if epoch_idx % self.args.save_freq == 0:
                save_name = os.path.join(self.args.model_save_dir, 'model_%d.pkl' % epoch_idx)
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'epoch': epoch_idx}, save_name)

        print('Done! \n Elpased time:{.2f} min'.format(timer.toc()))

        return


    def test(self):

        print('Start %d epoch test...' % self.epoch)

        test_loss = Adder()
        mIoU = Adder()
        bIoU = Adder()

        self.model.eval()
        with torch.no_grad():
            for idx, (_, inputs, masks) in enumerate(self.test_loader):
                inputs, masks = inputs.cuda(), masks.cuda()#to(self.args.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, masks.long())
                test_loss(loss.item())
                mIoU(get_miou(outputs, masks))
                bIoU(get_biou(outputs, masks))


        print("test_miou:{:.3f}\t test_boundary_iou:{:.3f}\t test:loss:{:.4f}\n".format(mIoU.average(),
                                                                                        bIoU.average(),
                                                                                        test_loss.average()))

        return mIoU.average()

