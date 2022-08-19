from option import args
from Trainer import Trainer
from Evaluation import evaluation
from dataset.horseDataset import make_dataloader
from models.denseaspp import DenseASPP

def main(args):

    train_loader, test_loader = make_dataloader(args)
    model = DenseASPP(args.num_class).to(args.device)
    t = Trainer(args, model, train_loader, test_loader)

    if args.phase == 'train':
        t.train()
    elif args.phase == 'eval':
        evaluation(args, model, test_loader)


if __name__ == '__main__':
    print(args)
    main(args)