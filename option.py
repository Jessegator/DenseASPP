import torch
import argparse


parser = argparse.ArgumentParser(description='DenseASPP')

parser.add_argument('--use_gpu', default=True, help='whether to use gpu')
parser.add_argument('--device', action='store_true', help='cpu or gpu')

parser.add_argument('--resume', type=str, default='')
parser.add_argument('--pretrained', default=True, help='whether to use pretrained model')
parser.add_argument('--seed', type=int, default=42, help='random seed')

parser.add_argument('--phase', default='train',
                    choices=('train', 'eval'),
                    help='mode (train | eval)')

# Data specifications
parser.add_argument('--root', type=str, default='./weizmann_horse_db',
                    help='dataset directory')
parser.add_argument('--train_list', type=str, default='./dataset/train_list.txt',
                    help='dataset directory')
parser.add_argument('--test_list', type=str, default='./dataset/test_list.txt',
                    help='dataset directory')
parser.add_argument('--workers', type=int, default=8,
                    help='num_workers for batch loader')

# Model specifications
parser.add_argument('--backbone', type=str, default='Densenet121',
                    choices=('Denset169', 'Densenet121'),
                    help='feature extractor')
parser.add_argument('--num_class', type=int, default=2,
                    help='number of classes')
# Training specifications
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--save_freq', type=int, default=1,
                    help='frequency of saving the model')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
# Optimization specifications

parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--scheduler_patience', type=float, default=6,
                    help='patience of scheduler')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='./pretrained/denseasppp.pkl',
                    help='file name to load')
parser.add_argument('--model_save_dir', type=str, default='./results/models/',
                    help='path of saved models')

args = parser.parse_args()

if args.use_gpu:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print('use_gpu')
else:
    args.device = torch.device('cpu')
    print('use_cpu')


