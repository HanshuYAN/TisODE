import argparse

def str2list(isFloat, s, split='_'):
    l = s.split('_')
    if isFloat:
        l = [float(x) for x in l]
    else:
        l = [int(x) for x in l]
    return l

def str2bool(s):
    assert s in ['True', 'False']
    if s == 'True':
        return True
    else:
        return False

""" MNIST Robustness """
# Params
parser = argparse.ArgumentParser(description='Robustness')
parser.add_argument('--isTrain', type=str, default='False')
parser.add_argument('--isODE', type=str, default='True')
parser.add_argument('--isSS', type=str, default='False')
parser.add_argument('--isRandom', type=str, default='True')
parser.add_argument('--device_ids', type=str, default='0_1_2_3')

args, _ = parser.parse_known_args()

parser.add_argument('--CNN_steps', default=10, type=int)
# only for ode method
if args.isODE == 'True':
    parser.add_argument('--rtol', default=1e-3, type=float)
    parser.add_argument('--atol', default=1e-3, type=float)
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--step_size', default=0.1, type=float) # default 0.1, 10 steps
    parser.add_argument('--TimePeriod', default='0_1_2', type=str)
    if args.isSS == 'True':
        parser.add_argument('--epoch_steady', default=5, type=int)
        parser.add_argument('--w_steady', default=0.01, type=float)

if args.isTrain == 'True':
    # For Training
    parser.add_argument('--exp_name', default='MNIST_ODE', type=str)
    parser.add_argument('--resume', type=str, default='True')
    # optimization
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=100, type=int)
    parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--milestones', default='30_60_90', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.2)
    # log
    parser.add_argument('--dir_logging',type=str, default='../../checkpoints/mnist')

elif args.isTrain == 'False':
    # For testing
    parser.add_argument('--dir_model', default='../../checkpoints/.', type=str)
    parser.add_argument('--noisy', default=True, type=bool)
    parser.add_argument('--noise_level', default=100, type=int)
    parser.add_argument('--dir_logging',type=str, default='../../results/MNIST')
    parser.add_argument('--logging_file',type=str, default='.')

args = parser.parse_args()

# Parse list and boolen
vars(args)['device_ids'] = str2list(isFloat=False, s=args.device_ids)
vars(args)['isTrain'] = str2bool(args.isTrain)
vars(args)['isODE'] = str2bool(args.isODE)
vars(args)['isRandom'] = str2bool(args.isRandom)

if args.isODE == True:
    vars(args)['TimePeriod'] = str2list(isFloat=True, s=args.TimePeriod)
if args.isTrain == True:
    vars(args)['resume'] = str2bool(args.resume)
    vars(args)['milestones'] = str2list(isFloat=False, s=args.milestones)

if __name__ == '__main__':
    import matplotlib.pyplot as plt; import pdb; pdb.set_trace()
    pass
