import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torchvision
from evaluation import evaluate_results, write_precision_list
from attack import PLG_inversion

from utils import perform_final_selection
from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
from models.generators.resnet64 import ResNetGenerator
from models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from dataset import FaceDataset
from engine import tune_cgan

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)

# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def white_box_attack(args, G, D, T, E, targets_single_id, iterations=600, round_num=1):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if round_num == 0:
        final_z_path = f"{save_dir}/baseline_{target_id:03d}.pt"
    else:
        final_z_path = f"{save_dir}/round{round_num}_{target_id:03d}.pt"

    mi_start_time = time.time()
    if round_num == 0 and os.path.exists(final_z_path):
        opt_z = torch.load(final_z_path)
        print(f"Loaded data from {final_z_path}")
    else:
        print(f"File {final_z_path} does not exist, skipping load.")
        opt_z = PLG_inversion(args, G, D, T, E, batch_size, targets_single_id, lr=args.lr, MI_iter_times=iterations)
        torch.save(opt_z.detach(), final_z_path)

    mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = perform_final_selection(
        opt_z,
        G,
        targets_single_id,
        T,
        samples_per_target=num_candidates,
        device=device,
        batch_size=batch_size,
    )
    # no selection
    # final_z, final_targets = opt_z, targets_single_id
    selection_time = time.time() - start_time

    print(f'Selected a total of {final_z.shape[0]} final images out of {opt_z.shape[0]} images',
          f'of target classes {set(final_targets.cpu().tolist())}.')

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, round_num, current_time, prefix, final_z, final_targets, train_dataset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Stage-2: Image Reconstruction')
    parser.add_argument('--public_data_root', type=str, default='reclassified_public_data/ffhq/VGG16_top30',
                        help='path to dataset root directory.')
    parser.add_argument('--train_data_root', type=str, default='datasets/celeba_private_domain',
                        help='path to dataset root directory.')
    parser.add_argument('--private_data_name', type=str, default='celeba', help='celeba | ffhq | facescrub')
    parser.add_argument('--public_data_name', type=str, default='ffhq', help='celeba | ffhq | facescrub')
    parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--MI_iter_times', type=int, default=600)

    # path
    parser.add_argument('--save_dir', type=str, default='PLG_MI_Inversion')
    parser.add_argument('--path_G', type=str, default='')
    parser.add_argument('--path_D', type=str, default='')

    # Inversion
    parser.add_argument('--exp_name',
                        default="baseline_id0-99",
                        type=str,
                        help='Directory to save output files (default: None)')
    # parser.add_argument('--iterations', type=int, default=600, help='Description of iterations')
    parser.add_argument('--num_round', type=int, default=1, help='Description of number of round')
    parser.add_argument('--num_candidates', type=int, default=100, help='Description of number of candidates')
    parser.add_argument('--target_classes', type=str, default='0-100', help='Description of target classes')
    parser.add_argument('--inv_loss_type', type=str, default='margin', help='ce | margin | poincare')
    parser.add_argument('--alpha', type=float, default=0.2, help='weight of inv loss. default: 0.2')

    # Log and Save interval configuration
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')

    # tune cGAN
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    parser.add_argument('--tune_iter_times', type=int, default=100)

    # Discriminator (Critic) configuration
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')

    # Optimizer settings
    parser.add_argument('--tune_cGAN_lr', type=float, default=0.0002,
                        help='Initial learning rate of Adam. default: 0.0002')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 (betas[0]) value of Adam. default: 0.0')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')
    parser.add_argument('--batch_size', '-B', type=int, default=64,
                        help='mini-batch size of training data. default: 64')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Interval of showing losses. default: 100')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of discriminator updater per generator updater. default: 5')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='loss function name. hinge (default) or dcgan.')
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                        help='Apply relativistic loss or not. default: False')
    parser.add_argument('--num_classes', '-nc', type=int, default=1000,
                        help='Number of classes in training data.  default: 1000')
    args = parser.parse_args()
    logger = get_logger()

    logger.info(args)
    logger.info("=> creating model ...")

    set_random_seed(42)

    prefix = "attack_results/PLG"

    # load Generator
    G = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        num_classes=1000, distribution=args.gen_distribution
    )
    gen_ckpt_path = args.path_G
    gen_ckpt = torch.load(gen_ckpt_path)['model']
    G.load_state_dict(gen_ckpt)
    G = G.cuda()
    G.eval()

    D = SNResNetProjectionDiscriminator(args.dis_num_features, 1000, F.relu).to(device)
    disc_ckpt_path = args.path_D
    disc_ckpt = torch.load(disc_ckpt_path)['model']
    D.load_state_dict(disc_ckpt)
    D = D.cuda()
    D.eval()

    original_G = deepcopy(G)
    original_D = deepcopy(D)

    # Load target model
    if args.model.startswith("VGG16"):
        T = VGG16(1000)
        path_T = '../checkpoints/target_model/target_ckp/VGG16_88.26.tar'
    elif args.model.startswith('IR152'):
        T = IR152(1000)
        path_T = '../checkpoints/target_model/target_ckp/IR152_91.16.tar'
    elif args.model == "FaceNet64":
        T = FaceNet64(1000)
        path_T = '../checkpoints/target_model/target_ckp/FaceNet64_88.50.tar'
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    # Load evaluation model
    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).cuda()
    path_E = '..checkpoints/target_model/target_ckp/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)

    # Attack Parameters
    num_candidates = args.num_candidates
    samples_per_target = args.num_candidates // 2
    target_classes = args.target_classes
    start, end = map(int, target_classes.split('-'))
    targets = torch.tensor([i for i in range(start, end)])
    targets = torch.repeat_interleave(targets, num_candidates)
    targets = targets.to(device)
    batch_size = 100

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = current_time + '_' + args.exp_name if args.exp_name is not None else current_time

    # dataset crop setting
    if args.private_data_name == 'celeba':
        re_size = 64
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    elif args.private_data_name == 'ffhq':
        crop_size = 88
        offset_height = (128 - crop_size) // 2
        offset_width = (128 - crop_size) // 2
        re_size = 64
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    elif args.private_data_name == 'facescrub':
        re_size = 64
        crop_size = 64
        offset_height = (64 - crop_size) // 2
        offset_width = (64 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    else:
        print("Wrong Dataname!")

    # load public dataset
    my_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(crop),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((re_size, re_size)),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = FaceDataset(args=args, root=args.train_data_root, transform=my_transform,
                                data_name=args.private_data_name)

    logger.info("=> Begin attacking ...")

    iterations = args.MI_iter_times
    num_round = args.num_round

    for target_id in sorted(list(set(targets.tolist()))):
        G = deepcopy(original_G)
        D = deepcopy(original_D)
        for cur_round in range(num_round):
            print(f"Target class: [{target_id}] round number: [{cur_round + 1}]")
            targets_single_id = targets[torch.where(targets == target_id)[0]].to(device)

            final_z, final_targets, time_list = white_box_attack(args, G, D, T, E, targets_single_id,
                                                           iterations=iterations,
                                                           round_num=cur_round)

            if cur_round == num_round - 1:
                break

            print("GAN Fine-tuning")

            start_time = time.time()

            G, D = tune_cgan(args, G, D, T, final_z[:samples_per_target], final_targets[:samples_per_target],
                             gan_max_iteration=args.tune_iter_times)

            tune_time = time.time() - start_time

            time_cost_list = [['target', 'mi', 'selection', 'tune_time'],
                              [target_id, time_list[0], time_list[1], tune_time]]

            _ = write_precision_list(
                f'{prefix}/{current_time}/time_cost_r{cur_round + 1}',
                time_cost_list
            )