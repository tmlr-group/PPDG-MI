import numpy as np
import os
import random
import time
import torch
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torchvision
from evaluation import evaluate_results, write_precision_list
from attack import PLG_inversion
from utils import perform_final_selection, load_json, get_GAN, load_model
from dataset import FaceDataset
from engine import tune_cgan

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = ArgumentParser(description='Inversion')
parser.add_argument('--private_data_name', type=str, default='celeba', help='celeba | ffhq | facescrub')
parser.add_argument('--public_data_name', type=str, default='ffhq', help='celeba | ffhq | facescrub')
parser.add_argument('--iterations', type=int, default=600)
parser.add_argument('--exp_name',
                    default="baseline_id0-99",
                    type=str,
                    help='Directory to save output files (default: None)')
parser.add_argument('--num_round', type=int, default=1, help='Description of number of round')
parser.add_argument('--num_candidates', type=int, default=100, help='Description of number of candidates')
parser.add_argument('--target_classes', type=str, default='0-100', help='Description of target classes')
parser.add_argument('--results_root', type=str, default='results',
                    help='Path to results directory. default: results')
parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                    help='Apply relativistic loss or not. default: False')
args = parser.parse_args()


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        opt_z = PLG_inversion(args, G, D, T, E, batch_size, targets_single_id, lr=cfg["attack"]["lr"], MI_iter_times=iterations)
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
    # load json file
    json_file = f"./config/{args.private_data_name}/attacking/{args.public_data_name}.json"
    cfg = load_json(json_file=args.configs)

    set_random_seed(42)

    prefix = os.path.join(cfg["root_path"], "plg")
    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]),
                               cfg["attack"]["inv_loss_type"])
    prefix = os.path.join(prefix, save_folder)

    # parameters
    n_classes = cfg['dataset']['n_classes']
    z_dim = cfg['attack']['z_dim']
    dis_num_features = cfg['GAN_configuration']['gen_num_features']
    gen_num_features = cfg['GAN_configuration']['gen_num_features']
    gen_bottom_width = cfg['GAN_configuration']['gen_bottom_width']
    gen_distribution = cfg['GAN_configuration']['gen_distribution']
    tune_iter_times = cfg['GAN_configuration']['tune_iter_times']

    # load GAN
    G, D = get_GAN(cfg['dataset']['name'], gan_model_dir=cfg['train']['gan_model_dir'],
                   z_dim=100, n_classes=n_classes, device=device,
                   gen_num_features=gen_num_features, gen_bottom_width=gen_bottom_width, gen_distribution=gen_distribution, dis_num_features=dis_num_features)
    G = G.cuda()
    D = D.cuda()
    G.eval()
    D.eval()

    original_G = deepcopy(G)
    original_D = deepcopy(D)

    # Load target model
    T = load_model(model_name=cfg['train']['model_type'], path_T=cfg['train']['cls_ckpt'], num_classes=n_classes)
    T.eval()

    # Load evaluation model
    E = load_model(model_name=cfg['train']['eval_model'], path_T=cfg['train']['eval_dir'], num_classes=n_classes)
    E.eval()

    # Attack Parameters
    num_candidates = args.num_candidates
    samples_per_target = args.num_candidates // 4
    target_classes = args.target_classes
    start, end = map(int, target_classes.split('-'))
    targets = torch.tensor([i for i in range(start, end)])
    targets = torch.repeat_interleave(targets, num_candidates)
    targets = targets.to(device)
    batch_size = 100

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = current_time + '_' + args.exp_name if args.exp_name is not None else current_time

    # dataset crop setting'
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

    train_dataset = FaceDataset(args=args, root=cfg["dataset"]["img_path"], transform=my_transform,
                                data_name=args.private_data_name)

    iterations = args.iterations
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

            G, D = tune_cgan(args, cfg, G, D, T, final_z[:samples_per_target], final_targets[:samples_per_target],
                             gan_max_iteration=tune_iter_times)

            tune_time = time.time() - start_time

            time_cost_list = [['target', 'mi', 'selection', 'tune_time'],
                              [target_id, time_list[0], time_list[1], tune_time]]

            _ = write_precision_list(
                f'{prefix}/{current_time}/time_cost_r{cur_round + 1}',
                time_cost_list
            )