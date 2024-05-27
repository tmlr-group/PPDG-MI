from engine import tune_specific_gan, tune_general_gan
from utils import *
from models.classify import *
from evaluation import evaluate_results, write_precision_list
from pathlib import Path
import torch
import os
import numpy as np
from attack import white_inversion, white_dist_inversion, black_inversion, label_only_inversion
from argparse import ArgumentParser
from copy import deepcopy
from SAC import Agent

torch.manual_seed(9)

parser = ArgumentParser(description='Inversion')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/celeba.json')
parser.add_argument('--exp_name',
                    default=None,
                    type=str,
                    help='Directory to save output files (default: None)')
parser.add_argument('--iter_times', type=int, default=1200, help='Description of iter_times')
parser.add_argument('-N', type=int, default=1, help='Description of N')
parser.add_argument('--num_candidates', type=int, default=1000, help='Description of num_candidates')
parser.add_argument('--target_classes', type=str, default='0-100', help='Description of target classes')
parser.add_argument("-max_episodes", type=int, default=10000)
parser.add_argument("-max_step", type=int, default=1)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-alpha", type=float, default=0)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_attack_args(cfg):
    if cfg["attack"]["method"] == 'kedmi':
        args.improved_flag = True
        args.clipz = True
        args.num_seeds = 1
    else:
        args.improved_flag = False
        args.clipz = False
        args.num_seeds = 5

    if cfg["attack"]["variant"] == 'L_logit' or cfg["attack"]["variant"] == 'ours':
        args.loss = 'logit_loss'
    elif cfg["attack"]["variant"] == 'cs':
        args.loss = 'cs_loss'
    else:
        args.loss = 'cel'

    if cfg["attack"]["variant"] == 'L_aug' or cfg["attack"]["variant"] == 'ours':
        args.classid = '0,1,2,3'
    else:
        args.classid = '0'


def white_attack(target_model, z, G, D, E, targets_single_id, used_loss, iter_times=2400, round_num=0):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}"
    if round_num == 0 and os.path.exists(final_z_path):
        print(f"Load final_z from: {final_z_path}")
        mi_time = 0
        all_final_w = torch.load(final_z_path)
        num_vectors_per_category = 1000
        id = int(targets_single_id[0])
        opt_z = all_final_w[id * num_vectors_per_category:(id + 1) * num_vectors_per_category]

    else:
        print("No final_z loading")
        mi_start_time = time.time()
        if args.improved_flag:
            opt_z = white_dist_inversion(G, D, target_model, E, targets_single_id[:batch_size], batch_size,
                                         num_candidates,
                                         used_loss=used_loss,
                                         fea_mean=fea_mean,
                                         fea_logvar=fea_logvar,
                                         iter_times=iter_times,
                                         improved=True,
                                         lam=cfg["attack"]["lam"])
        else:
            opt_z = white_inversion(G, D, target_model, E, batch_size, z, targets_single_id,
                                    used_loss=used_loss,
                                    fea_mean=fea_mean,
                                    fea_logvar=fea_logvar,
                                    iter_times=iter_times,
                                    lam=cfg["attack"]["lam"])

        mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = opt_z, targets_single_id
    # final_z, final_targets = perform_final_selection(
    #     opt_z,
    #     G,
    #     targets_single_id,
    #     target_model[0],
    #     samples_per_target=samples_per_target,
    #     device=device,
    #     batch_size=batch_size,
    # )
    selection_time = time.time() - start_time

    if round_num == 0:
        final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}.pt"
    else:
        final_z_path = f"{prefix}/final_z/round{round_num}_{target_id:03d}.pt"
    torch.save(final_z.detach(), final_z_path)

    print(f'Selected a total of {final_z.shape[0]} final images out of {opt_z.shape}',
          f'of target classes {set(final_targets.cpu().tolist())}.')

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, round_num, current_time, prefix, final_z, final_targets, trainset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]


def black_attack(agent, G, target_model, alpha, z, max_episodes, max_step, targets_single_id, round_num=0):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}"
    if round_num == 0 and os.path.exists(final_z_path):
        print(f"Load final_z from: {final_z_path}")
        mi_time = 0
        all_final_w = torch.load(final_z_path)
        num_vectors_per_category = 1000
        id = int(targets_single_id[0])
        opt_z = all_final_w[id * num_vectors_per_category:(id + 1) * num_vectors_per_category]

    else:
        print("No final_z Loading")
        mi_start_time = time.time()
        opt_z = black_inversion(agent, G, target_model, alpha, z, batch_size, max_episodes, max_step,
                                targets_single_id[0], model_name)
        mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = opt_z, targets_single_id
    # final_z, final_targets = perform_final_selection(
    #     opt_z,
    #     G,
    #     targets_single_id,
    #     target_model[0],
    #     samples_per_target=samples_per_target,
    #     device=device,
    #     batch_size=batch_size,
    # )
    selection_time = time.time() - start_time

    if round_num == 0:
        final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}.pt"
    else:
        final_z_path = f"{prefix}/final_z/round{round_num}_{target_id:03d}.pt"
    torch.save(final_z.detach(), final_z_path)

    print(f'Selected a total of {final_z.shape[0]} final images out of {opt_z.shape}',
          f'of target classes {set(final_targets.cpu().tolist())}.')

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, round_num, current_time, prefix, final_z, final_targets, trainset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]


def label_only_attack(attack_params, criterion, G, target_model, E, z, targets_single_id, target_id, round_num=0):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}"
    if round_num == 0 and os.path.exists(final_z_path):
        print(f"Load final_z from: {final_z_path}")
        mi_time = 0
        all_final_w = torch.load(final_z_path)
        num_vectors_per_category = 1000
        id = int(targets_single_id[0])
        opt_z = all_final_w[id * num_vectors_per_category:(id + 1) * num_vectors_per_category]

    else:
        print("No Loading")
        mi_start_time = time.time()
        opt_z = label_only_inversion(z, target_id, targets_single_id, G, target_model, E, attack_params, criterion,
                                     save_dir)
        mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = opt_z, targets_single_id
    # final_z, final_targets = perform_final_selection(
    #     opt_z,
    #     G,
    #     targets_single_id,
    #     target_model[0],
    #     samples_per_target=samples_per_target,
    #     device=device,
    #     batch_size=batch_size,
    # )
    selection_time = time.time() - start_time

    if round_num == 0:
        final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}.pt"
    else:
        final_z_path = f"{prefix}/final_z/round{round_num}_{target_id:03d}.pt"
    torch.save(final_z.detach(), final_z_path)

    print(f'Selected a total of {final_z.shape[0]} final images out of {opt_z.shape}',
          f'of target classes {set(final_targets.cpu().tolist())}.')

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, round_num, current_time, prefix, final_z, final_targets, trainset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]


if __name__ == "__main__":
    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)

    attack_method = cfg["attack"]["method"]
    # Save dir
    if args.improved_flag == True:
        prefix = os.path.join(cfg["root_path"], "kedmi")
    else:
        prefix = os.path.join(cfg["root_path"], attack_method)

    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]),
                               cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = os.path.join(prefix, "latent")
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))
    args.log_path = os.path.join(prefix, "invertion_logs")

    os.makedirs(prefix, exist_ok=True)
    os.makedirs(f"{prefix}/final_z", exist_ok=True)
    print(f"{prefix}/final_z")
    os.makedirs(args.log_path, exist_ok=True)

    train_file = cfg['dataset']['train_file_path']
    print("load training data!")
    trainset, trainloader = utils.init_dataloader(cfg, train_file, mode="train")

    # Load models
    targetnets, E, G, D, n_classes, fea_mean, fea_logvar = get_attack_model(args, cfg)
    original_G = deepcopy(G)
    original_D = deepcopy(D)

    num_candidates = args.num_candidates
    samples_per_target = args.num_candidates
    target_classes = args.target_classes
    start, end = map(int, target_classes.split('-'))
    targets = torch.tensor([i for i in range(start, end)])
    targets = torch.repeat_interleave(targets, num_candidates)
    targets = targets.to(device)
    batch_size = 100

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = current_time + '_' + args.exp_name if args.exp_name is not None else current_time
    dataset_name = cfg['dataset']['name']
    model_name = cfg['dataset']['model_name']

    max_step = cfg['RLB_MI']['max_step']
    seed = cfg['RLB_MI']['seed']
    alpha = cfg['RLB_MI']['alpha']
    n_target = cfg['train']['Nclass']
    max_episodes = args.iter_times

    z_dim = cfg['BREP_MI']['z_dim']
    batch_dim_for_initial_points = cfg['BREP_MI']['batch_dim_for_initial_points']
    targeted_attack = cfg['BREP_MI']['targeted_attack']
    iden_range_min = cfg['BREP_MI']['iden_range_min']
    iden_range_max = cfg['BREP_MI']['iden_range_max']
    point_clamp_min = cfg['BREP_MI']['point_clamp_min']
    point_clamp_max = cfg['BREP_MI']['point_clamp_max']

    if args.improved_flag:
        mode = "specific"
    else:
        mode = "general"

    iter_times = args.iter_times
    N = args.N

    for target_id in sorted(list(set(targets.tolist()))):
        G = deepcopy(original_G)
        D = deepcopy(original_D)
        for round_num in range(N):
            print(f"Target class: [{target_id}] round number: [{round_num}]")
            targets_single_id = targets[torch.where(targets == target_id)[0]].to(device)

            if attack_method == "brep":
                toogle_grad(G, False)
                toogle_grad(D, False)
                z = gen_initial_points_targeted(batch_dim_for_initial_points,
                                                G,
                                                targetnets[0],
                                                point_clamp_min,
                                                point_clamp_max,
                                                z_dim,
                                                len(targets_single_id),
                                                target_id)
                criterion = nn.CrossEntropyLoss().cuda()
                final_z, final_targets, time_list = label_only_attack(cfg, criterion, G, targetnets[0], E, z,
                                                                      targets_single_id, target_id,
                                                                      round_num=round_num)

            elif attack_method == 'rlb':
                z = torch.randn(len(targets_single_id), 100).to(device).float()
                agent = Agent(state_size=z_dim, action_size=z_dim, random_seed=seed, hidden_size=256,
                              action_prior="uniform")
                final_z, final_targets, time_list = black_attack(agent, G, targetnets[0], alpha, z,
                                                                 max_episodes,
                                                                 max_step, targets_single_id,
                                                                 round_num=round_num)

            else:
                z = torch.randn(len(targets_single_id), 100).to(device).float()
                final_z, final_targets, time_list = white_attack(targetnets, z, G, D, E, targets_single_id,
                                                                 used_loss=args.loss,
                                                                 iter_times=iter_times,
                                                                 round_num=round_num)

            print("GAN Fine-tuning")

            start_time = time.time()
            if args.improved_flag:
                json_path = f"./config/celeba/training_GAN/{mode}_gan/{dataset_name}.json"
                with open(json_path, 'r') as f:
                    config = json.load(f)
                config["VGG16"]["epochs"] = 10
                with open(json_path, 'w') as f:
                    json.dump(config, f, indent=8)

                G, D = tune_specific_gan(config, G, D, targetnets[0], final_z[:samples_per_target], epochs=10)
            else:
                json_path = f"./config/celeba/training_GAN/{mode}_gan/{dataset_name}.json"
                with open(json_path, 'r') as f:
                    config = json.load(f)
                config["train_gan_first_stage"]["epochs"] = 10
                with open(json_path, 'w') as f:
                    json.dump(config, f, indent=8)

                G, D = tune_general_gan(config, G, D, final_z[:samples_per_target], epochs=10)
            tune_time = time.time() - start_time

            time_cost_list = [['target', 'mi', 'selection', 'tune_time'],
                              [target_id, time_list[0], time_list[1], tune_time]]

            _ = write_precision_list(
                f'{prefix}/{current_time}/time_cost_r{round_num + 1}',
                time_cost_list
            )