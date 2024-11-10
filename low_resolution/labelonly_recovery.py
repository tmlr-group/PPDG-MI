from engine import tune_general_gan
from utils import *
from evaluation import evaluate_results, write_precision_list
from pathlib import Path
import torch
import os
from attack import BREP_inversion
from argparse import ArgumentParser
from copy import deepcopy

torch.manual_seed(42)

parser = ArgumentParser(description='Inversion')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/celeba.json')
parser.add_argument('--exp_name',
                    default="baseline_id0-99",
                    type=str,
                    help='Directory to save output files (default: None)')
parser.add_argument('--iterations', type=int, default=1200, help='Description of iterations')
parser.add_argument('--num_round', type=int, default=1, help='Description of number of round')
parser.add_argument('--num_epoch', type=int, default=10, help='Description of epoch of GAN fine-tuning')
parser.add_argument('--num_candidates', type=int, default=1000, help='Description of number of candidates')
parser.add_argument('--target_classes', type=str, default='0-100', help='Description of target classes')
parser.add_argument('--max_radius', type=float, default=16.3, help='Description of max_radius to stop in BREP-MI')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def labelonly_attack(attack_params, G, target_model, E, z, targets_single_id, target_id, max_iters_at_radius_before_terminate, max_radius, used_loss, round_num=0):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if round_num == 0:
        final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}.pt"
    else:
        final_z_path = f"{prefix}/final_z/round{round_num}_{target_id:03d}.pt"

    if os.path.exists(final_z_path):
        print(f"Load data from: {final_z_path}.")
        mi_time = 0
        opt_z = torch.load(final_z_path)
    else:
        print(f"File {final_z_path} does not exist, skipping load.")
        mi_start_time = time.time()
        opt_z = BREP_inversion(z, target_id, targets_single_id, G, target_model, E, attack_params, used_loss,
                            max_iters_at_radius_before_terminate, max_radius, initial_radius, expansion_coeff, save_dir)

        mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = perform_final_selection(
        opt_z,
        G,
        targets_single_id,
        target_model,
        samples_per_target=num_candidates,
        device=device,
        batch_size=batch_size,
    )
    # no selection
    # final_z, final_targets = opt_z, targets_single_id
    selection_time = time.time() - start_time

    if round_num == 0:
        final_z_path = f"{prefix}/final_z/baseline_{target_id:03d}.pt"
    else:
        final_z_path = f"{prefix}/final_z/round{round_num}_{target_id:03d}.pt"
    torch.save(final_z.detach(), final_z_path)

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, round_num, current_time, prefix, final_z, final_targets, trainset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]


if __name__ == "__main__":
    cfg = load_json(json_file=args.configs)

    attack_method = cfg["attack"]["method"]

    # Save dir
    prefix = os.path.join(cfg["root_path"], attack_method)

    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]),
                               cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = os.path.join(prefix, "latent")
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))
    args.log_path = os.path.join(prefix, "invertion_logs")

    train_file = cfg['dataset']['train_file_path']
    print("load training data!")
    trainset, trainloader = init_dataloader(cfg, train_file, mode="train")

    # Load models
    targetnets, E, G, D, n_classes, fea_mean, fea_logvar = get_attack_model(args, cfg)
    original_G = deepcopy(G)
    original_D = deepcopy(D)

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
    dataset_name = cfg['dataset']['name']
    model_name = cfg['dataset']['model_name']
    z_dim = cfg['attack']['z_dim']

    initial_radius = 2.0
    expansion_coeff = 1.3
    batch_dim_for_initial_points = cfg['BREP_MI']['batch_dim_for_initial_points']
    point_clamp_min = cfg['BREP_MI']['point_clamp_min']
    point_clamp_max = cfg['BREP_MI']['point_clamp_max']
    max_iters_at_radius_before_terminate = args.iterations
    max_radius = args.max_radius

    mode = "general"

    iterations = args.iterations
    num_round = args.num_round

    for target_id in sorted(list(set(targets.tolist()))):
        G = deepcopy(original_G)
        D = deepcopy(original_D)
        for round in range(num_round):
            print(f"\nAttack target class: [{target_id}] round number: [{round}]")
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
                                                num_candidates,
                                                target_id)

                if round > 0:
                    initial_radius = cfg['BREP_MI']['current_sphere_radius']
                    expansion_coeff = cfg['BREP_MI']['sphere_expansion_coeff']

                final_z, final_targets, time_list = labelonly_attack(cfg, G, targetnets[0], E, z,
                                                                      targets_single_id, target_id,
                                                                      max_iters_at_radius_before_terminate, max_radius,
                                                                      initial_radius, expansion_coeff,
                                                                      used_loss="ce_loss",
                                                                      round_num=round)

            else:
                print("Attack method does not match this python program.")
                break

            print(f"Select a total of {samples_per_target} images from {num_candidates} images for the target classes {target_id}.\n")
            selected_z = final_z[:samples_per_target]
            selected_targets = final_z[:samples_per_target]

            if round < num_round - 1 :
                print("Start GAN fine-tuning.")

                start_time = time.time()
                json_path = f"./config/celeba/training_GAN/{mode}_gan/{dataset_name}.json"
                with open(json_path, 'r') as f:
                    config = json.load(f)

                G, D = tune_general_gan(config, G, D, selected_z, epochs=args.num_epoch)

                tune_time = time.time() - start_time

                time_cost_list = [['target', 'mi', 'selection', 'tune_time'],
                                [target_id, time_list[0], time_list[1], tune_time]]

                _ = write_precision_list(
                    f'{prefix}/{current_time}/time_cost_r{round + 1}',
                    time_cost_list
                )
            else:
                print("Final round reached, GAN fine-tuning skipped.")