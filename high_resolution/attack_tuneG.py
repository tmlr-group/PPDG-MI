import argparse
import csv
import math
import random
from random import choice
import os
from tqdm import tqdm
import time

import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

from PIL import Image
import torchvision.transforms as transforms

from datetime import datetime
import torchvision.utils as vutils
from torch.nn.functional import cosine_similarity

import lpips
import numpy as np
import torch
import torchvision.transforms as T
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import dnnlib

from attacks.final_selection import perform_final_selection
from attacks.optimize import Optimization
from datasets.custom_subset import ClassSubset
from metrics.classification_acc import ClassificationAccuracy
from metrics.fid_score import FID_Score
from metrics.prcd import PRCD
from utils.attack_config_parser import AttackConfigParser
from utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)
from utils.stylegan import create_image, load_discrimator, load_generator
from utils.wandb import *
from utils import w_projector_lpips as w_projector

from losses import l2_loss
from losses import mmd_loss
from losses import ct_loss
from losses.localitly_regulizer import Space_Regulizer

from lpips import LPIPS
from utils.models_utils import toogle_grad, set_requires_grad, write_list
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load VGG16 feature detector.
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    parser.add_argument('-m', '--measure',
                        default='pointwise',
                        type=str)
    parser.add_argument('--exp_name',
                        default=None,
                        type=str,
                        help='Directory to save output files (default: None)')
    parser.add_argument('--num_inv_points',
                        type=int,
                        default=5,
                        help='An integer specifying the number of inv')
    parser.add_argument('--iterations', type=int, default=70, help='Optimization iterations per class')
    parser.add_argument('--num_round', type=int, default=1, help='Number of tuning round')

    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


def create_initial_vectors(config, G, target_model, targets):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)

        w_init = deepcopy(w)

    return w, w_init


def log_nearest_neighbors_local(imgs, targets, eval_model, dataset,
                                img_size, seed, save_dir, nrow, round_num):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(eval_model, None, img_size, None, dataset,
                                   seed)
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)

    grid = vutils.make_grid(closest_samples, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    img.save(os.path.join(save_dir, f'nearest_neighbor_r{round_num + 1}.png'))

    for i, (img, d) in enumerate(zip(closest_samples, distances)):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(((img + 1) * 0.5 * 255).astype('uint8'))
        img.save(os.path.join(save_dir, f'r{round_num + 1}_{i:02d}_target={target_id:03d}_distance_{d:.2f}.png'))

    return


def log_final_images_local(imgs, predictions, max_confidences, target_confidences,
                           idx2cls, save_dir, nrow, round_num):
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    img.save(os.path.join(save_dir, f'final_images_r{round_num + 1}.png'))

    for i, (img, pred, max_conf, target_conf) in enumerate(
            zip(imgs.cpu(), predictions, max_confidences, target_confidences)):
        img = img.permute(1, 2, 0).numpy()
        img = Image.fromarray(((img + 1) * 0.5 * 255).astype('uint8'))
        img.save(os.path.join(save_dir,
                              f'r{round_num + 1}_{i:02d}_target={target_id:03d} ({target_conf:.2f})_pred={pred.item():03d} ({max_conf:.2f}).png'))
    return


def calc_loss(generated_images, target_images, target_features, method, G=None, w_batch=None):
    loss = 0.0
    l2_loss_val = torch.tensor(-1)
    loss_dist = torch.tensor(0.0)
    ball_holder_loss_val = torch.tensor(-1)

    if config.tuneG['dist_lambda'] > 0:
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (generated_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

        if method == 'ct':
            loss_dist = ct_loss.ct(synth_features, target_features)
        elif method == 'mmd':
            loss_dist = mmd_loss.mmd(synth_features, target_features)
        else:
            loss_dist = lpips_loss(generated_images, target_images).mean()

        loss += loss_dist * config.tuneG['dist_lambda']

    if config.tuneG['use_locality_regularization']:
        ball_holder_loss_val = space_regulizer.space_regulizer_loss(G, w_batch)
        loss += ball_holder_loss_val

    return loss, l2_loss_val, loss_dist, ball_holder_loss_val


def attack_single_id(target_model, w, synthesis, discriminator, targets_single_id):
    if config.logging:
        save_dir = os.path.join('results', current_time, f'{target_id: 03d}')
        Path(f"results/{current_time}/{target_id:03d}").mkdir(parents=True, exist_ok=True)

    ####################################
    #        Attack Preparation        #
    ####################################

    # Print attack configuration
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets_single_id.cpu().numpy()))}.')

    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.'
    )
    # Initialize RTPT
    rtpt = None
    if args.rtpt:
        max_iterations = math.ceil(w.shape[0] / batch_size) \
                         + int(math.ceil(w.shape[0] / (batch_size * 3))) \
                         + 2 * int(math.ceil(config.candidates['num_candidates']
                                             * len(set(targets_single_id.cpu().tolist())) / (batch_size * 3))) \
                         + 2 * len(set(targets_single_id.cpu().tolist()))
        rtpt = RTPT(name_initials='SPD',
                    experiment_name='Attack',
                    max_iterations=max_iterations)
        rtpt.start()

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()

    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(target_model, synthesis, discriminator,
                                attack_transformations, num_ws, config)

    # Collect results
    w_optimized = []

    start_time = time.time()

    final_w_path = f" "
    # final_w_path = f"results/final_w/CelebA_ResNet18_id_0_99_35iters.pt"
    # final_w_path = f"results/final_w/CelebA_resnest50_id_0_99_35iters.pt"
    if round_num == 0 and os.path.exists(final_w_path):
        all_final_w = torch.load(final_w_path)
        num_vectors_per_category = 100
        id = int(targets_single_id[0])
        w_optimized_unselected = all_final_w[id * num_vectors_per_category:(id + 1) * num_vectors_per_category]

        # final_w_path = f"results/20240430_120537_id_0_99_35iters_baseline/
        # {target_id:03d}/final_w_selected_{round_num + 1}.pt"
        # if round_num == 0 and os.path.exists(final_w_path):
        #     w_optimized_unselected = torch.load(final_w_path)

    else:
        # Prepare batches for attack
        for i in range(math.ceil(w.shape[0] / batch_size)):
            w_batch = w[i * batch_size:(i + 1) * batch_size].cuda()
            targets_batch = targets_single_id[i * batch_size:(i + 1) * batch_size].cuda()
            print(
                f'\nOptimizing batch {i + 1} of {math.ceil(w.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
            )

            # Run attack iteration
            torch.cuda.empty_cache()
            w_batch_optimized = optimization.optimize(w_batch, targets_batch,
                                                      num_epochs).detach().cpu()

            if rtpt:
                num_batches = math.ceil(w.shape[0] / batch_size)
                rtpt.step(subtitle=f'batch {i + 1} of {num_batches}')

            # Collect optimized style vectors
            w_optimized.append(w_batch_optimized)

        # Concatenate optimized style vectors
        w_optimized_unselected = torch.cat(w_optimized, dim=0)
        torch.cuda.empty_cache()
        del discriminator

    mi_time = time.time() - start_time

    ####################################
    #          Filter Results          #
    ####################################

    start_time = time.time()
    # Filter results
    if config.final_selection:
        print(
            f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
            f'images per target using {config.final_selection["approach"]} approach.'
        )
        final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets_single_id,
            target_model,
            device=device,
            batch_size=batch_size * 10,
            **config.final_selection,
            rtpt=rtpt)
        print(f'Selected a total of {final_w.shape[0]} final images ',
              f'of target classes {set(final_targets.cpu().tolist())}.')
    else:
        final_w, final_targets = w_optimized_unselected, targets_single_id[:config.final_selection.samples_per_target]
    del target_model

    # final_w, final_targets = w_optimized_unselected, targets_single_id[:config.final_selection['samples_per_target']]
    # print(f"Selection execution time: {(time.time() - start_time):.4f} seconds")

    selection_time = time.time() - start_time

    # Log selected vectors
    if config.logging:
        optimized_w_path_selected = f"results/{current_time}/{target_id:03d}/final_w_selected_{round_num + 1}.pt"
        torch.save(final_w.detach(), optimized_w_path_selected)

    ####################################
    #         Attack Accuracy          #
    ####################################
    try:
        evaluation_model = config.create_evaluation_model()
        evaluation_model = torch.nn.DataParallel(evaluation_model)
        evaluation_model.to(device)
        evaluation_model.eval()
        class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                     device=device)

        acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            w_optimized_unselected,
            targets_single_id,
            synthesis,
            config,
            batch_size=batch_size * 2,
            resize=299,
            rtpt=rtpt)

        # Compute attack accuracy on filtered samples
        if config.final_selection:
            acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
                final_w,
                final_targets,
                synthesis,
                config,
                batch_size=batch_size * 2,
                resize=299,
                rtpt=rtpt)

            if config.logging:
                _ = write_list(
                    f'results/{current_time}/precision_list_r{round_num + 1}',
                    precision_list
                )

            print(
                f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
                f'accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
        del evaluation_model

    except Exception:
        print(traceback.format_exc())

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_facenet = None
    try:
        if target_dataset in [
            'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            # Load FaceNet model for face recognition
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # Compute average feature distance on facenet
            evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
                                                   config.attack_center_crop,
                                                   target_dataset, config.seed)
            avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                final_w,
                final_targets,
                batch_size=batch_size_single,
                rtpt=rtpt)
            if config.logging:
                _ = write_list(
                    f'results/{current_time}/distance_facenet_r{round_num + 1}',
                    mean_distances_list)
                # wandb.save(filename_distance)

            print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    except Exception:
        print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')

        num_classes = len(set(targets_single_id.tolist()))
        num_imgs = 10  # config.tuneG['num_inv_points']
        # Sample final images from the first and last classes
        label_subset = set(
            list(set(targets_single_id.tolist()))[:int(num_classes / 2)] +
            list(set(targets_single_id.tolist()))[-int(num_classes / 2):])

        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []
        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            w_masked = final_w[mask][:num_imgs]
            imgs = create_image(w_masked,
                                synthesis,
                                crop_size=config.attack_center_crop,
                                resize=config.attack_resize
                                )
            log_imgs.append(imgs)
            log_targets += [label for i in range(num_imgs)]
            log_predictions.append(torch.tensor(predictions)[mask][:num_imgs])
            log_max_confidences.append(
                torch.tensor(maximum_confidences)[mask][:num_imgs])
            log_target_confidences.append(
                torch.tensor(target_confidences)[mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0)
        log_max_confidences = torch.cat(log_max_confidences, dim=0)
        log_target_confidences = torch.cat(log_target_confidences, dim=0)

        log_final_images_local(log_imgs, log_predictions, log_max_confidences,
                               log_target_confidences, idx_to_class,
                               save_dir=f"results/{current_time}/{target_id:03d}", nrow=num_imgs, round_num=round_num)

        # Use FaceNet only for facial images
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        if target_dataset in [
            'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            log_nearest_neighbors_local(log_imgs,
                                        log_targets,
                                        facenet,
                                        target_dataset,
                                        img_size=160,
                                        seed=config.seed,
                                        save_dir=f"results/{current_time}/{target_id:03d}",
                                        nrow=num_imgs,
                                        round_num=round_num)

    return final_w, final_targets, [mi_time, selection_time]


if __name__ == '__main__':
    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    lpips_loss = LPIPS(net=config.tuneG['lpips_type']).to(device).eval()

    torch.set_num_threads(24)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:
        class KeyDict(dict):

            def __missing__(self, key):
                return key


        idx_to_class = KeyDict()

    # Load target model and set dataset
    target_model = config.create_target_model()
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset()

    # Load basic attack parameters
    num_epochs = config.attack['num_epochs'] = args.iterations
    N = config.attack['num_round'] = args.num_round

    batch_size_single = config.attack['batch_size']

    batch_size = config.attack['batch_size'] * torch.cuda.device_count()
    targets = config.create_target_vector()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model)
    D = load_discrimator(config.stylegan_model)
    num_ws = G.num_ws

    # Distribute models
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    G.synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    G.synthesis.num_ws = num_ws
    discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    original_G = copy.deepcopy(G)
    space_regulizer = Space_Regulizer(config, original_G, lpips_loss)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = current_time + '_' + args.exp_name if args.exp_name is not None else current_time

    if args.measure == 'mmd':
        config.tuneG['inv_steps'] = 100
        config.tuneG['tune_steps'] = 150
        config.tuneG['num_inv_points'] = args.num_inv_points
    elif args.measure == 'ct':
        config.tuneG['inv_steps'] = 300
        config.tuneG['tune_steps'] = 50
        config.tuneG['num_inv_points'] = args.num_inv_points
    else:
        config.tuneG['inv_steps'] = 300
        config.tuneG['tune_steps'] = 150
        config.tuneG['num_inv_points'] = args.num_inv_points

    # G_blocks_to_train = ['b4', 'b8', 'b16']
    # G_blocks_to_train = ['b4', 'b8', 'b16', 'b32', 'b64', 'b128']
    G_blocks_to_train = ['b4', 'b8', 'b16', 'b32', 'b64', 'b128', 'b256', 'b512', 'b1024']
    print(current_time)
    print('G_blocks_to_train:', G_blocks_to_train)

    all_final_w = [[] for _ in range(N)]
    all_final_targets = [[] for _ in range(N)]

    for target_id in sorted(list(set(targets.tolist()))):
        G = copy.deepcopy(original_G)

        for round_num in range(N):
            targets_single_id = targets[torch.where(targets == target_id)[0]].cpu()

            w, w_init = create_initial_vectors(config, G, target_model, targets_single_id)
            # step 1: model inversion & HQ data selection
            final_w, final_targets, time_list = attack_single_id(target_model, w, G.synthesis, discriminator,
                                                                 targets_single_id)
            all_final_w[round_num].append(final_w)
            all_final_targets[round_num].append(final_targets)

            # continue
            if round_num == N - 1:
                break

            # step3: tune generator
            pp_ws = final_w[:config.tuneG['num_inv_points']].to(device)
            pp_data = create_image(
                pp_ws,
                G.synthesis,
                crop_size=None,
                resize=None
            )
            # pp_data = torch.clamp(pp_data, min=-1, max=1)

            # Step 3.1
            start_time = time.time()
            ##################################################### CT ###############################################################
            if args.measure == 'ct':
                print('---> ct')
                w_pivots, dist_loss, discriminator_loss = w_projector.ct(G,
                                                                         discriminator,
                                                                         pp_data,
                                                                         device=device,
                                                                         w_avg_samples=600,
                                                                         num_steps=config.tuneG['inv_steps'],
                                                                         )

            ##################################################### MMD ###############################################################
            elif args.measure == 'mmd':
                print('---> mmd')
                w_pivots, dist_loss, discriminator_loss = w_projector.mmd(G,
                                                                          discriminator,
                                                                          pp_data,
                                                                          device=device,
                                                                          w_avg_samples=600,
                                                                          num_steps=config.tuneG['inv_steps'],
                                                                          )

            ##################################################### Point-wise ########################################################
            else:
                print('---> point-wise')
                w_pivots = []
                for j in range(pp_data.shape[0]):
                    target = pp_data[j]
                    w_pivot, dist_loss, discriminator_loss = w_projector.pointwise(G,
                                                                                   discriminator,
                                                                                   target,
                                                                                   # initial_w=pp_ws[j].unsqueeze(0),
                                                                                   device=device,
                                                                                   w_avg_samples=600,
                                                                                   num_steps=config.tuneG['inv_steps']
                                                                                   )
                    w_pivots.append(w_pivot)

                w_pivots = torch.cat(w_pivots, dim=0)

            step_3_1_time = time.time() - start_time

            inv_dist_list = [['target', 'dist_loss', 'discriminator_loss'],
                             [target_id, dist_loss.item(), discriminator_loss.item()]]

            if config.logging:
                _ = write_list(
                    f'results/{current_time}/inv_loss_r{round_num + 1}',
                    inv_dist_list)

            ###########################################################################################
            toogle_grad(G, False)

            for param in G.mapping.parameters():
                param.requires_grad = True

            set_requires_grad(G.synthesis.module, G_blocks_to_train, True)

            all_blocks = ['b4', 'b8', 'b16', 'b32', 'b64', 'b128', 'b256', 'b512', 'b1024']
            blocks_to_freeze = list(set(all_blocks) - set(G_blocks_to_train))
            set_requires_grad(G.synthesis.module, blocks_to_freeze, False)
            ###########################################################################################

            # toogle_grad(G, True)
            optimizer = torch.optim.Adam(G.parameters(), lr=config.tuneG['tune_learning_rate'])
            G.train()

            scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            target_images = ((pp_data + 1) * (255 / 2)).to(device)
            if target_images.shape[2] > 256:
                target_images = F.interpolate(target_images, size=(256, 256), mode='area')
            # Features for target images.
            target_features = vgg16(target_images, resize_images=False, return_lpips=True)

            # Step 3.2
            start_time = time.time()
            for i in tqdm(range(config.tuneG['tune_steps'])):
                # X_p
                GAN_inv_imgs = create_image(
                    w_pivots,
                    G.synthesis,
                    crop_size=None,  # config.attack_center_crop,
                    resize=None,  # config.attack_resize
                    require_grad=True
                )

                # X_p -> pp_data
                GAN_inv_imgs = GAN_inv_imgs.to(device)
                pp_data = pp_data.to(device)
                loss, l2_loss_val, loss_dist, ball_holder_loss_val = calc_loss(GAN_inv_imgs, pp_data, target_features,
                                                                               method=args.measure,
                                                                               G=G, w_batch=w_pivots)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i == 0 or (i + 1) % config.tuneG['tune_steps'] == 0:
                    grid = vutils.make_grid(GAN_inv_imgs, nrow=config.tuneG['num_inv_points'], padding=2,
                                            normalize=True)
                    # 转换为 PIL 图像并保存
                    img = Image.fromarray(
                        grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

                    resize_scale = 0.5
                    if resize_scale != 1:
                        new_size = (int(img.width * resize_scale), int(img.height * resize_scale))
                        img = img.resize(new_size, Image.ANTIALIAS)
                    img.save(
                        os.path.join(f"results/{current_time}/{target_id:03d}",
                                     f'tune_images_r{round_num + 1}_{i + 1:03d}.png'))

                if i == 0 or (i + 1) % 20 == 0:
                    print(
                        f"Iteration {i + 1}: Loss = {loss.item():.2f}, "
                        f"L2 Loss = {l2_loss_val.item():.2f}, "
                        f"Dist Loss = {loss_dist.item():.2f}, "
                        f"ball_holder_loss_val={ball_holder_loss_val.item():.4f}"
                    )

            inv_dist_list = [['target', 'loss', 'l2_loss_val', 'loss_dist', 'hold_loss'],
                             [target_id, loss.item(), l2_loss_val.item(), loss_dist.item(),
                              ball_holder_loss_val.item()]]

            if config.logging:
                _ = write_list(
                    f'results/{current_time}/tune_loss_r{round_num + 1}',
                    inv_dist_list
                )

            step_3_2_time = time.time() - start_time
            time_cost_list = [['target', 'mi', 'selection', 'step_3_1', 'step_3_2'],
                              [target_id, time_list[0], time_list[1], step_3_1_time, step_3_2_time]]

            if config.logging:
                _ = write_list(
                    f'results/{current_time}/time_cost_r{round_num + 1}',
                    time_cost_list
                )

            del pp_data, w, w_init, w_pivots
            torch.cuda.empty_cache()

            toogle_grad(G, False)

    all_final_w = [torch.cat(final_w, dim=0) for final_w in all_final_w]
    all_final_targets = [torch.cat(target_id, dim=0) for target_id in all_final_targets]
    rtpt = None

    for i, (final_w, final_targets) in enumerate(zip(all_final_w, all_final_targets)):

        ####################################
        #    FID Score and GAN Metrics     #
        ####################################

        fid_score = None
        precision, recall = None, None
        density, coverage = None, None
        try:
            # set transformations
            crop_size = config.attack_center_crop
            target_transform = T.Compose([
                T.ToTensor(),
                T.Resize((299, 299), antialias=True),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            # create datasets
            attack_dataset = TensorDataset(final_w, final_targets)
            attack_dataset.targets = final_targets
            training_dataset = create_target_dataset(target_dataset,
                                                     target_transform)
            training_dataset = ClassSubset(
                training_dataset,
                target_classes=torch.unique(final_targets).cpu().tolist())

            # compute precision, recall, density, coverage
            prdc = PRCD(training_dataset,
                        attack_dataset,
                        device=device,
                        crop_size=crop_size,
                        generator=G.synthesis,
                        batch_size=batch_size * 3,
                        dims=2048,
                        num_workers=8,
                        gpu_devices=gpu_devices)
            precision, recall, density, coverage = prdc.compute_metric(
                num_classes=config.num_classes, k=3, rtpt=rtpt)

            print(
                f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
            )
            if config.logging:
                prdc_list = [['precision', 'recall', 'density', 'coverage'],
                             [precision, recall, density, coverage]]
                _ = write_list(
                    f'results/{current_time}/prdc_list_r{i + 1}', prdc_list)
            # compute FID score
            fid_evaluation = FID_Score(training_dataset,
                                       attack_dataset,
                                       device=device,
                                       crop_size=crop_size,
                                       generator=G.synthesis,
                                       batch_size=batch_size * 3,
                                       dims=2048,
                                       num_workers=8,
                                       gpu_devices=gpu_devices)
            fid_score = fid_evaluation.compute_fid(rtpt)
            print(
                f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
            )
            if config.logging:
                prdc_list = [['fid', 'precision', 'recall', 'density', 'coverage'],
                             [fid_score, precision, recall, density, coverage]]
                _ = write_list(
                    f'results/{current_time}/prdc_list_r{i + 1}', prdc_list)

        except Exception:
            print(traceback.format_exc())
