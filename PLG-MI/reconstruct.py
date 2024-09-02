import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
import kornia
from kornia import augmentation
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import math
import torchvision
from evaluation import evaluate_results, write_precision_list

import losses as L
import utils
from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
from models.generators.resnet64 import ResNetGenerator
from models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from dataset import FaceDataset, InfiniteSamplerWrapper, sample_from_data, sample_from_gen
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=1):
    score = torch.zeros_like(targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            output = target_model(imgs_transformed)[1]
            prediction_vector = output.softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score


def perform_final_selection(z,
                            G,
                            targets,
                            target_model,
                            samples_per_target,
                            batch_size,
                            device,
                            rtpt=None):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_z = []
    target_model.eval()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomRotation(5),
    ])

    for step, target in enumerate(target_values):
        mask = torch.where(targets == target, True, False).cpu()
        z_masked = z[mask]
        targets_masked = targets[mask]

        candidates_list = []
        for i in range(0, len(z_masked), batch_size):
            z_batch = z_masked[i:i + batch_size]
            targets_batch = targets_masked[i:i + batch_size]
            candidates_batch = G(z_batch, targets_batch).cpu()
            candidates_list.append(candidates_batch)

        candidates = torch.cat(candidates_list, dim=0)

        scores = []
        dataset = TensorDataset(candidates, targets_masked)
        for imgs, t in DataLoader(dataset, batch_size=batch_size):
            imgs, t = imgs.to(device), t.to(device)

            scores.append(
                scores_by_transform(imgs, t, target_model, transforms))

        scores = torch.cat(scores, dim=0).cpu()
        indices = torch.sort(scores, descending=True).indices
        selected_indices = indices[:samples_per_target]
        final_targets.append(targets_masked[selected_indices].cpu())
        final_z.append(z_masked[selected_indices].cpu())

        if rtpt:
            rtpt.step(
                subtitle=f'Sample Selection step {step} of {len(target_values)}'
            )
    print(scores[selected_indices])
    final_targets = torch.cat(final_targets, dim=0).to(device)
    final_z = torch.cat(final_z, dim=0).to(device)
    return final_z, final_targets


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


def inversion(args, G, D, T, E, batch_size, targets, lr=2e-2, MI_iter_times=600):
    G.eval()
    D.eval()
    E.eval()
    T.eval()

    aug_list = augmentation.container.ImageSequential(
        augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        augmentation.ColorJitter(brightness=0.2, contrast=0.2),
        augmentation.RandomHorizontalFlip(),
        augmentation.RandomRotation(5),
    )

    z_opt = []
    # Prepare batches for attack
    for i in range(math.ceil(len(targets) / batch_size)):
        z = utils.sample_z(
            batch_size, args.gen_dim_z, device, args.gen_distribution
        )
        iden = targets[i * batch_size:(i + 1) * batch_size]

        target_classes_set = set(iden.tolist())

        print(
            f'Optimizing batch {i + 1} of {math.ceil(len(targets) / batch_size)} target classes {target_classes_set}.')

        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(MI_iter_times):
            fake = G(z, iden)

            out1 = T(aug_list(fake))[-1]
            out2 = T(aug_list(fake))[-1]

            if z.grad is not None:
                z.grad.data.zero_()

            if args.inv_loss_type == 'ce':
                inv_loss = L.cross_entropy_loss(out1, iden) + L.cross_entropy_loss(out2, iden)
            elif args.inv_loss_type == 'margin':
                inv_loss = L.max_margin_loss(out1, iden) + L.max_margin_loss(out2, iden)
            elif args.inv_loss_type == 'poincare':
                inv_loss = L.poincare_loss(out1, iden) + L.poincare_loss(out2, iden)

            optimizer.zero_grad()
            inv_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    fake_img = G(z.detach(), iden)

                    eval_prob = E(augmentation.Resize((112, 112))(fake_img))[-1]
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / batch_size
                    print("Iteration:{}\tInv Loss:{:.2f}\tAttack Acc:{:.3f}".format(i + 1,
                                                                                    inv_loss,
                                                                                    acc))

                    outputs = T(fake)
                    confidence_vector = outputs[-1].softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, iden.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                    min_conf = confidences.min().detach().cpu().item()  # Get minimum confidence
                    max_conf = confidences.max().detach().cpu().item()  # Get maximum confidence
                    print(f'mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})\n')

        z_opt.append(z.detach())

    return torch.concat(z_opt, dim=0)


def PLG_attack(args, G, D, T, E, targets_single_id, iterations=600, round_num=1):
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
        opt_z = inversion(args, G, D, T, E, batch_size, targets_single_id, lr=args.lr, MI_iter_times=iterations)
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


def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = os.path.join(args.results_root,
                        args.public_data_name, args.model)
    os.makedirs(root, exist_ok=True)

    train_image_root = os.path.join(root, "preview", "train")
    eval_image_root = os.path.join(root, "preview", "eval")
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(eval_image_root, exist_ok=True)

    args.results_root = root
    args.train_image_root = train_image_root
    args.eval_image_root = eval_image_root

    return args


def tune_cgan(args, gen, dis, target_model, final_z, final_y, gan_max_iteration=1000):
    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img

    def toogle_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    # dataset crop setting
    if args.public_data_name == 'celeba':
        re_size = 64
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    elif args.public_data_name == 'ffhq':
        crop_size = 88
        offset_height = (128 - crop_size) // 2
        offset_width = (128 - crop_size) // 2
        re_size = 64
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    elif args.public_data_name == 'facescrub':
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
        _noise_adder
    ])
    public_dataset = FaceDataset(args=args, root=args.public_data_root, transform=my_transform,
                                 data_name=args.public_data_name)
    print(f"public dataset len {len(public_dataset)}")

    # pseudo-private dataset
    with torch.no_grad():
        pseudo_private_data = gen(final_z, final_y).cpu()

    final_y = [int(y.item()) for y in final_y]
    # Combine pseudo_private_data and final_y into a dataset
    pseudo_private_dataset = [(data, label) for data, label in zip(pseudo_private_data, final_y)]
    print(f"pesudo-private dataset len {len(pseudo_private_dataset)}")

    combined_dataset = ConcatDataset([public_dataset, pseudo_private_dataset])
    print(f"combined dataset len {len(combined_dataset)}\n")
    train_loader = iter(torch.utils.data.DataLoader(
        combined_dataset, args.batch_size,
        sampler=InfiniteSamplerWrapper(combined_dataset),
    )
    )

    def check_grad_status(model):
        for name, param in model.named_parameters():
            print(f"Parameter {name}: requires_grad = {param.requires_grad}")

    # load optimizer
    toogle_grad(gen, True)
    toogle_grad(dis, True)

    opt_gen = torch.optim.Adam(gen.parameters(), args.tune_cGAN_lr, (args.beta1, args.beta2))
    opt_dis = torch.optim.Adam(dis.parameters(), args.tune_cGAN_lr, (args.beta1, args.beta2))
    # get adversarial loss
    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)

    # data augmentation module in stage-1 for the generated images
    aug_list = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    )

    args = prepare_results_dir(args)

    # Training loop
    for n_iter in range(1, gan_max_iteration + 1):
        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_inv_loss = 0.
        cumulative_loss_dis = .0

        cumulative_target_acc = .0
        target_correct = 0
        count = 0

        for i in range(args.n_dis):  # args.ndis=5, Gen update 1 time, Dis update ndis times.
            if i == 0:
                fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen)
                dis_fake = dis(fake, pseudo_y)
                # random transformation on the generated images
                fake_aug = aug_list(fake)
                # calc the L_inv
                if args.inv_loss_type == 'ce':
                    inv_loss = L.cross_entropy_loss(target_model(fake_aug)[-1], pseudo_y)
                elif args.inv_loss_type == 'margin':
                    inv_loss = L.max_margin_loss(target_model(fake_aug)[-1], pseudo_y)
                elif args.inv_loss_type == 'poincare':
                    inv_loss = L.poincare_loss(target_model(fake_aug)[-1], pseudo_y)
                # not used
                if args.relativistic_loss:
                    real, y = sample_from_data(args, device, train_loader)
                    dis_real = dis(real, y)
                else:
                    dis_real = None
                # calc the loss of G
                loss_gen = gen_criterion(dis_fake, dis_real)
                loss_all = loss_gen + inv_loss * args.alpha
                # update the G
                gen.zero_grad()
                loss_all.backward()
                opt_gen.step()
                _l_g += loss_gen.item()

                cumulative_inv_loss += inv_loss.item()

            # generate fake images
            fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen)
            # sample the real images
            real, y = sample_from_data(args, device, train_loader)
            # calc the loss of D
            dis_fake, dis_real = dis(fake, pseudo_y), dis(real, y)
            loss_dis = dis_criterion(dis_fake, dis_real)
            # update D
            dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            cumulative_loss_dis += loss_dis.item()

            with torch.no_grad():
                count += fake.shape[0]
                T_logits = target_model(fake)[-1]
                T_preds = T_logits.max(1, keepdim=True)[1]
                target_correct += T_preds.eq(pseudo_y.view_as(T_preds)).sum().item()
                cumulative_target_acc += round(target_correct / count, 4)

        # ==================== End of 1 iteration. ====================
        args.log_interval = 100
        if n_iter % args.log_interval == 0:
            print(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
                    n_iter, gan_max_iteration, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                    cumulative_target_acc)
            )
            # Save previews
            utils.save_images(
                n_iter, n_iter // args.log_interval, args.results_root,
                args.train_image_root, fake, real
            )

    toogle_grad(gen, False)
    toogle_grad(dis, False)

    return gen, dis


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
    path_E = 'checkpoints/target_model/target_ckp/FaceNet_95.88.tar'
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

            final_z, final_targets, time_list = PLG_attack(args, G, D, T, E, targets_single_id,
                                                           iterations=iterations,
                                                           round_num=cur_round)

            if cur_round == num_round - 1:
                break

            print("GAN Fine-tuning")

            start_time = time.time()

            # G, D = tune_cgan(args, G, D, T, final_z[:50], final_targets[:50], gan_max_iteration=args.tune_iter_times)
            G, D = tune_cgan(args, G, D, T, final_z[:samples_per_target], final_targets[:samples_per_target],
                             gan_max_iteration=args.tune_iter_times)

            tune_time = time.time() - start_time

            time_cost_list = [['target', 'mi', 'selection', 'tune_time'],
                              [target_id, time_list[0], time_list[1], tune_time]]

            _ = write_precision_list(
                f'{prefix}/{current_time}/time_cost_r{cur_round + 1}',
                time_cost_list
            )