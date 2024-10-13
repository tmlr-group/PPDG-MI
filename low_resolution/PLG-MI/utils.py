import json
import numpy
import os
import shutil
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.generators.resnet64 import ResNetGenerator
from models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from models.classifiers import VGG16, IR152, FaceNet, FaceNet64

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

class Dict2Args(object):
    """Dict-argparse object converter."""

    def __init__(self, dict_args):
        for key, value in dict_args.items():
            setattr(self, key, value)


def generate_images(gen, device, batch_size=64, dim_z=128, distribution=None,
                    num_classes=None, class_id=None):
    """Generate images.

    Priority: num_classes > class_id.

    Args:
        gen (nn.Module): generator.
        device (torch.device)
        batch_size (int)
        dim_z (int)
        distribution (str)
        num_classes (int, optional)
        class_id (int, optional)

    Returns:
        torch.tensor

    """

    z = sample_z(batch_size, dim_z, device, distribution)
    if num_classes is None and class_id is None:
        y = None
    elif num_classes is not None:
        y = sample_pseudo_labels(num_classes, batch_size, device)
    elif class_id is not None:
        y = torch.tensor([class_id] * batch_size, dtype=torch.long).to(device)
    else:
        y = None
    with torch.no_grad():
        fake = gen(z, y)

    return fake


def sample_z(batch_size, dim_z, device, distribution=None):
    """Sample random noises.

    Args:
        batch_size (int)
        dim_z (int)
        device (torch.device)
        distribution (str, optional): default is normal

    Returns:
        torch.FloatTensor or torch.cuda.FloatTensor

    """

    if distribution is None:
        distribution = 'normal'
    if distribution == 'normal':
        return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
    else:
        return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).uniform_()


def sample_pseudo_labels(num_classes, batch_size, device):
    """Sample pseudo-labels.

    Args:
        num_classes (int): number of classes in the dataset.
        batch_size (int): size of mini-batch.
        device (torch.Device): For compatibility.

    Returns:
        ~torch.LongTensor or torch.cuda.LongTensor.

    """

    pseudo_labels = torch.from_numpy(
        numpy.random.randint(low=0, high=num_classes, size=(batch_size))
    )
    pseudo_labels = pseudo_labels.type(torch.long).to(device)
    return pseudo_labels


def save_images(n_iter, count, root, train_image_root, fake, real):
    """Save images (torch.tensor).

    Args:
        root (str)
        train_image_root (root)
        fake (torch.tensor)
        real (torch.tensor)

    """

    fake_path = os.path.join(
        train_image_root,
        'fake_{}_iter_{:07d}.png'.format(count, n_iter)
    )
    real_path = os.path.join(
        train_image_root,
        'real_{}_iter_{:07d}.png'.format(count, n_iter)
    )
    torchvision.utils.save_image(
        fake, fake_path, nrow=4, normalize=True, scale_each=True
    )
    shutil.copy(fake_path, os.path.join(root, 'fake_latest.png'))
    torchvision.utils.save_image(
        real, real_path, nrow=4, normalize=True, scale_each=True
    )
    shutil.copy(real_path, os.path.join(root, 'real_latest.png'))


def save_checkpoints(args, n_iter, count, gen, opt_gen, dis, opt_dis):
    """Save checkpoints.

    Args:
        args (argparse object)
        n_iter (int)
        gen (nn.Module)
        opt_gen (torch.optim)
        dis (nn.Module)
        opt_dis (torch.optim)

    """

    count = n_iter // args.checkpoint_interval
    gen_dst = os.path.join(
        args.results_root,
        'gen_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
    )
    torch.save({
        'model': gen.state_dict(), 'opt': opt_gen.state_dict(),
    }, gen_dst)
    shutil.copy(gen_dst, os.path.join(args.results_root, 'gen_latest.pth.tar'))
    dis_dst = os.path.join(
        args.results_root,
        'dis_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
    )
    torch.save({
        'model': dis.state_dict(), 'opt': opt_dis.state_dict(),
    }, dis_dst)
    shutil.copy(dis_dst, os.path.join(args.results_root, 'dis_latest.pth.tar'))


def resume_from_args(args_path, gen_ckpt_path, dis_ckpt_path):
    """Load generator & discriminator with their optimizers from args.json.

    Args:
        args_path (str): Path to args.json
        gen_ckpt_path (str): Path to generator checkpoint or relative path
                             from args['results_root']
        dis_ckpt_path (str): Path to discriminator checkpoint or relative path
                             from args['results_root']

    Returns:
        gen, opt_dis
        dis, opt_dis

    """

    from models.generators import resnet64
    from models.discriminators import snresnet64

    with open(args_path) as f:
        args = json.load(f)
    conditional = args['cGAN']
    num_classes = args['num_classes'] if conditional else 0
    # Initialize generator
    gen = resnet64.ResNetGenerator(
        args['gen_num_features'], args['gen_dim_z'], args['gen_bottom_width'],
        num_classes=num_classes, distribution=args['gen_distribution']
    )
    opt_gen = torch.optim.Adam(
        gen.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    # Initialize discriminator
    if args['dis_arch_concat']:
        dis = snresnet64.SNResNetConcatDiscriminator(
            args['dis_num_features'], num_classes, dim_emb=args['dis_emb']
        )
    else:
        dis = snresnet64.SNResNetProjectionDiscriminator(
            args['dis_num_features'], num_classes
        )
    opt_dis = torch.optim.Adam(
        dis.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    if not os.path.exists(gen_ckpt_path):
        gen_ckpt_path = os.path.join(args['results_root'], gen_ckpt_path)
    gen, opt_gen = load_model_optim(gen_ckpt_path, gen, opt_gen)
    if not os.path.exists(dis_ckpt_path):
        dis_ckpt_path = os.path.join(args['results_root'], dis_ckpt_path)
    dis, opt_dis = load_model_optim(dis_ckpt_path, dis, opt_dis)
    return Dict2Args(args), gen, opt_gen, dis, opt_dis


def load_model_optim(checkpoint_path, model=None, optim=None):
    """Load trained weight.

    Args:
        checkpoint_path (str)
        model (nn.Module)
        optim (torch.optim)

    Returns:
        model
        optim

    """

    ckpt = torch.load(checkpoint_path)
    if model is not None:
        model.load_state_dict(ckpt['model'])
    if optim is not None:
        optim.load_state_dict(ckpt['opt'])
    return model, optim

def load_optim(checkpoint_path, optim):
    """Load optimizer from checkpoint.

    Args:
        checkpoint_path (str)
        optim (torch.optim)

    Returns:
        optim

    """

    return load_model_optim(checkpoint_path, None, optim)[1]


def get_GAN(dataset, gan_model_dir, z_dim, n_classes, device, gen_num_features, gen_bottom_width, gen_distribution, dis_num_features):
    G = ResNetGenerator(
        gen_num_features, z_dim, gen_bottom_width,
        num_classes=n_classes, distribution=gen_distribution
    )

    D = SNResNetProjectionDiscriminator(dis_num_features, 1000, F.relu).to(device)

    path = os.path.join(gan_model_dir, dataset)
    path_G = os.path.join(path, "{}_G.tar".format(dataset))
    path_D = os.path.join(path, "{}_D.tar".format(dataset))

    G = torch.nn.DataParallel(G).to(device)
    D = torch.nn.DataParallel(D).to(device)
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=True)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=True)

    return G, D


def load_model(model_name, path_T, num_classes=1000):
    if model_name.startswith("VGG16"):
        T = VGG16(num_classes)
    elif model_name.startswith('IR152'):
        T = IR152(num_classes)
    elif model_name == "FaceNet64":
        T = FaceNet64(num_classes)
    elif model_name == "FaceNet64":
        T = FaceNet(num_classes)
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=True)
    return T

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

    final_targets = torch.cat(final_targets, dim=0).to(device)
    final_z = torch.cat(final_z, dim=0).to(device)
    return final_z, final_targets

