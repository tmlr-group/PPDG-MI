import os, models.facenet as facenet, sys
import json, time, torch
import shutil

from models import classify
from models.classify import *
from models.discri import *
from models.generator import *
import torch.nn as nn
import torch.nn.functional as F
import losses as L
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
import dataloader
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from torch.autograd import grad

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()


def init_dataloader(args, file_path, batch_size=64, mode="gan", iterator=False):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    data_set = dataloader.ImageFolder(args, file_path, mode)

    if iterator:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_flag,
                                                  drop_last=True,
                                                  num_workers=0,
                                                  pin_memory=True).__iter__()
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_flag,
                                                  drop_last=True,
                                                  num_workers=2,
                                                  pin_memory=True)
        interval = time.time() - tf
        print('Initializing data loader took %ds' % interval)

    return data_set, data_loader


def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)

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


def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)


def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]

    img = img.cuda()
    return img


def get_model(attack_name, classes):
    if attack_name.startswith("VGG16"):
        T = classify.VGG16(classes)
    elif attack_name.startswith("IR50"):
        T = classify.IR50(classes)
    elif attack_name.startswith("VGG16_HSIC"):
        T = classify.IR50(classes)
    elif attack_name.startswith("IR152"):
        T = classify.IR152(classes)
    elif attack_name.startswith("FaceNet64"):
        T = facenet.FaceNet64(classes)
    else:
        print("Model doesn't exist")
        exit()

    T = torch.nn.DataParallel(T).cuda()
    return T


def get_augmodel(model_name, nclass, path_T=None, dataset='celeba'):
    if model_name == "VGG16":
        model = VGG16(nclass)
    elif model_name == "FaceNet":
        model = FaceNet(nclass)
    elif model_name == "FaceNet64":
        model = FaceNet64(nclass)
    elif model_name == "IR152":
        model = IR152(nclass)
    elif model_name == "efficientnet_b0":
        model = classify.EfficientNet_b0(nclass)
    elif model_name == "efficientnet_b1":
        model = classify.EfficientNet_b1(nclass)
    elif model_name == "efficientnet_b2":
        model = classify.EfficientNet_b2(nclass)

    model = torch.nn.DataParallel(model).cuda()
    if path_T is not None:
        ckp_T = torch.load(path_T)
        t=model.load_state_dict(ckp_T['state_dict'], strict=True)
    return model

from collections import OrderedDict


def fix_state_dict_keys(state_dict):
    """
    Adjusts the keys in the state dictionary. If keys don't start with 'module.',
    it will add 'module.' to the start to match with DataParallel module keys.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # This prefix adjustment assumes the model is being wrapped with DataParallel
        new_key = k if k.startswith('module.') else 'module.' + k
        new_state_dict[new_key] = v
    return new_state_dict


def get_augmodel(model_name, nclass, path_T=None, dataset='celeba'):
    # Mapping of model names to model constructors
    model_classes = {
        "VGG16": VGG16,
        "FaceNet": FaceNet,
        "FaceNet64": FaceNet64,
        "IR152": IR152,
        "efficientnet_b0": classify.EfficientNet_b0,
        "efficientnet_b1": classify.EfficientNet_b1,
        "efficientnet_b2": classify.EfficientNet_b2,
    }

    # Instantiate the model
    if model_name in model_classes:
        model = model_classes[model_name](nclass)
    else:
        raise ValueError(f"Model name {model_name} is not supported.")

    # Wrap the model with DataParallel and move to GPU
    model = torch.nn.DataParallel(model).cuda()

    # Load checkpoint if provided
    if path_T is not None:
        ckp_T = torch.load(path_T)
        try:
            # Attempt to fix state dict keys
            fixed_state_dict = fix_state_dict_keys(ckp_T['state_dict'])
            # Load the state dict into the model
            model.load_state_dict(fixed_state_dict, strict=True)
        except RuntimeError as e:
            # Handle errors in state dict loading
            print(f"Failed to load state dictionary: {e}")
            # Optionally, you might want to raise the error or handle it differently
            raise e

    return model


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


# define "soft" cross-entropy with pytorch tensor operations
def softXEnt(input, target):
    targetprobs = nn.functional.softmax(target, dim=1)
    logprobs = nn.functional.log_softmax(input, dim=1)
    return -(targetprobs * logprobs).sum() / input.shape[0]


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def find_criterion(used_loss):
    criterion = None
    if used_loss == 'logit':
        criterion = L.nll_loss().to(device)
        print('criterion:{}'.format(used_loss))
    elif used_loss == 'poincare':
        criterion = L.poincare_loss().to(device)
        print('criterion', criterion)
    elif used_loss == 'margin':
        criterion = L.max_margin_loss().to(device)
        print('criterion', criterion)
    elif used_loss == 'cel':
        criterion = nn.CrossEntropyLoss().to(device)
        print('criterion', criterion)
    else:
        print('criterion:{}'.format(used_loss))
    return criterion

def gradient_penalty(x, y, DG):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def custom_collate(batch):
    images = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            images.append(item.to(device))
        elif isinstance(item, tuple):
            img = item[0].to(device)
            label = item[1].to(device) if isinstance(item[1], torch.Tensor) else item[1]
            images.append((img, label))
        else:
            raise TypeError("Unsupported type in batch")

    return torch.utils.data.dataloader.default_collate(images)


def get_GAN(dataset, gan_type, gan_model_dir, n_classes, z_dim, target_model):
    G = Generator(z_dim)
    if gan_type == True:
        D = MinibatchDiscriminator(n_classes=n_classes)
    else:
        D = DGWGAN(3)

    if gan_type == True:
        path = os.path.join(os.path.join(gan_model_dir, dataset), target_model)
        path_G = os.path.join(path, "improved_{}_G.tar".format(dataset))
        path_D = os.path.join(path, "improved_{}_D.tar".format(dataset))
    else:
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


def get_attack_model(args, args_json, eval_mode=False):
    now = datetime.now()  # current date and time

    if not eval_mode:
        log_file = "invertion_logs_{}_{}.txt".format(args.loss, now.strftime("%m_%d_%Y_%H_%M_%S"))
        Tee(os.path.join(args.log_path, log_file), 'w')

    n_classes = args_json['dataset']['n_classes']

    model_types_ = args_json['train']['model_types'].split(',')
    checkpoints = args_json['train']['cls_ckpts'].split(',')

    G, D = get_GAN(args_json['dataset']['name'], gan_type=args.improved_flag,
                   gan_model_dir=args_json['train']['gan_model_dir'],
                   n_classes=n_classes, z_dim=100, target_model=model_types_[0])

    dataset = args_json['dataset']['name']
    cid = args.classid.split(',')
    # target and student classifiers

    for i in range(len(cid)):
        id_ = int(cid[i])
        model_types_[id_] = model_types_[id_].strip()
        checkpoints[id_] = checkpoints[id_].strip()
        print('Load classifier {} at {}'.format(model_types_[id_], checkpoints[id_]))
        model = get_augmodel(model_types_[id_], n_classes, checkpoints[id_], dataset)
        model = model.to(device)
        model = model.eval()
        if i == 0:
            targetnets = [model]
        else:
            targetnets.append(model)

        # p_reg
        if args.loss == 'logit_loss':
            if model_types_[id_] == "IR152" or model_types_[id_] == "VGG16" or model_types_[id_] == "FaceNet64" or \
                    model_types_[id_] == "VGG16_HSIC":
                # target model
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_p_reg.pt'.format(dataset, model_types_[
                    id_]))  # './p_reg/{}_{}_p_reg.pt'.format(dataset,model_types_[id_])
            else:
                # aug model
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"],
                                     '{}_{}_{}_p_reg.pt'.format(dataset, model_types_[0], model_types_[
                                         id_]))  # './p_reg/{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])
            # print('p_reg',p_reg)
            if not os.path.exists(p_reg):
                _, dataloader_gan = init_dataloader(args_json, args_json['dataset']['gan_file_path'], 50, mode="gan")
                from attack import get_act_reg
                fea_mean_, fea_logvar_ = get_act_reg(dataloader_gan, model, device)
                torch.save({'fea_mean': fea_mean_, 'fea_logvar': fea_logvar_}, p_reg)
            else:
                fea_reg = torch.load(p_reg)
                fea_mean_ = fea_reg['fea_mean']
                fea_logvar_ = fea_reg['fea_logvar']
            if i == 0:
                fea_mean = [fea_mean_.to(device)]
                fea_logvar = [fea_logvar_.to(device)]
            else:
                fea_mean.append(fea_mean_)
                fea_logvar.append(fea_logvar_)
            # print('fea_logvar_',i,fea_logvar_.shape,fea_mean_.shape)

        else:
            fea_mean, fea_logvar = 0, 0

    # evaluation classifier
    E = get_augmodel(args_json['train']['eval_model'], n_classes, args_json['train']['eval_dir'])
    E.eval()
    G.eval()
    D.eval()

    return targetnets, E, G, D, n_classes, fea_mean, fea_logvar


# returns the predicted label on the evaluator model (which requires low2high to increase the input resolution)
def decision_Evaluator(imgs, model, score=False, target=None, criterion=None):
    return decision(imgs, model, score=score, target=target, criterion=criterion, islow2high=True)


# returns the predicted label on the evaluator model
def decision(imgs, model, score=False, target=None, criterion=None, islow2high=False):
    if islow2high:
        imgs = low2high(imgs)

    with torch.no_grad():
        T_out = model(imgs)[-1]
        val_iden = torch.argmax(T_out, dim=1).view(-1)

    if score:
        T_out = T_out.cuda()

        single_target_index = target[0].long()
        target = torch.tensor([single_target_index]).cuda()

        return val_iden, criterion(T_out, target)
    else:
        return val_iden

    # returns whether a batch of images belong to a target class or not


# if they belong, 1 is returned, else -1 is returned
def is_target_class(idens, target, model, score=False, criterion=None):
    if score:
        target_class_tensor = torch.tensor([target]).cuda()
        val_iden, score = decision(idens, model, score, target_class_tensor, criterion=criterion)
    else:
        val_iden = decision(idens, model)
    val_iden[val_iden != target] = -1
    val_iden[val_iden == target] = 1
    return val_iden


def gen_initial_points_targeted(batch_size, G, target_model, min_clip, max_clip, z_dim, num_candidates, target_id):
    print(f'Generating initial points for class: {target_id}')
    current_iter = 0
    initial_points = []
    with torch.no_grad():
        while True:
            z = torch.randn(batch_size, z_dim).cuda().float().clamp(min=min_clip, max=max_clip)
            first_img = G(z)

            # our target class is now the current class of the generated image
            target_classes = decision(first_img, target_model)

            for i in range(target_classes.shape[0]):
                current_label = target_classes[i].item()
                if current_label != target_id:
                    continue

                initial_points.append(z[i])

                if len(initial_points) % 100 == 0:
                    print("Sample {}-th latent vectors at iteration {}".format(len(initial_points), current_iter))

                if len(initial_points) == num_candidates:
                    break

            current_iter += 1
            if len(initial_points) == num_candidates:
                break

    initial_points = torch.stack(initial_points)

    return initial_points


def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=100):
    score = torch.zeros_like(targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            output = target_model(imgs_transformed)
            if type(output) is tuple:
                output = target_model(imgs_transformed)[1]
            prediction_vector = output.softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score


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

    transformation = transforms.Compose([
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5)
    ])

    for step, target in enumerate(target_values):
        mask = torch.where(targets == target, True, False).cpu()
        z_masked = z[mask]
        candidates = G(z_masked).cpu()
        targets_masked = targets[mask].cpu()
        scores = []
        dataset = TensorDataset(candidates, targets_masked)
        for imgs, t in DataLoader(dataset, batch_size=batch_size):
            imgs, t = imgs.to(device), t.to(device)

            scores.append(
                scores_by_transform(imgs, t, target_model, transformation))
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
    final_targets = torch.cat(final_targets, dim=0)
    final_z = torch.cat(final_z, dim=0)
    return final_z, final_targets
