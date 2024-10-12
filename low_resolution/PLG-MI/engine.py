import torch, torchvision
import losses as L
import kornia
import utils
from utils import prepare_results_dir
from dataset import FaceDataset, InfiniteSamplerWrapper, sample_from_data, sample_from_gen
from torch.utils.data import ConcatDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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