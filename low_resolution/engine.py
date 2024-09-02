import torch, utils
import torch.nn as nn
from copy import deepcopy
import losses as L
from utils import *
from models.discri import MinibatchDiscriminator, DGWGAN
from models.generator import Generator
from models.classify import *
from torch.utils.data import ConcatDataset, DataLoader
from dataset import FaceDataset, InfiniteSamplerWrapper, sample_from_data, sample_from_gen

torch.autograd.set_detect_anomaly(True)


def test(model, criterion=None, dataloader=None, device='cuda'):
    tf = time.time()
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (img, iden) in enumerate(dataloader):
            img, iden = img.to(device), iden.to(device)

            bs = img.size(0)
            iden = iden.view(-1)
            _, out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()

            _, top5 = torch.topk(out_prob, 5, dim=1)
            for ind, top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1

            cnt += bs

    return ACC * 100.0 / cnt, correct_top5 * 100.0 / cnt


def train_reg(args, model, criterion, optimizer, trainloader, testloader, n_epochs, device='cuda'):
    best_ACC = (0.0, 0.0)

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC


def train_vib(args, model, criterion, optimizer, trainloader, testloader, n_epochs, device='cuda'):
    best_ACC = (0.0, 0.0)

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0

        for i, (img, iden) in enumerate(trainloader):
            img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            ___, out_prob, mu, std = model(img, "train")
            cross_loss = criterion(out_prob, one_hot)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_loss, test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC


def get_T(model_name_T, cfg):
    if model_name_T.startswith("VGG16"):
        T = VGG16(cfg['dataset']["n_classes"])
    elif model_name_T.startswith('IR152'):
        T = IR152(cfg['dataset']["n_classes"])
    elif model_name_T == "FaceNet64":
        T = FaceNet64(cfg['dataset']["n_classes"])
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(cfg[cfg['dataset']['model_name']]['cls_ckpts'])
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    return T


def train_specific_gan(cfg):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name_T = cfg['dataset']['model_name']
    batch_size = cfg[model_name_T]['batch_size']
    z_dim = cfg[model_name_T]['z_dim']
    n_critic = cfg[model_name_T]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, model_name_T))
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # Log file
    log_path = os.path.join(save_model_dir, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "improvedGAN_{}.txt".format(model_name_T)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    # writer = SummaryWriter(log_path)

    # Load target model
    T = get_T(model_name_T=model_name_T, cfg=cfg)

    # Dataset
    dataset, dataloader = utils.init_dataloader(cfg, file_path, cfg[model_name_T]['batch_size'], mode="gan")

    # Start Training
    print("Training GAN for %s" % model_name_T)
    utils.print_params(cfg["dataset"], cfg[model_name_T])

    G = Generator(cfg[model_name_T]['z_dim'])
    DG = MinibatchDiscriminator()

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))

    entropy = HLoss()

    step = 0
    for epoch in range(cfg[model_name_T]['epochs']):
        start = time.time()
        _, unlabel_loader1 = init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)

        for i, imgs in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = unlabel_loader1.next()
            x_unlabel2 = unlabel_loader2.next()

            toogle_grad(G, False)
            toogle_grad(DG, True)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake = DG(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel))) - torch.mean(
                log_sum_exp(output_unlabel)) + torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            acc = torch.mean((output_label.max(1)[1] == y).float())

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            writer.add_scalar('loss_label_batch', loss_lab, current_iter)
            writer.add_scalar('loss_unlabel_batch', loss_unlab, current_iter)
            writer.add_scalar('DG_loss_batch', dg_loss, current_iter)
            writer.add_scalar('Acc_batch', acc, current_iter)

            # train G
            if step % n_critic == 0:
                toogle_grad(DG, False)
                toogle_grad(G, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim=0)
                mom_unlabel = torch.mean(mom_unlabel, dim=0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start

        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        torch.save({'state_dict': G.state_dict()},
                   os.path.join(save_model_dir, "improved_{}_G.tar".format(dataset_name)))
        torch.save({'state_dict': DG.state_dict()},
                   os.path.join(save_model_dir, "improved_{}_D.tar".format(dataset_name)))

        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(),
                               os.path.join(save_img_dir, "improved_celeba_img_{}.png".format(epoch)), nrow=8)


def train_general_gan(cfg):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    epochs = cfg[model_name]['epochs']
    n_critic = cfg[model_name]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, 'general_GAN'))
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # Log file
    log_path = os.path.join(save_model_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "GAN_{}.txt".format(dataset_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    # writer = SummaryWriter(log_path)

    # Dataset
    dataset, dataloader = init_dataloader(cfg, file_path, batch_size, mode="gan")

    # Start Training
    print("Training general GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    G = Generator(z_dim)
    DG = DGWGAN(3)

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            toogle_grad(G, False)
            toogle_grad(DG, True)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data, DG=DG)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                toogle_grad(DG, False)
                toogle_grad(G, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))
        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch)),
                               nrow=8)

        torch.save({'state_dict': G.state_dict()}, os.path.join(save_model_dir, "celeba_G.tar"))
        torch.save({'state_dict': DG.state_dict()}, os.path.join(save_model_dir, "celeba_D.tar"))


def tune_cgan(args, cfg, generator, discriminator, target_model, final_z, final_y, epoch=1000):
    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img

    criterion = find_criterion(args.loss)
    model_name = cfg['dataset']['model_name']
    n_critic = cfg[model_name]['n_critic']

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
        pseudo_private_data = generator(final_z, final_y).cpu()

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

    # load optimizer
    toogle_grad(generator, True)
    toogle_grad(discriminator, True)

    opt_gen = torch.optim.Adam(generator.parameters(), args.tune_cGAN_lr, (args.beta1, args.beta2))
    opt_dis = torch.optim.Adam(discriminator.parameters(), args.tune_cGAN_lr, (args.beta1, args.beta2))
    # get adversarial loss
    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)

    # data augmentation module in stage-1 for the generated images
    aug_list = transforms.Compose([
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5)
    ])

    args = prepare_results_dir(args)

    # Training loop
    for n_iter in range(1, epoch + 1):
        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_inv_loss = 0.
        cumulative_loss_dis = .0

        cumulative_target_acc = .0
        target_correct = 0
        count = 0

        for i in range(n_critic):  # args.ndis=5, Gen update 1 time, Discriminator update ndis times.
            if i == 0:
                fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, generator)
                dis_fake = discriminator(fake, pseudo_y)
                # random transformation on the generated images
                fake_aug = aug_list(fake)
                # calc the L_inv
                inv_loss = criterion(target_model(fake_aug)[-1], pseudo_y)
                # not used
                if args.relativistic_loss:
                    real, y = sample_from_data(args, device, train_loader)
                    dis_real = discriminator(real, y)
                else:
                    dis_real = None
                # calc the loss of G
                loss_gen = gen_criterion(dis_fake, dis_real)
                loss_all = loss_gen + inv_loss * cfg["PLG-MI"]["inv_loss_weight"]
                # update the G
                generator.zero_grad()
                loss_all.backward()
                opt_gen.step()
                _l_g += loss_gen.item()

                cumulative_inv_loss += inv_loss.item()

            # generate fake images
            fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, generator)
            # sample the real images
            real, y = sample_from_data(args, device, train_loader)
            # calc the loss of D
            dis_fake, dis_real = discriminator(fake, pseudo_y), discriminator(real, y)
            loss_dis = dis_criterion(dis_fake, dis_real)
            # update D
            discriminator.zero_grad()
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
        log_interval = 100
        if n_iter % log_interval == 0:
            print(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
                    n_iter, epoch, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                    cumulative_target_acc)
            )
            # Save previews
            utils.save_images(
                n_iter, n_iter // log_interval, args.results_root,
                args.train_image_root, fake, real
            )

    toogle_grad(generator, False)
    toogle_grad(discriminator, False)

    return generator, discriminator


def tune_specific_gan(cfg, generator, discriminator, T, final_z, epochs):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    n_critic = cfg[model_name]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Dataset
    public_dataset = dataloader.ImageFolder(cfg, file_path, mode="gan")
    print(f"public dataset len {len(public_dataset)}")

    with torch.no_grad():
        pseudo_private_dataset = generator(final_z).cpu()
    print(f"pesudo dataset len {len(pseudo_private_dataset)}")

    combined_dataset = ConcatDataset([public_dataset, pseudo_private_dataset])
    print(f"combined dataset len {len(combined_dataset)}")

    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Start Training
    print("Training general GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    toogle_grad(generator, True)
    toogle_grad(discriminator, True)

    dg_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()

    step = 0
    for epoch in range(epochs):
        start = time.time()
        unlabel_loader1 = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True).__iter__()
        unlabel_loader2 = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True).__iter__()

        for i, imgs in enumerate(combined_loader):
            current_iter = epoch * len(combined_loader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = next(unlabel_loader1)
            x_unlabel2 = next(unlabel_loader2)

            toogle_grad(generator, False)
            toogle_grad(discriminator, True)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = generator(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)

            _, output_label = discriminator(imgs)
            _, output_unlabel = discriminator(x_unlabel)
            _, output_fake = discriminator(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel))) - torch.mean(
                log_sum_exp(output_unlabel)) + torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            acc = torch.mean((output_label.max(1)[1] == y).float())

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G
            if step % n_critic == 0:
                toogle_grad(discriminator, False)
                toogle_grad(generator, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = generator(z)
                mom_gen, output_fake = discriminator(f_imgs)
                mom_unlabel, _ = discriminator(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim=0)
                mom_unlabel = torch.mean(mom_unlabel, dim=0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))
    toogle_grad(generator, False)
    toogle_grad(discriminator, False)

    return generator, discriminator


def tune_general_gan(cfg, generator, discriminator, final_z, epochs):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    n_critic = cfg[model_name]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Dataset
    public_dataset = dataloader.ImageFolder(cfg, file_path, mode="gan")
    print(f"public dataset len {len(public_dataset)}")

    with torch.no_grad():
        pseudo_private_dataset = generator(final_z).cpu()
    print(f"pesudo dataset len {len(pseudo_private_dataset)}")

    combined_dataset = ConcatDataset([public_dataset, pseudo_private_dataset])
    print(f"combined dataset len {len(combined_dataset)}")

    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Start Training
    print("Training general GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    toogle_grad(generator, True)
    toogle_grad(discriminator, True)
    dg_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(combined_loader):

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            toogle_grad(generator, False)
            toogle_grad(discriminator, True)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = generator(z)

            r_logit = discriminator(imgs)
            f_logit = discriminator(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data, DG=discriminator)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                toogle_grad(discriminator, False)
                toogle_grad(generator, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = generator(z)
                logit_dg = discriminator(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))

    toogle_grad(generator, False)
    toogle_grad(discriminator, False)

    return generator, discriminator

def train_general_CT_gan(cfg):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    epochs = cfg[model_name]['epochs']
    n_critic = cfg[model_name]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, 'general_CT_GAN'))
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # Log file
    log_path = os.path.join(save_model_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "GAN_{}.txt".format(dataset_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    # writer = SummaryWriter(log_path)

    # Dataset
    print("load training data!")
    dataset, dataloader = init_dataloader(cfg, file_path, batch_size, mode="gan")

    # Start Training
    print("Training general CT GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    G = Generator(z_dim)
    DG = DGWGAN(3)
    DG.ls[5] = torch.nn.Sequential(
        torch.nn.Conv2d(512, 256, kernel_size=(4, 4), stride=(1, 1)),
        # torch.nn.Flatten()
    )

    class Navigator(nn.Module):
        def __init__(self, dim, hidden=512):
            super(Navigator, self).__init__()

            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden // 2)
            self.fc3 = nn.Linear(hidden // 2, 1)
            self.act = nn.LeakyReLU()
            self.norm1 = nn.BatchNorm1d(hidden)
            self.norm2 = nn.BatchNorm1d(hidden // 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.act(x)

            return self.fc3(x)

    navigator = Navigator(dim=256)

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()
    navigator = torch.nn.DataParallel(navigator).cuda()

    lr = 2e-4
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.0, 0.9))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    nav_optimizer = torch.optim.Adam(navigator.parameters(), lr=lr, betas=(0.0, 0.9))

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(dg_optimizer, gamma=0.99)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.99)
    scheduler_n = torch.optim.lr_scheduler.ExponentialLR(nav_optimizer, gamma=0.99)

    step = 0
    rho = 0.5

    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            for _ in range(1):
                toogle_grad(G, False)
                toogle_grad(DG, True)

                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)

                feat = DG(f_imgs).reshape(bs, -1)
                feat = feat / torch.sqrt(torch.sum(feat.square(), dim=-1, keepdim=True))
                feat_x = DG(imgs).reshape(bs, -1)
                feat_x = feat_x / torch.sqrt(torch.sum(feat_x.square(), dim=-1, keepdim=True))

                mse_n = (feat_x[:, None] - feat).pow(2)  # [256, 256, 256]
                cost = mse_n.sum(-1)  # [256, 256]
                d = navigator(mse_n).squeeze().mul(-1)  # [256, 256]
                m_forward = torch.softmax(d, dim=1)  # [256, 256]
                m_backward = torch.softmax(d, dim=0)  # [256, 256]

                disc_loss = - rho * (cost * m_forward).sum(1).mean() - (1 - rho) * (cost * m_backward).sum(0).mean()

                dg_optimizer.zero_grad()
                disc_loss.backward()
                dg_optimizer.step()

            for _ in range(5):
                # train G
                toogle_grad(DG, False)
                toogle_grad(G, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)

                feat = DG(f_imgs).reshape(bs, -1)
                feat = feat / torch.sqrt(torch.sum(feat.square(), dim=-1, keepdim=True))
                feat_x = DG(imgs).reshape(bs, -1)
                feat_x = feat_x / torch.sqrt(torch.sum(feat_x.square(), dim=-1, keepdim=True))

                mse_n = (feat_x[:, None] - feat).pow(2)  # [256, 256, 256]
                cost = mse_n.sum(-1)  # [256, 256]
                d = navigator(mse_n).squeeze().mul(-1)  # [256, 256]
                m_forward = torch.softmax(d, dim=1)  # [256, 256]
                m_backward = torch.softmax(d, dim=0)  # [256, 256]

                g_loss = rho * (cost * m_forward).sum(1).mean() + (1 - rho) * (cost * m_backward).sum(0).mean()

                g_optimizer.zero_grad()
                nav_optimizer.zero_grad()

                g_loss.backward()
                g_optimizer.step()
                nav_optimizer.step()

        scheduler_d.step()
        scheduler_g.step()
        scheduler_n.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))
        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch + 1)),
                               nrow=8)

        torch.save({'state_dict': G.state_dict()}, os.path.join(save_model_dir, "celeba_G.tar"))
        torch.save({'state_dict': DG.state_dict()}, os.path.join(save_model_dir, "celeba_D.tar"))


def train_augmodel(cfg):
    # Hyperparams
    target_model_name = cfg['train']['target_model_name']
    student_model_name = cfg['train']['student_model_name']
    device = cfg['train']['device']
    lr = cfg['train']['lr']
    temperature = cfg['train']['temperature']
    dataset_name = cfg['dataset']['name']
    n_classes = cfg['dataset']['n_classes']
    batch_size = cfg['dataset']['batch_size']
    seed = cfg['train']['seed']
    epochs = cfg['train']['epochs']
    log_interval = cfg['train']['log_interval']

    # Create save folder
    save_dir = os.path.join(cfg['root_path'], dataset_name)
    save_dir = os.path.join(save_dir, '{}_{}_{}_{}'.format(target_model_name, student_model_name, lr, temperature))
    os.makedirs(save_dir, exist_ok=True)

    # Log file    
    now = datetime.now()  # current date and time
    log_file = "studentKD_logs_{}.txt".format(now.strftime("%m_%d_%Y_%H_%M_%S"))
    utils.Tee(os.path.join(save_dir, log_file), 'w')
    torch.manual_seed(seed)

    kwargs = {'batch_size': batch_size}
    if device == 'cuda':
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    # Get models
    teacher_model = get_augmodel(target_model_name, n_classes, cfg['train']['target_model_ckpt'])
    model = get_augmodel(student_model_name, n_classes)
    model = model.to(device)
    print('Target model {}: {} params'.format(target_model_name, count_parameters(model)))
    print('Augmented model {}: {} params'.format(student_model_name, count_parameters(teacher_model)))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Get dataset
    _, dataloader_train = init_dataloader(cfg, cfg['dataset']['gan_file_path'], batch_size, mode="gan")
    _, dataloader_test = init_dataloader(cfg, cfg['dataset']['test_file_path'], batch_size, mode="test")

    # Start training
    top1, top5 = test(teacher_model, dataloader=dataloader_test)
    print("Target model {}: top 1 = {}, top 5 = {}".format(target_model_name, top1, top5))

    loss_function = nn.KLDivLoss(reduction='sum')
    teacher_model.eval()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, data in enumerate(dataloader_train):
            data = data.to(device)

            curr_batch_size = len(data)
            optimizer.zero_grad()
            _, output_t = teacher_model(data)
            _, output = model(data)

            loss = loss_function(
                F.log_softmax(output / temperature, dim=-1),
                F.softmax(output_t / temperature, dim=-1)
            ) / (temperature * temperature) / curr_batch_size

            loss.backward()
            optimizer.step()

            if (log_interval > 0) and (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader_train.dataset),
                           100. * batch_idx / len(dataloader_train), loss.item()))

        scheduler.step()
        top1, top5 = test(model, dataloader=dataloader_test)
        print("epoch {}: top 1 = {}, top 5 = {}".format(epoch, top1, top5))

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir,
                                     "{}_{}_kd_{}_{}.pt".format(target_model_name, student_model_name, seed, epoch + 1))
            torch.save({'state_dict': model.state_dict()}, save_path)
