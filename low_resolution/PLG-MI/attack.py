import kornia
import math
from utils import sample_z
import torch
import losses as L

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def PLG_inversion(args, G, D, T, E, batch_size, targets, lr=2e-2, MI_iter_times=600):
    G.eval()
    D.eval()
    E.eval()
    T.eval()

    aug_list = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    )

    z_opt = []
    # Prepare batches for attack
    for i in range(math.ceil(len(targets) / batch_size)):
        z = sample_z(
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

                    eval_prob = E(kornia.augmentation.Resize((112, 112))(fake_img))[-1]
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