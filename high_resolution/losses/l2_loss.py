import torch

l2_criterion = torch.nn.MSELoss(reduction='mean')


def l2_loss(real_images, generated_images):
    loss = l2_criterion(real_images, generated_images)
    return loss


def l2_loss_all_pairs(real_images, generated_images):
    # 扩展 real_images 和 generated_images 来形成所有可能的配对
    # real_images: [N, C, H, W] -> [N, 1, C, H, W]
    # generated_images: [M, C, H, W] -> [1, M, C, H, W]
    # 扩展后的形状为 [N, M, C, H, W]
    real_imgs_exp = real_images.unsqueeze(1)
    gen_imgs_exp = generated_images.unsqueeze(0)

    # 计算所有配对的 L2 损失
    loss_matrix = torch.nn.functional.mse_loss(real_imgs_exp, gen_imgs_exp, reduction='none')
    loss_matrix = loss_matrix.mean(dim=[2, 3, 4])  # 在 C, H, W 维度上计算均值

    # 返回 NxM 损失矩阵
    return loss_matrix.mean()
