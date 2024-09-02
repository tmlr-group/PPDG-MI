from torch.utils.data import TensorDataset, DataLoader
from metrics.classification_acc import ClassificationAccuracy
import torch
import os
import csv
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

from metrics.distance_metrics import DistanceEvaluation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        if file_exists:
            for row in precision_list[1:]:
                wr.writerow(row)
        else:
            for row in precision_list:
                wr.writerow(row)
    return filename


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

    transforms = T.Compose([
        T.RandomHorizontalFlip(0.5)
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
    final_targets = torch.cat(final_targets, dim=0)
    final_z = torch.cat(final_z, dim=0)
    # return final_z, final_targets
    return final_z, final_targets


def log_nearest_neighbors_local(target_id, imgs, targets, evaluater,
                                save_dir, nrow, round_num):
    # Find closest training samples to final results
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)

    grid = vutils.make_grid(closest_samples, nrow=nrow, padding=2, normalize=True)
    # 转换为 PIL 图像并保存
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    img.save(os.path.join(save_dir, f'nearest_neighbor_r{round_num + 1}.png'))

    for i, (img, d) in enumerate(zip(closest_samples, distances)):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(os.path.join(save_dir, f'r{round_num + 1}_{i:02d}_target={target_id:03d}_distance_{d:.2f}.png'))

    return


def log_final_images_local(target_id, imgs, predictions, max_confidences, target_confidences,
                           idx2cls, save_dir, nrow, round_num):
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    img.save(os.path.join(save_dir, f'final_images_r{round_num + 1}.png'))

    for i, (img, pred, max_conf, target_conf) in enumerate(
            zip(imgs.cpu(), predictions, max_confidences, target_confidences)):
        img = img.permute(1, 2, 0).detach().numpy()
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(os.path.join(save_dir,
                              f'r{round_num + 1}_{i:02d}_target={target_id:03d} ({target_conf:.2f})_pred={idx2cls[pred.item()]:03d} ({max_conf:.2f}).png'))

    return


def evaluate_results(E, G, batch_size, round_num, current_time, prefix, final_z, final_targets, trainset, targets_single_id, save_dir):
    target_id = targets_single_id[0]
    evaluation_model = E
    evaluation_model.to(device)
    evaluation_model.eval()
    class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                 device=device)

    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
        final_z,
        final_targets,
        G,
        batch_size=batch_size,
    )

    filename_precision = write_precision_list(
        f'{prefix}/{current_time}/precision_list_r{round_num + 1}',
        precision_list)

    print(
        f'\nEvaluation of {final_z.shape[0]} images on Eval Model: \taccuracy@1={acc_top1:4f}',
        f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )


    avg_dist_facenet = None
    # Compute average feature distance on facenet
    evaluater_facenet = DistanceEvaluation(evaluation_model, G, trainset)
    avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
        final_z,
        final_targets,
        batch_size=batch_size)

    filename_distance = write_precision_list(
        f'{prefix}/{current_time}/distance_facenet_r{round_num + 1}',
        mean_distances_list)

    print('Mean Distance on Eval Model: ', avg_dist_facenet.cpu().item())

    print('Finishing attack, logging results and creating sample images.')

    class KeyDict(dict):
        def __missing__(self, key):
            return key

    idx_to_class = KeyDict()

    num_classes = len(set(targets_single_id.tolist()))
    num_imgs = 10
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
        mask = torch.where(final_targets == label, True, False).cpu()
        z_masked = final_z[mask][:num_imgs]
        imgs = G(z_masked)
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

    log_final_images_local(target_id,
                           log_imgs,
                           log_predictions,
                           log_max_confidences,
                           log_target_confidences,
                           idx_to_class,
                           save_dir, nrow=num_imgs, round_num=round_num)

    log_nearest_neighbors_local(target_id,
                                log_imgs,
                                log_targets,
                                evaluater_facenet,
                                save_dir=save_dir, nrow=num_imgs, round_num=round_num)
