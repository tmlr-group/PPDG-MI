from copy import deepcopy
import torch, utils
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.autograd import Variable
import torch.optim as optim
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def reg_loss(featureT, fea_mean, fea_logvar):
    fea_reg = reparameterize(fea_mean, fea_logvar)
    fea_reg = fea_mean.repeat(featureT.shape[0], 1)
    loss_reg = torch.mean((featureT - fea_reg).pow(2))
    # print('loss_reg',loss_reg)
    return loss_reg


def attack_acc(fake, iden, E, ):
    eval_prob = E(utils.low2high(fake))[-1]

    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

    cnt, cnt5 = 0, 0
    bs = fake.shape[0]
    # print('correct id')
    for i in range(bs):
        gt = iden[i].item()
        if eval_iden[i].item() == gt:
            cnt += 1
            # print(gt)
        _, top5_idx = torch.topk(eval_prob[i], 5)
        if gt in top5_idx:
            cnt5 += 1
    return cnt * 100.0 / bs, cnt5 * 100.0 / bs


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


def reparameterize_batch(mu, logvar, batch_size):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((batch_size, std.shape[1]), device=std.device)

    return eps * std + mu


def find_criterion(used_loss):
    criterion = None
    if used_loss == 'logit_loss':
        criterion = nn.NLLLoss().to(device)
        print('criterion:{}'.format(used_loss))
    elif used_loss == 'cel':
        criterion = nn.CrossEntropyLoss().to(device)
        print('criterion', criterion)
    else:
        print('criterion:{}'.format(used_loss))
    return criterion


def get_act_reg(train_loader, T, device, Nsample=5000):
    all_fea = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):  # batchsize =100
            # print(data.shape)
            if batch_idx * len(data) > Nsample:
                break
            data = data.to(device)
            fea, _ = T(data)
            if batch_idx == 0:
                all_fea = fea
            else:
                all_fea = torch.cat((all_fea, fea))
    fea_mean = torch.mean(all_fea, dim=0)
    fea_logvar = torch.std(all_fea, dim=0)

    print(fea_mean.shape, fea_logvar.shape, all_fea.shape)
    return fea_mean, fea_logvar


def iden_loss(T, fake, iden, used_loss, criterion, fea_mean=0, fea_logvar=0, lam=0.1):
    Iden_Loss = 0
    loss_reg = 0
    for tn in T:

        feat, out = tn(fake)
        if used_loss == 'logit_loss':  # reg only with the target classifier, reg is randomly from distribution
            if Iden_Loss == 0:
                loss_sdt = criterion(out, iden)
                loss_reg = lam * reg_loss(feat, fea_mean[0], fea_logvar[0])  # reg only with the target classifier

                Iden_Loss = Iden_Loss + loss_sdt
            else:
                loss_sdt = criterion(out, iden)
                Iden_Loss = Iden_Loss + loss_sdt

        elif used_loss == 'cs_loss':
            input_features = feat
            class_weights = tn.module.fc_layer.weight[iden]
            input_features_norms = torch.norm(input_features, p=2, dim=1).mean()
            class_weights_norms = torch.norm(class_weights, p=2, dim=1).mean()

            cosine_similarity = F.cosine_similarity(input_features, class_weights).mean()

            Iden_Loss = -1 * cosine_similarity

        else:
            loss_sdt = criterion(out, iden)
            Iden_Loss = Iden_Loss + loss_sdt

    Iden_Loss = Iden_Loss / len(T) + loss_reg
    return Iden_Loss


def white_dist_inversion(G, D, T, E, iden, batch_size, num_candidates, lr=2e-2, momentum=0.9, lamda=100,
                         iter_times=1500, clip_range=1.0, improved=False, num_seeds=5,
                         used_loss='cel', prefix='', random_seed=0, save_img_dir='', fea_mean=0,
                         fea_logvar=0, lam=0.1, clipz=False):
    iden = iden.view(-1).long().to(device)
    criterion = find_criterion(used_loss)
    bs = iden.shape[0]

    G.eval()
    D.eval()
    E.eval()

    # NOTE
    mu = Variable(torch.zeros(1, 100), requires_grad=True)
    log_var = Variable(torch.ones(1, 100), requires_grad=True)

    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)

    for i in range(iter_times):
        z = reparameterize_batch(mu, log_var, batch_size)
        if clipz == True:
            z = torch.clamp(z, -clip_range, clip_range).float()
        fake = G(z)

        if improved == True:
            _, label = D(fake)
        else:
            label = D(fake)

        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()
        Iden_Loss = iden_loss(T, fake, iden, used_loss, criterion, fea_mean, fea_logvar, lam)

        if improved:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            Prior_Loss = - label.mean()

        Total_Loss = Prior_Loss + lamda * Iden_Loss

        Total_Loss.backward()
        solver.step()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i + 1) % 300 == 0:
            with torch.no_grad():
                z = reparameterize_batch(mu, log_var, batch_size)
                if clipz == True:
                    z = torch.clamp(z, -clip_range, clip_range).float()
                fake_img = G(z.detach())
                eval_prob = E(utils.low2high(fake_img))[-1]

                eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
                print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1,
                                                                                                    Prior_Loss_val,
                                                                                                    Iden_Loss_val,
                                                                                                    acc))

                outputs = T[0](fake)
                confidence_vector = outputs[-1].softmax(dim=1)
                confidences = torch.gather(
                    confidence_vector, 1, iden.unsqueeze(1))
                mean_conf = confidences.mean().detach().cpu()

                min_conf = confidences.min().detach().cpu().item()  # Get minimum confidence
                max_conf = confidences.max().detach().cpu().item()  # Get maximum confidence
                print(f'mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})')

    return reparameterize_batch(mu, log_var, num_candidates)


def white_inversion(G, D, T, E, batch_size, z_init, targets, lr=2e-2, momentum=0.9, lamda=100,
                    iter_times=1500, clip_range=1, improved=False,
                    used_loss='cel', prefix='', save_img_dir='', fea_mean=0,
                    fea_logvar=0, lam=0.1, istart=0, same_z=''):
    criterion = find_criterion(used_loss)

    G.eval()
    D.eval()
    E.eval()

    z_opt = []
    # Prepare batches for attack
    for i in range(math.ceil(z_init.shape[0] / batch_size)):
        z = z_init[i * batch_size:(i + 1) * batch_size]
        iden = targets[i * batch_size:(i + 1) * batch_size]

        target_classes_set = set(iden.tolist())

        print(
            f'Optimizing batch {i + 1} of {math.ceil(z_init.shape[0] / batch_size)} target classes {target_classes_set}.')

        z.requires_grad = True
        v = torch.zeros(z.shape[0], 100).to(device).float()

        for i in range(iter_times):
            fake = G(z)
            if improved == True:
                _, label = D(fake)
            else:
                label = D(fake)

            if z.grad is not None:
                z.grad.data.zero_()

            if improved:
                Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
            else:
                Prior_Loss = - label.mean()

            Iden_Loss = iden_loss(T, fake, iden, used_loss, criterion, fea_mean, fea_logvar, lam)

            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + (- momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if (i + 1) % 300 == 0:
                with torch.no_grad():
                    fake_img = G(z.detach())

                    eval_prob = E(utils.low2high(fake_img))[-1]
                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / batch_size
                    print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.3f}".format(i + 1,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        acc))

                    outputs = T[0](fake)
                    confidence_vector = outputs[-1].softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, iden.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                    min_conf = confidences.min().detach().cpu().item()  # Get minimum confidence
                    max_conf = confidences.max().detach().cpu().item()  # Get maximum confidence
                    print(f'mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})\n')

        z_opt.append(z.detach())

    return torch.concat(z_opt, dim=0)


def black_inversion(agent, G, target_model, alpha, z_init, batch_size, max_episodes, max_step, label, model_name):
    print("Target Label : " + str(label.item()))
    z_opt = []
    for i in range(z_init.shape[0]):
        print(f'Optimizing target class {label} ({i + 1}/{z_init.shape[0]})')
        z = z_init[i].unsqueeze(0)
        start = time.time()
        for i_episode in range(1, max_episodes + 1):
            y = torch.tensor([label]).cuda()

            # Initialize the state at the beginning of each episode.
            state = deepcopy(z.cpu().numpy())
            for t in range(max_step):

                # Update the state and create images from the updated state and action.
                action = agent.act(state)
                z = alpha * z + (1 - alpha) * action.clone().detach().reshape((1, len(action))).cuda()

                next_state = deepcopy(z.cpu().numpy())
                state_image = G(z).detach()
                action_image = G(action.clone().detach().reshape((1, len(action))).cuda()).detach()

                # Calculate the reward.
                _, state_output = target_model(state_image)
                _, action_output = target_model(action_image)
                score1 = float(
                    torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(state_output, dim=-1)).data, 1, y))))
                score2 = float(
                    torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_output, dim=-1)).data, 1, y))))

                score3 = math.log(max(1e-7,
                                      float(torch.index_select(F.softmax(state_output, dim=-1).data, 1, y)) - float(
                                          torch.max(torch.cat((F.softmax(state_output, dim=-1)[0, :y],
                                                               F.softmax(state_output, dim=-1)[0, y + 1:])), dim=-1)[
                                              0])))

                reward = 2 * score1 + 2 * score2 + 8 * score3

                # Update policy.
                if t == max_step - 1:
                    done = True
                else:
                    done = False

                agent.step(state, action, reward, next_state, done, t)
                state = next_state

            if i_episode % 1000 == 0:
                end = time.time()
                print(f"Current episode: {i_episode} Time: {(end - start):2f}")
                start = end

                state = torch.from_numpy(state).float()
                opt_img = G(state).detach()
                _, opt_output = target_model(opt_img)
                probabilities = F.softmax(opt_output, dim=-1)
                target_probabilities = probabilities[torch.arange(probabilities.size(0)), y]
                confidence = float(torch.mean(target_probabilities))
                print(f"confidence: {confidence:2f}")

        z_opt.append(torch.tensor(state))
    return torch.cat(z_opt, dim=0)


# Sample "#points_count" points around a sphere centered on "current_point" with radius =  "sphere_radius"
def gen_points_on_sphere(current_point, points_count, sphere_radius):
    # get random perturbations
    points_shape = (points_count,) + current_point.shape
    perturbation_direction = torch.randn(*points_shape).cuda()
    dims = tuple([i for i in range(1, len(points_shape))])

    # normalize them such that they are uniformly distributed on a sphere with the given radius
    perturbation_direction = (sphere_radius / torch.sqrt(
        torch.sum(perturbation_direction ** 2, axis=dims, keepdims=True))) * perturbation_direction

    # add the perturbations to the current point
    sphere_points = current_point + perturbation_direction
    return sphere_points, perturbation_direction


def label_only_inversion(z, target_id, targets_single_id, G, target_model, E, attack_params, criterion, max_iters_at_radius_before_terminate,
                         current_iden_dir, round_num):
    final_z = []
    start = time.time()

    for i in range(len(z)):
        end = time.time()
        print(f'Optimizing target class {target_id} ({i + 1}/{len(z)}) Time: {(end - start):2f}')
        start = end

        log_file = open(os.path.join(current_iden_dir, 'train_log'), 'w')
        log_file.write("{} of target class: {}".format(i + 1, target_id))

        current_point = z[i]
        current_iter = 0
        last_iter_when_radius_changed = 0

        losses = []

        if round_num == 0:
            # default hyper-parameters used in BREPMI. PPDG uses these parameters for baseline attack.
            current_sphere_radius = 2.0
            sphere_expansion_coeff = 1.3
        else:
            # parameters used in our official work. PPDG uses these parameters for vanilla attack.
            current_sphere_radius = attack_params['BREP_MI']['current_sphere_radius']
            sphere_expansion_coeff = attack_params['BREP_MI']['sphere_expansion_coeff']

        last_success_on_eval = False
        # Outer loop handle all sphere radii
        while current_iter - last_iter_when_radius_changed < max_iters_at_radius_before_terminate:

            # inner loop handle one single sphere radius
            while current_iter - last_iter_when_radius_changed < max_iters_at_radius_before_terminate:

                new_radius = False

                # step size is similar to learning rate
                # we limit max step size to 3. But feel free to change it
                step_size = min(current_sphere_radius / 3, 3)

                # sample points on the sphere
                new_points, perturbation_directions = gen_points_on_sphere(current_point, attack_params['BREP_MI'][
                    'sphere_points_count'], current_sphere_radius)

                # get the predicted labels of the target model on the sphere points
                new_points_classification = is_target_class(G(new_points), target_id, target_model)

                # handle case where all(or some percentage) sphere points lie in decision boundary. We increment sphere size

                if new_points_classification.sum() > 0.75 * attack_params['BREP_MI'][
                    'sphere_points_count']:  # == attack_params['sphere_points_count']:
                    # save_tensor_images(G(current_point.unsqueeze(0))[0].detach(),

                    # save_tensor_images(G(current_point.unsqueeze(0))[0].detach(),
                    #                    os.path.join(current_iden_dir,
                    #                                 "z{}_last_img_of_radius_{:.4f}_iter_{}.png".format(
                    #                                     i, current_sphere_radius, current_iter)))
                    # update the current sphere radius
                    current_sphere_radius = current_sphere_radius * sphere_expansion_coeff

                    log_file.write("new sphere radius at iter: {} ".format(current_iter))
                    new_radius = True
                    last_iter_when_radius_changed = current_iter

                # get the update direction, which is the mean of all points outside boundary if 'repulsion_only' is used. Otherwise it is the mean of all points * their classification (1,-1)
                if attack_params['BREP_MI']['repulsion_only'] == "True":
                    new_points_classification = (new_points_classification - 1) / 2

                grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions,
                                            axis=0) / current_sphere_radius

                # move the current point with stepsize towards grad_direction
                current_point_new = current_point + step_size * grad_direction
                current_point_new = current_point_new.clamp(min=attack_params['BREP_MI']['point_clamp_min'],
                                                            max=attack_params['BREP_MI']['point_clamp_max'])

                current_img = G(current_point_new.unsqueeze(0))

                # current_img = G(current_point_new)
                if is_target_class(current_img, target_id, target_model)[0] == -1:
                    log_file.write("current point is outside target class boundary")
                    break

                current_point = current_point_new

                _, current_loss = decision(current_img, target_model, score=True, criterion=criterion,
                                           target=targets_single_id)

                # if current_iter % 50 == 0 or (current_iter < 200 and current_iter % 20 == 0):
                #     save_tensor_images(current_img[0].detach(),
                #                        os.path.join(current_iden_dir, "iter{}.png".format(current_iter)))

                eval_decision = decision_Evaluator(current_img, E)

                correct_on_eval = True if eval_decision == target_id else False
                if new_radius:
                    point_before_inc_radius = current_point.clone()
                    last_success_on_eval = correct_on_eval
                    break
                iter_str = "iter: {}, current_sphere_radius: {}, step_size: {:.2f} sum decisions: {}, loss: {:.4f}, eval predicted class {}, classified correct on Eval {}".format(
                    current_iter, current_sphere_radius, step_size,
                    new_points_classification.sum(),
                    current_loss.item(),
                    eval_decision,
                    correct_on_eval)

                log_file.write(iter_str + '\n')
                losses.append(current_loss.item())
                current_iter += 1

            if round_num == 0:  # baseline setting
                if current_sphere_radius > 16.30:
                    print("Reach maximum radius, break!")
                    break
            else:               # PPDG setting
                if current_sphere_radius > 8.91:
                    print("Reach maximum radius, break!")
                    break

        log_file.close()
        final_z.append(current_point.unsqueeze(0))
    return torch.cat(final_z)