import os
import sys
import shutil
import argparse
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import ramps, losses, test_patch, feature_memory, correlation
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_lambda_c(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_lambda_o(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency_o * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CAML', help='exp_name')
parser.add_argument('--model', type=str, default='caml3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=4, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance supervised loss')
parser.add_argument('--consistency', type=float, default=1, help='lambda_c')
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--embedding_dim', type=int, default=64, help='dim of embeddings to calculate similarity')
parser.add_argument('--num_filtered', type=int, default=12800,
                    help='num of unlabeled embeddings to calculate similarity')
args = parser.parse_args()

snapshot_path = "./model/LA_{}_{}_memory{}_feat{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
    args.exp,
    args.labelnum,
    args.memory_num,
    args.embedding_dim,
    args.num_filtered,
    args.consistency,
    args.consistency_rampup,
    args.consistency_o,
    args.max_iteration,
    args.seed,
    args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    memory_bank = feature_memory.MemoryBank(num_labeled_samples=args.labelnum, num_cls=num_classes)
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),
                           with_idx=True)
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
            volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

            model.train()
            outputs_v, outputs_a, embedding_v, embedding_a = model(volume_batch)
            outputs_list = [outputs_v, outputs_a]
            num_outputs = len(outputs_list)

            y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape)

            loss_s = 0
            for i in range(num_outputs):
                y = outputs_list[i][:labeled_bs, ...]
                y_prob = F.softmax(y, dim=1)
                loss_s += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs, ...] == 1)

                y_all = outputs_list[i]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[i] = y_prob_all
                y_pseudo_label[i] = sharpening(y_prob_all)

            loss_c = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

            outputs_v_soft = F.softmax(outputs_v, dim=1)  # [batch, num_class, h, w, d]
            outputs_a_soft = F.softmax(outputs_a, dim=1)  # soft prediction of fa
            labeled_features_v = embedding_v[:args.labeled_bs, ...]
            labeled_features_a = embedding_a[:args.labeled_bs, ...]

            # unlabeled embeddings to calculate correlation matrix with embeddings sampled from the memory bank
            unlabeled_features_v = embedding_v[args.labeled_bs:, ...]
            unlabeled_features_a = embedding_a[args.labeled_bs:, ...]

            y_v = outputs_v_soft[:args.labeled_bs]
            y_a = outputs_a_soft[:args.labeled_bs]
            true_labels = label_batch[:args.labeled_bs]

            _, prediction_label_v = torch.max(y_v, dim=1)
            _, prediction_label_a = torch.max(y_a, dim=1)
            predicted_unlabel_prob_v, predicted_unlabel_v = torch.max(outputs_v_soft[args.labeled_bs:],
                                                                      dim=1)  # v_unlabeled_mask
            predicted_unlabel_prob_a, predicted_unlabel_a = torch.max(outputs_a_soft[args.labeled_bs:],
                                                                      dim=1)  # a_unlabeled_mask

            # Select the correct predictions including the foreground class and the background class
            mask_prediction_correctly = (
                    ((prediction_label_a == true_labels).float() + (prediction_label_v == true_labels).float()) == 2)

            labeled_features_v = labeled_features_v.permute(0, 2, 3, 4, 1).contiguous()
            b, h, w, d, labeled_features_dim = labeled_features_v.shape

            # get projected features
            model.eval()
            proj_labeled_features_v = model.projection_head1(labeled_features_v.view(-1, labeled_features_dim))
            proj_labeled_features_v = proj_labeled_features_v.view(b, h, w, d, -1)

            proj_labeled_features_a = model.projection_head2(labeled_features_a.view(-1, labeled_features_dim))
            proj_labeled_features_a = proj_labeled_features_a.view(b, h, w, d, -1)
            model.train()

            labels_correct_list = []
            labeled_features_correct_list = []
            labeled_index_list = []
            for i in range(args.labeled_bs):
                labels_correct_list.append(true_labels[i][mask_prediction_correctly[i]])
                labeled_features_correct_list.append((proj_labeled_features_v[i][mask_prediction_correctly[i]] +
                                                      proj_labeled_features_a[i][mask_prediction_correctly[i]]) / 2)
                labeled_index_list.append(idx[i])

            # updated memory bank
            labeled_index = idx[:args.labeled_bs]
            memory_bank.update_labeled_features(labeled_features_correct_list, labels_correct_list,
                                                labeled_index_list)

            # sample memory bank size labeled features from memory bank
            memory = memory_bank.sample_labeled_features(args.memory_num)

            # get the mask with the same prediction between fv and fa on unlabeled data
            mask_consist_unlabeled = predicted_unlabel_v == predicted_unlabel_a  # [b, h, w, d]
            # use model V's predicted label and prob to filter unlabeled feature online
            consist_unlabel = predicted_unlabel_v[mask_consist_unlabeled]  # [num_consist]
            consist_unlabel_prob = predicted_unlabel_prob_v[mask_consist_unlabeled]  # [num_consist]

            unlabeled_features_v = unlabeled_features_v.permute(0, 2, 3, 4, 1)
            unlabeled_features_a = unlabeled_features_a.permute(0, 2, 3, 4, 1)
            unlabeled_features_v = unlabeled_features_v[mask_consist_unlabeled, :]  # [num_consist, feat_dim]
            unlabeled_features_a = unlabeled_features_a[mask_consist_unlabeled, :]

            # get fv's correlation matrix
            projected_feature_v = model.projection_head1(unlabeled_features_v)
            predicted_feature_v = model.prediction_head1(projected_feature_v)
            corr_v, corr_v_available = correlation.cal_correlation_matrix(predicted_feature_v,
                                                                          consist_unlabel_prob,
                                                                          consist_unlabel,
                                                                          memory,
                                                                          num_classes,
                                                                          num_filtered=args.num_filtered)

            # get fa's correlation matrix
            projected_feature_a = model.projection_head2(unlabeled_features_a)
            predicted_feature_a = model.prediction_head2(projected_feature_a)
            corr_a, corr_a_available = correlation.cal_correlation_matrix(predicted_feature_a,
                                                                          consist_unlabel_prob,
                                                                          consist_unlabel,
                                                                          memory,
                                                                          num_classes,
                                                                          num_filtered=args.num_filtered)

            # calculate omni-correlation consistency loss
            if corr_v_available and corr_a_available:
                num_samples = corr_a.shape[0]
                loss_o = torch.sum(torch.sum(-corr_a * torch.log(corr_v + 1e-8), dim=1)) / num_samples
            else:
                loss_o = 0

            iter_num = iter_num + 1
            lambda_c = get_lambda_c(iter_num // 150)
            lambda_o = get_lambda_o(iter_num // 150)

            loss = args.lamda * loss_s + lambda_c * loss_c + lambda_o * loss_o

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_o: %03f' % (
                iter_num, loss, loss_s, loss_c, loss_o))

            writer.add_scalar('Labeled_loss/loss_s', loss_s, iter_num)
            writer.add_scalar('Co_loss/loss_c', loss_c, iter_num)
            writer.add_scalar('Co_loss/loss_o', loss_o, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                # Notification!
                # Here we just save the best result to perform performance comparison with some SOTA methods that
                # report their best results during training obtained during training.
                # In our paper, we only use the model and corresponding results from the final training iteration.
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
