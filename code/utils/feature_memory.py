import torch
import random
import numpy as np


class MemoryBank:
    def __init__(self, num_labeled_samples, num_cls):
        self.memory_feature = [None] * num_labeled_samples
        self.memory_label = [None] * num_labeled_samples
        self.num_cls = num_cls

    def update_labeled_features(self, features, labels, idxs):
        for i in range(len(idxs)):
            self.memory_feature[idxs[i].cpu().numpy()] = features[i].detach().cpu().numpy()
            self.memory_label[idxs[i].cpu().numpy()] = labels[i].detach().cpu().numpy()

    def sample_labeled_features(self, num_sampled_per_cls):
        tmp_feature_list = []
        tmp_label_list = []
        for i in range(len(self.memory_label)):
            if self.memory_label[i] is not None:
                tmp_feature_list.append(self.memory_feature[i])
                tmp_label_list.append(self.memory_label[i])
        tmp_feature_list = np.concatenate(tmp_feature_list, axis=0)
        tmp_label_list = np.concatenate(tmp_label_list, axis=0)
        selected_feature_list = []
        for c in range(self.num_cls):
            mask_c = tmp_label_list == c
            features_c = tmp_feature_list[mask_c]
            if features_c.shape[0] >= num_sampled_per_cls:
                num_features = features_c.shape[0]
                selected = torch.tensor(random.sample(range(num_features), num_sampled_per_cls)).numpy()
                selected_feature_list.append(features_c[selected])
            else:
                selected_feature_list.append(None)
        return selected_feature_list


def sample_labeled_features_from_both_memory_bank(memory_a, memory_b, num_sampled_per_cls):
    tmp_feature_list_a, tmp_feature_list_b = [], []
    tmp_label_list = []
    for i in range(len(memory_a.memory_label)):
        if memory_a.memory_label[i] is not None:
            tmp_feature_list_a.append(memory_a.memory_feature[i])
            tmp_feature_list_b.append(memory_b.memory_feature[i])
            tmp_label_list.append(memory_a.memory_label[i])
    tmp_feature_list_a = np.concatenate(tmp_feature_list_a, axis=0)
    tmp_feature_list_b = np.concatenate(tmp_feature_list_b, axis=0)
    tmp_label_list = np.concatenate(tmp_label_list, axis=0)

    selected_feature_list_a,  selected_feature_list_b = [], []
    for c in range(memory_a.num_cls):
        mask_c = tmp_label_list == c
        features_a = tmp_feature_list_a[mask_c]
        features_b = tmp_feature_list_b[mask_c]
        if features_a.shape[0] >= num_sampled_per_cls:
            num_features = features_a.shape[0]
            selected = torch.tensor(random.sample(range(num_features), num_sampled_per_cls)).numpy()
            selected_feature_list_a.append(features_a[selected])
            selected_feature_list_b.append(features_b[selected])
        else:
            selected_feature_list_a.append(None)
            selected_feature_list_b.append(None)
    return selected_feature_list_a, selected_feature_list_b
