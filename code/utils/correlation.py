import torch


def cal_correlation_matrix(features, prob, label, memory, num_classes, temperature=0.1, num_filtered=64):
    """
    features: projector -> predictor -> features [num_consist_filtered, embedding_dim]

    Flexible implementation of the omni-correlation matrix calculation

    """
    correlation_list = []

    for i in range(num_classes):
        class_mask = label == i
        class_feature = features[class_mask, :]  # [num, dim]
        class_prob = prob[class_mask]
        _, indices = torch.sort(class_prob, descending=True)
        indices = indices[:num_filtered]
        class_feature = class_feature[indices]  # [num_filtered, feat_dim]

        logits_list = []
        for memory_c in memory:
            if memory_c is not None and class_feature.shape[0] > 1 and memory_c.shape[0] > 1:
                logits = torch.matmul(class_feature, torch.from_numpy(memory_c.T).cuda())
                logits_list.append(logits)
        # for each unlabeled data sampled online, calculate its corresponding correlation among the whole memory bank
        if logits_list:
            logits = torch.cat(logits_list, dim=1)
            correlation = torch.softmax(logits / temperature, dim=1)  # [1, num]
            correlation_list.append(correlation)
    if not correlation_list:
        return [], False
    else:
        correlation_list = torch.cat(correlation_list, dim=0)
        return correlation_list, True
