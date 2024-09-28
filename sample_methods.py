import torch
import numpy as np
from collections import Counter
import os
from torch.utils.data import Sampler

class ClassBalanceSampler2(Sampler):
    def __init__(self, dataset, label_to_count, beta, result_path, ind_train, indices=None, num_samples=7112):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.label_to_count = label_to_count
        self.beta = beta
        self.result_path = result_path
        self.ind_train = ind_train
        effective_num = 1.0 - np.power(beta, list(label_to_count.values()))
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        weights = self._multi_label_weight(dataset, per_cls_weights)

        self.weights = torch.DoubleTensor(weights)
        os.makedirs(result_path, exist_ok=True)
        self.log_sample = open(os.path.join(result_path, 'sample_list.csv'), 'w')


    def _get_label(self, dataset, idx):
        return dataset.tensors[1][idx].numpy()

    def _multi_label_weight(self, dataset, per_cls_weights):
        label_dataset = torch.tensor(dataset.tensors[1].numpy(), dtype=torch.double)
        weight_sum = torch.mm(label_dataset, torch.tensor(per_cls_weights).unsqueeze(1))
        label_weight = 1.0 / torch.sum(label_dataset, dim=1)
        weights = weight_sum.squeeze(1) * label_weight
        return weights

    def __iter__(self):
        data_sampled = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        return iter(data_sampled)

    def __len__(self):
        return self.num_samples

class ClassBalanceSampler(Sampler):
    def __init__(self, dataset, freq_train, beta_rs, result_path, ind_train=None):
        self.dataset = dataset
        self.freq_train = freq_train
        self.beta_rs = beta_rs
        self.result_path = result_path
        self.ind_train = ind_train
        self.labels = dataset.tensors[1].numpy()
        self.num_classes = self.labels.shape[1]
        self.class_counts = np.sum(self.labels, axis=0)
        self.class_weights = self._compute_class_weights()

    def _compute_class_weights(self):
        weights = np.array(self.freq_train, dtype=np.float32)
        weights = 1.0 / (weights + 1e-8)
        weights /= np.sum(weights)
        return weights

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)
        sample_weights = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            label = self.labels[i]
            weight = np.sum(self.class_weights * label)
            sample_weights[i] = weight
        sampled_indices = np.random.choice(indices, size=num_samples, replace=True,
                                           p=sample_weights / np.sum(sample_weights))
        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)

class InstanceBalanceSampler(Sampler):
    def __init__(self, dataset, freq_train, beta_rs, result_path, ind_train=None):
        self.dataset = dataset
        self.freq_train = freq_train
        self.beta_rs = beta_rs
        self.result_path = result_path
        self.ind_train = ind_train
        self.labels = dataset.tensors[1].numpy()
        self.num_classes = self.labels.shape[1]
        self.class_counts = np.sum(self.labels, axis=0)

    def _compute_instance_weights(self):
        weights = np.zeros(len(self.dataset), dtype=np.float32)
        for i, label in enumerate(self.labels):
            instance_weight = np.sum(label)
            weights[i] = instance_weight
        return weights

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)
        instance_weights = self._compute_instance_weights()
        instance_weights /= np.sum(instance_weights)
        sampled_indices = np.random.choice(indices, size=num_samples, replace=True,
                                           p=instance_weights)
        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)


class InstanceBalanceSampler_word(Sampler):
    def __init__(self, dataset, label_to_count, beta, result_path, ind_train, indices=None, num_samples=7112):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.arrary_label = [0] * len(label_to_count)
        self.label_to_count = label_to_count
        label_counts = np.array(list(label_to_count.values()))

        if beta == 1:
            per_cls_weights = np.sum(label_counts) / label_counts
        elif beta == 0.5:
            per_cls_weights = np.sqrt(np.sum(label_counts)) / np.sqrt(label_counts)
        else:
            per_cls_weights = (1 - beta) / (1 - np.power(beta, label_counts))
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(label_counts)

        if isinstance(dataset[0][1], torch.Tensor) and dataset[0][1].dim() == 0:
            weights = [per_cls_weights[self._get_label(dataset, idx)] for idx in self.indices]
        else:
            weights = self._multi_label_weight(dataset, per_cls_weights)


        self.weights = torch.DoubleTensor(weights)
        os.makedirs(result_path, exist_ok=True)
        self.log_sample = open(os.path.join(result_path, 'sample_list.csv'), 'w')
        self.ind_train = ind_train
        self.log_sample.write('sample_id,label\n')

    def _get_label(self, dataset, idx):
        return dataset[idx][1].item()

    def _multi_label_weight(self, dataset, per_cls_weights):
        label_dataset = [sample[1].numpy() for sample in dataset]
        weight_sum = torch.mm(torch.tensor(label_dataset).double(), torch.tensor(per_cls_weights).unsqueeze(1))
        label_weight = 1.0 / torch.sum(torch.tensor(label_dataset), dim=1)
        weights = weight_sum.squeeze(1) * label_weight.double()
        return weights


    def __iter__(self):
        data_sampled = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        if isinstance(self.dataset[0][1], torch.Tensor) and self.dataset[0][1].dim() == 0:
            samplecount = np.array(sample_count(self.dataset, self.label_to_count, data_sampled))
            self.log_sample.write(str(samplecount[self.ind_train]) + '\n')
            self.log_sample.flush()
        else:
            class_list = torch.sum(torch.tensor([self.dataset[idx][1].numpy() for idx in data_sampled]), dim=0)
            if isinstance(self.ind_train, int):
                ind_train_indices = [self.ind_train]
            else:
                ind_train_indices = self.ind_train

            valid_indices = [i for i in ind_train_indices if i < len(class_list)]
            class_list = class_list[valid_indices]

            self.log_sample.write(str(class_list) + '\n')
            self.log_sample.flush()
        return iter(data_sampled)

def sample_count(dataset, label_to_count, data_sampled):
    num_class = len(label_to_count)
    labels = np.array([dataset[idx][1].item() for idx in range(len(dataset))])
    class_list = [0] * num_class
    class_count = Counter(labels[data_sampled])
    for c in class_count:
        class_list[c] += class_count[c]
    return class_list