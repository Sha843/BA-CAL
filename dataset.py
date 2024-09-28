
import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import logging
from scipy.sparse import csr_matrix
from sample_methods import ClassBalanceSampler, InstanceBalanceSampler, InstanceBalanceSampler_word, \
    ClassBalanceSampler2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(args, sample_level_da=None):
    # Load data
    train_texts = np.load(os.path.join(args.data_dir, args.train_texts), allow_pickle=True)
    train_labels = np.load(os.path.join(args.data_dir, args.train_labels), allow_pickle=True)
    test_texts = np.load(os.path.join(args.data_dir, args.test_texts), allow_pickle=True)
    test_labels = np.load(os.path.join(args.data_dir, args.test_labels), allow_pickle=True)


    emb_init = get_word_emb(os.path.join(args.data_dir, args.emb_init))
    X_train, X_valid, train_y, valid_y = train_test_split(train_texts, train_labels,
                                                          test_size=args.valid_size,
                                                          random_state=args.seed)
    X_test = test_texts
    test_y = test_labels

    mlb = get_mlb(os.path.join(args.data_dir, args.labels_binarizer), np.hstack((train_y, valid_y)))
    y_train, y_valid = mlb.transform(train_y), mlb.transform(valid_y)
    y_test = mlb.transform(test_y)
    args.label_size = len(mlb.classes_)

    if sample_level_da:
        X_train, y_train = slda(X_train, y_train)

    logger.info(f'Size of Training Set: {len(X_train)}')
    logger.info(f'Size of Validation Set: {len(X_valid)}')

    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train.A).type(torch.LongTensor))

    val_data = data_utils.TensorDataset(torch.from_numpy(X_valid).type(torch.LongTensor),
                                        torch.from_numpy(y_valid.A).type(torch.LongTensor))

    test_data = data_utils.TensorDataset(torch.from_numpy(X_test).type(torch.LongTensor),
                                         torch.from_numpy(y_test.A).type(torch.LongTensor))
    # Get class frequencies
    # _, freq_train, _, _ = get_cls_freq('vireo')
    ind_train, label_to_count, freq_train_nonzero_indices, freq_train, freq_test_nonzero_indices, freq_test = get_cls_freq(
        'vireo')

    if args.rebalance == 'cb_resample2':
        train_sampler = ClassBalanceSampler2(
            train_data, label_to_count, args.beta_rs, args.result_path, ind_train
        )
        num_samples = len(list(train_sampler))
        print(f"cb_resample2 Number of samples in train_sampler: {num_samples}")

        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
                                  num_workers=4)
        print("sampling")
        train_size = len(train_loader.dataset)
        print(f"Training dataset size: {train_size}")
        num_samples_in_loader = 0
        for batch in train_loader:
            num_samples_in_loader += len(batch[0])
        print(f"Number of samples processed by train_loader: {num_samples_in_loader}")

    elif args.rebalance == 'instance_balance_word':
        train_sampler = InstanceBalanceSampler_word(
            train_data, label_to_count, args.beta_rs, args.result_path, ind_train
        )
        num_samples = len(list(train_sampler))
        print(f"instance_balance_word Number of samples in train_sampler: {num_samples}")
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
                                  num_workers=4)
        print("sampling")
        train_size = len(train_loader.dataset)
        print(f"Training dataset size: {train_size}")
        num_samples_in_loader = 0
        for batch in train_loader:
            num_samples_in_loader += len(batch[0])
        print(f"Number of samples processed by train_loader: {num_samples_in_loader}")
    elif args.rebalance == 'instance_balance':
        train_sampler = InstanceBalanceSampler(
            train_data, freq_train, args.beta_rs, args.result_path
        )
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
                                  num_workers=4)
    elif args.rebalance == 'cb_resample':
        train_sampler = ClassBalanceSampler(
            train_data, freq_train, args.beta_rs, args.result_path
        )
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
                                  num_workers=4)
    else:
        print('no sample')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=False)
    return train_loader, val_loader, test_loader, emb_init, mlb, args

def get_word_emb(vec_path, vocab_path=None):
    if vocab_path is not None:
        with open(vocab_path) as fp:
            vocab = {word: idx for idx, word in enumerate(fp)}
        return np.load(vec_path, allow_pickle=True), vocab
    else:
        return np.load(vec_path, allow_pickle=True)

def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb

def slda(X_train, y_train):
    #sample level data augemtation
    y_train = y_train.A
    x = []
    y = []
    for i in range(10):
        x.append(X_train)
        y.append(y_train)
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (-1, x.shape[-1]))
    y = np.reshape(y, (-1, y.shape[-1]))
    y = csr_matrix(y)
    return x, y


def get_cls_num_list(dataset):
    if dataset == 'vireo':
        ind_train = np.load('train_labels.npy',
                            allow_pickle=True)
        ind_test = np.load('test_labels.npy',
                           allow_pickle=True)
        mlb = MultiLabelBinarizer()
        mlb.fit(np.concatenate((ind_train, ind_test)))
        ind_train_binarized = mlb.transform(ind_train)
        ind_test_binarized = mlb.transform(ind_test)

        cls_num_train = np.where(ind_train_binarized.sum(axis=0) > 0)[0]
        cls_num_test = np.where(ind_test_binarized.sum(axis=0) > 0)[0]
        return cls_num_train, cls_num_test
    else:
        raise ValueError(f"Error：{dataset}")


def get_cls_freq(dataset):
    if dataset == 'vireo':
        train_labels_path = 'train_labels.npy'
        test_labels_path = 'test_labels.npy'

        ind_train = np.load(train_labels_path, allow_pickle=True)
        ind_test = np.load(test_labels_path, allow_pickle=True)

        mlb = MultiLabelBinarizer()
        mlb.fit(np.concatenate((ind_train, ind_test)))
        ind_train_binarized = mlb.transform(ind_train)
        ind_test_binarized = mlb.transform(ind_test)

        freq_train = ind_train_binarized.sum(axis=0)
        freq_test = ind_test_binarized.sum(axis=0)

        label_to_count = dict(zip(mlb.classes_, freq_train))

        ind_train_indices = np.arange(len(ind_train))

        return ind_train_indices, label_to_count, np.where(freq_train > 0)[0], freq_train, np.where(freq_test > 0)[
            0], freq_test
    else:
        raise ValueError(f"Error：{dataset}")


def reweight_method(opt, epoch):
    if opt.rebalance == 'cb_reweight':
        beta = opt.beta_rw
        cls_num_train, _ = get_cls_num_list('vireo')
        effective_num = 1.0 - np.power(beta, cls_num_train)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_train)
        per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float).cuda()
        return per_cls_weights
    elif opt.rebalance == 'ldam_drw':
        cls_num_train, _ = get_cls_num_list('vireo')
        idx = 1 if epoch // int(opt.epoch_drw) > 0 else 0
        betas = [0, opt.beta_rw]
        effective_num = 1.0 - np.power(betas[idx], cls_num_train)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_train)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        return per_cls_weights
    else:
        return None


def resample_method(opt, train_set, cls_num_train, epoch):
    if opt.rebalance == 'cb_resample':
        result_path = opt.result_path
        ind_train = opt.ind_train
        train_sampler = ClassBalanceSampler(train_set, cls_num_train, opt.beta_rs, result_path, ind_train)
        return train_sampler
    elif opt.rebalance == 'crt_rv':
        result_path = opt.result_path
        ind_train = opt.ind_train
        if epoch >= opt.epoch_crt:
            train_sampler = InstanceBalanceSampler(train_set, cls_num_train, 1, result_path, ind_train)
            return train_sampler
        else:
            return None
    elif opt.rebalance == 'crt_sr':
        result_path = opt.result_path
        ind_train = opt.ind_train
        if epoch >= opt.epoch_crt:
            train_sampler = InstanceBalanceSampler(train_set, cls_num_train, 0.5, result_path, ind_train)
            return train_sampler
        else:
            return None
    else:
        return None





