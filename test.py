import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import sparse
from scipy.sparse import csr_matrix
import os
import warnings
warnings.filterwarnings('ignore')

def valid(model, test_data_loader, mlb, args):
    pre_K = 5
    model.to(args.device)
    model.eval()
    y_test = None
    y_pred = None
    f1_s3 = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(test_data_loader):
            src, trg = batch
            input_id = src.to(args.device)
            test_label = trg.to(args.device)
            output,_ = model(input_id)
            input2 = src.cuda()
            target2 = trg.cuda()
            output2,_ = model(input2)
            target = target2.data.cpu().float()
            output3 = output2.data.cpu()
            output3[output3 > 0.5] = 1
            output3[output3 <= 0.5] = 0
            f1_s3 += f1_score(target, output3, average='micro', zero_division=1)
            if y_test is None:
                y_test = test_label
                y_pred = output
            else:
                y_test = torch.cat((y_test, test_label), 0)
                y_pred = torch.cat((y_pred, output), 0)
    f1_s3 = f1_s3/len(test_data_loader)
    y_scores_, y_pred1 = torch.topk(y_pred, pre_K)
    y_test = y_test.detach().cpu().numpy()
    y_pred2 = y_pred1.detach().cpu().numpy()
    result = evaluate_valid(y_test, y_pred2, mlb, args)
    result.append(f1_s3)
    return result

def test(model, test_loader, mlb, args):
    pre_K = 5
    model.eval()
    y_pred = None
    f1 = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(test_loader):
            src, trg = batch
            input_id = src.to(args.device)
            output,_ = model(input_id)
            input2 = src.cuda()
            target2 = trg.cuda()
            output2,_ = model(input2)
            target = target2.data.cpu().float()
            output3 = output2.data.cpu()
            output3[output3 > 0.5] = 1
            output3[output3 <= 0.5] = 0
            f1 += f1_score(target, output3, average='micro', zero_division=1)

            if y_pred is None:
                y_pred = output
            else:
                y_pred = torch.cat((y_pred, output), 0)

    _, top_pre_k = torch.topk(y_pred, pre_K)
    top_pre_k = top_pre_k.cpu().numpy()
    f1 = f1 / len(test_loader)
    top_pre_k = mlb.classes_[top_pre_k]
    test_labels = np.load(os.path.join(args.data_dir, args.test_labels), allow_pickle=True)
    for sublist in test_labels:
        if sublist:
            sublist[-1] = sublist[-1].rstrip('\n')
    if args.save_prediction==True:
        np.save(args.prediction_path, top_pre_k)
        print(f'prediction saved to {args.prediction_path}')

    mlb = MultiLabelBinarizer(sparse_output=True)
    y_test = mlb.fit_transform(test_labels)
    result = evaluate_test(y_test, top_pre_k, mlb, args)
    result.append(f1)
    return result

def macro_precision(true, pred):
    hit = 0
    for i, j in zip(true, pred):
        if i == 1 and j == 1 and i == j:
            hit += 1
    p = hit / sum(pred)
    return p

def my_precision_score(y_test, y_pred_category):
    record = []
    for true, pred in zip(y_test.T, y_pred_category.T):
        if sum(pred) == 0:
            p = 0
        else:
            p = macro_precision(true, pred)
        record.append(p)
    return np.array(record)

def compute_macro_p5(prediction, targets, mlb, top_K):
    targets = sparse.csr_matrix(targets)
    prediction = mlb.transform(prediction[:, :top_K])
    p5 = precision_score(targets, prediction, average=None)
    print(p5.mean())
    print(p5.shape)

def evaluate_valid(targets, prediction, args):
    def get_precision(prediction, targets, mlb, top_K):
        targets = sparse.csr_matrix(targets)
        prediction = mlb.transform(prediction[:, :top_K])
        precision = prediction.multiply(targets).sum() / (top_K * targets.shape[0])
        return round(precision * 100, 2)

    def get_ndcg(prediction, targets, top_K,):
        mlb = MultiLabelBinarizer(classes=range(targets.shape[1]), sparse_output=True)
        mlb.fit(None)
        log = 1.0 / np.log2(np.arange(top_K) + 2)
        dcg = np.zeros((targets.shape[0], 1))
        targets = sparse.csr_matrix(targets)
        for i in range(top_K):
            p = mlb.transform(prediction[:, i: i + 1])
            dcg += p.multiply(targets).sum(axis=-1) * log[i]
        ndcg = np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top_K) - 1])
        return round(ndcg * 100, 2)

    mlb = MultiLabelBinarizer(classes=range(targets.shape[1]), sparse_output=True)
    mlb.fit(None)
    result = []
    for top_K in [1, 2, 3,4,5]:
        precision = get_precision(prediction, targets, mlb, top_K, args)
        result.append(precision)
    for top_K in [1, 2, 3,4,5]:
        ndcg = get_ndcg(prediction, targets, mlb, top_K, args)
        result.append(ndcg)

    return result

def evaluate_test(targets, prediction, mlb, args):

    def get_precision(prediction, targets, mlb, top_K):
        targets = sparse.csr_matrix(targets)
        prediction = mlb.transform(prediction[:, :top_K])
        precision = prediction.multiply(targets).sum() / (top_K * targets.shape[0])
        return round(precision * 100, 2)

    def get_ndcg(prediction, targets, mlb, top_K):
        log = 1.0 / np.log2(np.arange(top_K) + 2)
        dcg = np.zeros((targets.shape[0], 1))
        targets = sparse.csr_matrix(targets)
        for i in range(top_K):
            p = mlb.transform(prediction[:, i: i + 1])
            dcg += p.multiply(targets).sum(axis=-1) * log[i]
        ndcg = np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top_K) - 1])
        return round(ndcg * 100, 2)

    def get_inv_propensity(train_y, a=0.55, b=1.5):
        n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
        c = (np.log(n) - 1) * ((b + 1) ** a)
        return 1.0 + c * (number + b) ** (-a)

    def get_psp(prediction, targets, mlb, inv_w, top_K):
        if not isinstance(targets, csr_matrix):
            targets = mlb.transform(targets)
        prediction = mlb.transform(prediction[:, :top_K]).multiply(inv_w)
        num = prediction.multiply(targets).sum()
        t, den = csr_matrix(targets.multiply(inv_w)), 0
        for i in range(t.shape[0]):
            den += np.sum(np.sort(t.getrow(i).data)[-top_K:])
        return round(num / den * 100, 2)

    result = []
    for top_K in [1, 2, 3, 4, 5]:
        precision = get_precision(prediction, targets, mlb, top_K, args)
        result.append(precision)

    for top_K in [1, 2, 3, 4, 5]:
        ndcg = get_ndcg(prediction, targets, mlb, top_K, args)
        result.append(ndcg)

    if args.report_psp == True:
        a = 0.6
        b = 2.6
        train_labels = np.load(os.path.join(args.data_dir, args.train_labels), allow_pickle=True)
        inv_w = get_inv_propensity(mlb.transform(train_labels), a, b)
        for top_K in [1, 2, 3]:
            psp = get_psp(prediction, targets, mlb, inv_w, top_K, args)
            result.append(psp)
    return result

def evaluator(y_true, y_pred, top_K):
    precision_K = []
    for i in range(y_pred.shape[0]):
        if np.sum(y_true[i, :])==0:
            continue
        top_indices = y_pred[i].argsort()[-top_K:]
        p = np.sum(y_true[i, top_indices]) / top_K
        precision_K.append(p)
    precision = np.mean(np.array(precision_K))
    return precision
