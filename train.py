from tqdm import tqdm
from BCE_LOSS import bce_loss
from test import valid, test
import torch
from collections import deque
import collector
from contrastive import train_one_batch_contrastive
import numpy as np
import torch.nn as nn
import os
import time
import math
# from loss_methods import FocalLoss, LDAMLoss
from sample_methods import ClassBalanceSampler, InstanceBalanceSampler, InstanceBalanceSampler_word, \
    ClassBalanceSampler2
from sklearn.preprocessing import MultiLabelBinarizer


def train(model, optimizer, train_loader, val_loader, test_loader, mlb, args):
    state = {}
    global_step = 0
    num_stop_dropping = 0
    best_valid_result = 0
    step = 100
    gradient_norm_queue = deque([np.inf], maxlen=5)
    results_dir = " "
    os.makedirs(results_dir, exist_ok=True)
    result_file_name = ""
    filepath = os.path.join(results_dir, result_file_name)

    with open(filepath, 'w') as result_file:
        for epoch in range(args.epochs):
            if args.swa_mode == True and epoch == args.swa_warmup:
                swa_init(state, model)
            if args.contrastive_mode == True and epoch >= args.contrastive_warmup and epoch % 2 == 0:
                if args.pre_feature_dict == True:
                    feature_dict = np.load(args.pre_feature_dict_path, allow_pickle=True).item()
                else:
                    collector.collect(model, train_loader, args)
                    feature_dict = np.load(args.feature_dict_path, allow_pickle=True).item()
                prototype_queue = collector.get_queue(feature_dict, args)
            prototype_queue = np.load(' ',
                                      allow_pickle=True)
            print(args.contrastive_warmup)
            for i, batch in enumerate(train_loader, 1):
                global_step += 1
                if args.contrastive_mode == True and epoch >= args.contrastive_warmup and epoch % 2 == 0:
                    # batch_loss = train_one_batch_contrastive(batch, model, optimizer, gradient_norm_queue,
                    #                                          prototype_queue, args, epoch)
                    batch_loss = train_one_batch_contrastive(batch, model, optimizer, gradient_norm_queue,
                                                             prototype_queue, args)
                else:
                    batch_loss = train_one_batch(batch, model, optimizer, gradient_norm_queue, args)
                if global_step % step == 0:
                    if args.swa_mode == True:
                        swa_step(state, model)
                        swap_swa_params(state, model)
                    valid_result = valid(model, val_loader, mlb, args)[-1]
                    if valid_result > best_valid_result:
                        best_valid_result = valid_result
                        num_stop_dropping = 0
                        torch.save(model.state_dict(), args.check_pt_model_path)
                        print('best:', best_valid_result)
                    else:
                        num_stop_dropping += 1

                    if args.swa_mode == True:
                        swap_swa_params(state, model)

                    if args.test_each_epoch:
                        # train_result = valid(model, train_loader, mlb, args)
                        valid_result = valid(model, val_loader, mlb, args)
                        test_result = test(model, test_loader, mlb, args)
                        result_str = (f'Epoch: {epoch} | Loss: {batch_loss: .4f} | Stop: {num_stop_dropping} | '
                                      f' Valid: {valid_result} | Test: {test_result}')
                    else:
                        # train_result = valid(model, train_loader, mlb, args)
                        valid_result = valid(model, val_loader, mlb, args)
                        result_str = (
                            f'Epoch: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | '
                            f'Valid Result: {valid_result}')
                    result_file.write(result_str + '\n')
                    print(result_str)
            # train_result = valid(model, train_loader, mlb, args)
            # result_str = (
            #   f'Epoch: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | '
            #  f'**********train Result: {train_result}')
            # print(result_str)
            # result_file.write(result_str + '\n')

            if num_stop_dropping >= args.early_stop_tolerance:
                print('Have not increased for %d check points, early stop training' % num_stop_dropping)
                break


def train_one_batch(batch, model, optimizer, gradient_norm_queue, args):
    model.to(args.device)
    src, trg = batch
    input_id = src.to(args.device)
    trg = trg.to(args.device)
    optimizer.zero_grad()
    model.train()
    y_pred, _ = model(input_id)
    loss = bce_loss(y_pred, trg.float()).requires_grad_(True)
    loss.backward()
    clip_gradient(model, gradient_norm_queue, args)
    optimizer.step(closure=None)
    return loss.item()

def clip_gradient(model, gradient_norm_queue, args):
    if args.gradient_clip_value is not None:
        max_norm = max(gradient_norm_queue)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * args.gradient_clip_value)
        gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))



def swa_init(state, model):
    print('SWA Initializing')
    swa_state = state['swa'] = {'models_num': 1}
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()


def swa_step(state, model):
    if 'swa' in state:
        swa_state = state['swa']
        swa_state['models_num'] += 1
        beta = 1.0 / swa_state['models_num']
        with torch.no_grad():
            for n, p in model.named_parameters():
                swa_state[n].mul_(1.0 - beta).add_(beta, p.data)



def swap_swa_params(state, model):
    if 'swa' in state:
        swa_state = state['swa']
        for n, p in model.named_parameters():
            p.data, swa_state[n] = swa_state[n], p.data









