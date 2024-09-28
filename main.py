import webbrowser
import os
import logging
import argparse
import time
import random
#from prototype_get import prototype_get
import numpy as np
import dataset, train, test
import collector
# from collector import augment_tail_data
from utils import time_since, cprint
import torch
from torch.optim import Adam
from prototype_get import prototype_get
from TextRNN import TextRNN
from TextCNN import TextCNN
from bilstm import BiLSTM
from Transformer import Transformer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_function', type=str, default='BCEWithLogitsLoss',
                        choices=['BCEWithLogitsLoss', 'FocalLoss', 'ldam'],
                        help='Loss function to use')
    parser.add_argument('--focal_gamma', type=float, default=1.0,
                        help='Gamma value for Focal Loss')
    parser.add_argument('--m_ldam', type=float, default=0.5,
                        help='Margin scaling factor for LDAM Loss')
    parser.add_argument('--s_ldam', type=float, default=30,
                        help='Scaling factor for LDAM Loss')
    #dataset
    parser.add_argument("--data_dir", default="/home/admin1/LocalUsers/", type=str,
                        help="The input data directory")

    parser.add_argument("--train_texts", default="train_texts.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--train_labels", default="train_labels.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--test_texts", default="test_texts.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--test_labels", default="test_labels.npy", type=str,
                        help="data after preprocessing")
    # parser.add_argument("--valid_texts", default="valid_texts.npy", type=str,
    #                     help="data after preprocessing")
    # parser.add_argument("--valid_labels", default="valid_labels.npy", type=str,
    #                     help="data after preprocessing")
    # parser.add_argument("--vocab_path", default=None, type=str,
    #                     help="data before preprocessing")
    # parser.add_argument("--emb_init", default=None, type=str,
    #                     help="embedding layer from glove")
    parser.add_argument("--vocab_path", default="vocab.npy", type=str,
                        help="data before preprocessing")
    parser.add_argument("--emb_init", default="emb_init.npy", type=str,
                        help="embedding layer from glove")
    parser.add_argument("--labels_binarizer", default="labels_binarizer", type=str,
                        help="")
    parser.add_argument('--max_len', type=int, default=500,
                        help="max length of document")
    parser.add_argument('--vocab_size', type=int, default=500000,
                        help="vocabulary size of dataset")
    # parser.add_argument('--valid_size', type=int, default=5000,
    #                     help="size of validation set")

    parser.add_argument("--dropout", default=0.5, required=False, type=float,
                        help="dropout of LSFL")
    parser.add_argument("--learning_rate", default=1e-3, required=False, type=float,
                        help="learning rate of LSFL")

    #training
    parser.add_argument('--gpuid', type=int, default=0,
                        help="gpu id")
    parser.add_argument('--epochs', type=int, default=50,
                        help="epoch of LSFL")
    parser.add_argument('--early_stop_tolerance', type=int, default=15,
                        help="early stop of LSFL")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size of LSFL")
    parser.add_argument('--swa_warmup', type=int, default=5,
                        help="begin epoch of swa")
    parser.add_argument('--swa_mode', type=bool, default=False,
                        help="use swa strategy")
    parser.add_argument('--gradient_clip_value', type=int, default=5.0,
                        help="gradient clip")
    parser.add_argument('--seed', type=int, default=100,
                        help="random seed for initialization")
    parser.add_argument('--test_each_epoch', type=bool, default=True,#True False
                        help="test performance on each epoch")
    parser.add_argument('--report_psp', type=bool, default=True,
                        help="report psp metric")
    parser.add_argument('--save_prediction', type=bool, default=True,
                        help="save prediction for plot")
    parser.add_argument('--sample_level_da', type=bool, default=False,
                        help="try sample level da")

    #sampling
    parser.add_argument('--rebalance', type=str, default='ldam', help='sampling methods')
    parser.add_argument('--beta_rs', type=float, default=0.99, help='beta_cs')
    parser.add_argument('--epoch_drw', type=float, default=55, help='drw_epoch')
    parser.add_argument('--epoch_crt', type=int, default=10, help='crt_epoch')
    parser.add_argument('--result_path', type=str, default='results/', help='result path')
    parser.add_argument('--num_cls', type=int, default=7, help='num_classes')

    #contrastive
    parser.add_argument('--contrastive_mode', type=bool, default=False,
                        help="use contrastive.py learning")
    parser.add_argument('--contrastive_warmup', type=int, default=4,
                        help="begin epoch of contrastive.py learning")
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help="weight of contrastive.py learning")
    parser.add_argument('--initial_contrastive_weight', type=float, default=0.2,
                        help="initial weight of contrastive.py learning")
    parser.add_argument('--contrastive_batch_size', type=int, default=64,
                        help="batch size of contrastive.py learning")
    parser.add_argument('--T', type=float, default=0.07,
                        help="temperature of contrastive.py learning")

    parser.add_argument('--pretrained', type=bool, default=False,
                        help="use pretrained model")
    parser.add_argument("--pretrained_path", default='', type=str,
                        help="path of pretrained model")
    parser.add_argument('--pre_feature_dict', type=bool, default=False,
                        help="use former feature dictionary")
    parser.add_argument("--pre_feature_dict_path", default='', type=str,
                        help="path of former feature dictionary")

    args = parser.parse_args()
    args.model_path = os.path.join(args.data_dir, 'model')
    os.makedirs(args.model_path, exist_ok=True)
    args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    args.check_pt_model_path = os.path.join(args.model_path, "lstm_%s.pth" % args.timemark)
    args.feature_dict_path = os.path.join(args.model_path, "feature_dict_%s.npy" % args.timemark)
    args.da_dict_path = os.path.join(args.model_path, "da_dict_%s.npy" % args.timemark)
    args.prediction_path = os.path.join(args.model_path, "prediction_%s.npy" % args.timemark)
    cprint('args', args)

    #for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    da4mltc(args)

def da4mltc(args):

    #Dataset
    start_time = time.time()
    logger.info('Data Loading')
    train_loader, val_loader, test_loader, emb_init, mlb, args = dataset.get_data(args)
    load_data_time = time_since(start_time)
    logger.info('Time for loading the data: %.1f s' %load_data_time)

    #Model
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    args.device = torch.device('cuda:0')
    model =BiLSTM(emb_init)
    # model = TextRNN_Att(num_classes=7,embeddings=emb_init)
    # model = Transformer(emb_init,num_classes=7)
    # model = TextCNN(num_classes=7, embeddings=emb_init)
    model = model.to(args.device)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    #prototype_get(model,train_loader,args)

    print(f'epochs= ',args.epochs)
    print(f'lr =',args.learning_rate)
    print(f'valid_size= ',args.valid_size)
    train_size = len(train_loader.dataset)
    print(f"Training dataset size: {train_size}")

    #Training
    if args.pretrained==False:
        train.train(model, optimizer, train_loader, val_loader, test_loader, mlb, args)
        training_time = time_since(start_time)
        logger.info('Time for training: %.1f s' % training_time)
        logger.info(f'Best Model Path: {args.check_pt_model_path}')
    else:
        args.check_pt_model_path = args.pretrained_path
    model.load_state_dict(torch.load(args.check_pt_model_path, map_location=args.device))

    #Collecting
    if args.pre_feature_dict == False:
        logger.info('Collecting')
        start_time = time.time()
        feature_dict = collector.collect(model, train_loader, args)
        logger.info(f'Collected Feature Dictionary Path: {args.feature_dict_path}')
        logger.info('Time for Collecting: %.1f s' % time_since(start_time))
    else:
        feature_dict = np.load(args.pre_feature_dict_path, allow_pickle=True).item()
    # prototype_dict = collector.get_prototype(feature_dict)
    prototype_get(model, train_loader, args) #
    head_list, tail_list = collector.get_head(feature_dict, args)
    print('head',head_list)
    print('tail',tail_list)


if __name__ == '__main__':
    main()

