from cProfile import label
from random import shuffle
from re import T
from statistics import mode
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model.network import SAGNetworkHierarchical,SAGNetworkGlobal
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from tkinter import _flatten
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import argparse
import warnings
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
import os

def create_logger(branch_name):
    logger = logging.getLogger(branch_name)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join('log',branch_name+'.log'))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class EarlyStopping:
    """
    Dừng training sớm khi metric không cải thiện sau `patience` lần validation.
    
    Args:
        patience (int): Số lần validation cho phép không cải thiện trước khi dừng.
        min_delta (float): Mức cải thiện tối thiểu được coi là có ý nghĩa.
        monitor (str): Metric theo dõi, 'fscore' hoặc 'aupr'.
    """
    def __init__(self, patience=3, min_delta=1e-4, monitor='fscore'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_value = 0.0
        self.should_stop = False

    def step(self, current_value):
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


warnings.filterwarnings('ignore')
Thresholds = [x/100 for x in range(1,100)]

if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64,  help="the number of the bach size")
    parser.add_argument('-learningrate', '--learningrate',type=float,default=1e-4)
    parser.add_argument('-dropout', '--dropout',type=float,default=0.3)
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=328)
    parser.add_argument('-gpu', '--gpu',type=str,default='0')
    parser.add_argument('-exp_name', '--exp_name',type=str,default='')
    # Early stopping args
    parser.add_argument('-patience', '--patience', type=int, default=3,
                        help="Số lần validation không cải thiện trước khi dừng (0 = tắt early stopping)")
    parser.add_argument('-min_delta', '--min_delta', type=float, default=1e-4,
                        help="Mức cải thiện tối thiểu được tính là có ý nghĩa")
    parser.add_argument('-es_monitor', '--es_monitor', type=str, default='fscore',
                        choices=['fscore', 'aupr'],
                        help="Metric dùng để theo dõi early stopping")

    args = parser.parse_args()
    device = "cuda:" + args.gpu if torch.cuda.is_available() else "cpu"

    train_data_path = 'divided_data/'+args.branch+'_train_dataset'
    valid_data_path = 'divided_data/'+args.branch+'_valid_dataset'
    label_network_path = 'processed_data/label_'+args.branch+'_network'

    exp_name = args.branch + ("_" + args.exp_name if args.exp_name else "")
    logger = create_logger(exp_name)

    with open(train_data_path,'rb')as f:
        train_dataset = pickle.load(f)
    with open(valid_data_path,'rb')as f:
        valid_dataset = pickle.load(f)
    with open(label_network_path,'rb')as f:
        label_network=pickle.load(f)
    label_network = label_network.to(device)

    epoch_num = 50
    batch_size = args.batch_size
    learningrate = args.learningrate
    dropout = args.dropout
    labels_num = args.labels_num

    train_dataloader = GraphDataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    valid_dataloader = GraphDataLoader(dataset=valid_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    model = SAGNetworkHierarchical(154, 512, labels_num, num_convs=6, pool_ratio=0.75, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epoch_num*len(train_dataloader))
    criterion = nn.BCEWithLogitsLoss()

    best_fscore = 0
    best_aupr = 0
    best_scores = []
    best_score_dict = {}

    # Khởi tạo early stopping (patience=0 nghĩa là tắt)
    early_stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.es_monitor
    ) if args.patience > 0 else None

    logger.info('#########'+args.branch+'###########')
    logger.info('########start training###########')
    if early_stopper:
        logger.info(f'Early stopping: patience={args.patience}, min_delta={args.min_delta}, monitor={args.es_monitor}')

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        logger.info("epoch: "+str(epoch))
        model.train()
        train_loss = 0
        print("training")
        logger.info("training")
        for i,(pids, graphs, labels, seq_feats) in tqdm(enumerate(train_dataloader)):
            graphs = graphs.to(device)
            seq_feats = seq_feats.to(device)
            labels = labels.to(device).float()
            labels = torch.squeeze(labels)
            if len(labels.shape)==1:
                labels = labels.unsqueeze(0)

            optimizer.zero_grad()
            logits = model(graphs,seq_feats,label_network)
            loss = criterion(logits,labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
            if i%30 == 29:
                logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_dataloader)}, Loss(batch): {loss.item()}')

        # Validation mỗi 4 epoch
        if epoch%4==3:
            model.eval()
            print("validating")
            logger.info("validating")
            valid_loss = 0
            pred = []
            actual = []
            with torch.no_grad():
                for i,(pids, graphs, labels, seq_feats) in tqdm(enumerate(valid_dataloader)):
                    graphs = graphs.to(device)
                    seq_feats = seq_feats.to(device)
                    labels = labels.to(device)
                    labels = torch.squeeze(labels)
                    if len(labels.shape)==1:
                        labels = labels.unsqueeze(0)

                    logits = model(graphs,seq_feats,label_network)
                    loss = criterion(logits,labels)

                    valid_loss += loss.item()
                    pred += logits.tolist()
                    actual += labels.tolist()
                    if i%10 == 9:
                        logger.info(f'Valid Step: {i} / {len(valid_dataloader)}, Loss(batch): {loss.item()}')

            fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
            auc_score = auc(fpr, tpr)
            aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
            score_dict = {}
            each_best_fcore = 0
            each_best_scores = []
            for thresh in tqdm(Thresholds):
                f_score, precision, recall = calculate_performance(actual, pred, label_network, threshold=thresh)
                if f_score >= each_best_fcore:
                    each_best_fcore = f_score
                    each_best_scores = [thresh, f_score, recall, precision, auc_score]
                    scores = [f_score, recall, precision, auc_score]
                    score_dict[thresh] = scores

            if each_best_fcore >= best_fscore:
                best_fscore = each_best_fcore
                best_scores = each_best_scores
                best_score_dict = score_dict
                best_aupr = aupr
                torch.save(model, 'save_models/bestmodel_{}_{}_{}_{}_{}.pkl'.format(exp_name,batch_size,learningrate,dropout,args.gpu))

            thresh, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
            precision, auc_score = each_best_scores[3], each_best_scores[4]
            logger.info('########valid metric###########')
            logger.info('epoch{}, train_loss{}, valid_loss:{}'.format(epoch, train_loss/len(train_dataloader), valid_loss/len(valid_dataloader)))
            logger.info('threshold:{}, f_score{}, auc{}, recall{}, precision{}, aupr{}'.format(thresh, f_score, auc_score, recall, precision, aupr))

            # Kiểm tra early stopping
            if early_stopper:
                monitor_value = each_best_fcore if args.es_monitor == 'fscore' else aupr
                early_stopper.step(monitor_value)
                logger.info(f'Early stopping counter: {early_stopper.counter}/{early_stopper.patience} (best {args.es_monitor}: {early_stopper.best_value:.4f})')
                if early_stopper.should_stop:
                    logger.info(f'Early stopping triggered at epoch {epoch}. No improvement in {args.es_monitor} for {args.patience} validations.')
                    print(f'Early stopping triggered at epoch {epoch}.')
                    break

    logger.info('best_fscore: '+str(best_fscore))
    logger.info('best_scores[thresh,fmax,recall,precision,auc]: '+str(best_scores))
    logger.info('best_aupr: '+str(best_aupr))
    logger.info('best_score_dict: '+str(best_score_dict))