import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn import preprocessing, metrics
from models import *
from training import *

def prepare_data(data_dir, batch_size):

    unlabel_ge_data = 1
    unlabel_cna_data = 1
    try:
        train_ge_unlabeled = pd.read_csv(os.path.join(data_dir, 'train/train_val_split', 'df_train_GE_unlabeled.csv')).iloc[:, 1:].to_numpy()
    except:
        print('No unlabel GE data found!')
        unlabel_ge_data = 0
    try:
        train_cna_unlabeled = pd.read_csv(os.path.join(data_dir, 'train/train_val_split', 'df_train_CNA_unlabeled.csv')).iloc[:, 1:].to_numpy()
    except:
        print('No unlabel CNA data found!')
        unlabel_cna_data = 0

    train_cna = pd.read_csv(os.path.join(data_dir, 
                                         'train/train_val_split', 
                                         'df_train_CNA_labeled.csv')).iloc[:, 1:].to_numpy()
    val_cna = pd.read_csv(os.path.join(data_dir, 
                                       'train/train_val_split', 
                                       'df_val_CNA_labeled.csv')).iloc[:, 1:].to_numpy()
    test_cna = pd.read_csv(os.path.join(data_dir, 
                                        'test', 
                                        'df_test_CNA_labeled.csv')).iloc[:, 1:].to_numpy()
    train_ge = pd.read_csv(os.path.join(data_dir, 
                                        'train/train_val_split', 
                                        'df_train_GE_labeled.csv')).iloc[:, 1:].to_numpy()
    val_ge = pd.read_csv(os.path.join(data_dir, 
                                      'train/train_val_split', 
                                      'df_val_GE_labeled.csv')).iloc[:, 1:].to_numpy()
    test_ge = pd.read_csv(os.path.join(data_dir, 
                                       'test', 
                                       'df_test_GE_labeled.csv')).iloc[:, 1:].to_numpy()
    train_label = pd.read_csv(os.path.join(data_dir, 
                                           'train/train_val_split', 
                                           'df_label_train.csv')).iloc[:, 1].to_numpy()
    val_label = pd.read_csv(os.path.join(data_dir, 
                                         'train/train_val_split', 
                                         'df_label_val.csv')).iloc[:, 1].to_numpy()
    test_label = pd.read_csv(os.path.join(data_dir, 
                                          'test', 
                                          'df_label_test.csv')).iloc[:, 1].to_numpy()

    ord_enc = preprocessing.OrdinalEncoder(dtype='int64')
    ord_enc.fit(train_label.reshape(-1,1))
    y_train = torch.tensor(ord_enc.transform(train_label.reshape(-1,1))).squeeze()
    y_val = torch.tensor(ord_enc.transform(val_label.reshape(-1,1))).squeeze()
    y_test = torch.tensor(ord_enc.transform(test_label.reshape(-1,1))).squeeze()
    print('Classes: ', ord_enc.categories_[0])

    if len(ord_enc.categories_[0]) == 5: # for reproduce results purpose
        label_weight = torch.tensor([1,1,1,5,5], dtype=torch.float32)
    else:
        count_label = y_train.unique(return_counts=True)[1].float()
        if count_label.max()/count_label.min() >= 2:
            label_weight = count_label.sum()/count_label/5 # balanced
        else:
            label_weight = torch.ones_like(count_label)
    print('Weight for these classes:', label_weight)

    if unlabel_ge_data == 1:
        X_train_ge_AE = np.concatenate((train_ge, train_ge_unlabeled), axis=0)
    else:
        X_train_ge_AE = train_ge
    if unlabel_cna_data == 1:
        X_train_cna_AE = np.concatenate((train_cna, train_cna_unlabeled), axis=0)
    else:
        X_train_cna_AE = train_cna

    scaler_ge = preprocessing.StandardScaler().fit(X_train_ge_AE)
    scaler_cna = preprocessing.StandardScaler().fit(X_train_cna_AE)

    X_train_ge_AE = torch.tensor(scaler_ge.transform(X_train_ge_AE), dtype=torch.float32)
    X_train_ge_clf = torch.tensor(scaler_ge.transform(train_ge), dtype=torch.float32)
    X_val_ge = torch.tensor(scaler_ge.transform(val_ge), dtype=torch.float32)
    X_test_ge = torch.tensor(scaler_ge.transform(test_ge), dtype=torch.float32)
    X_train_cna_AE = torch.tensor(scaler_cna.transform(X_train_cna_AE), dtype=torch.float32)
    X_train_cna_clf = torch.tensor(scaler_cna.transform(train_cna), dtype=torch.float32)
    X_val_cna = torch.tensor(scaler_cna.transform(val_cna), dtype=torch.float32)
    X_test_cna = torch.tensor(scaler_cna.transform(test_cna), dtype=torch.float32)

    train_ge_AE_ds = TensorDataset(X_train_ge_AE)
    train_cna_AE_ds = TensorDataset(X_train_cna_AE)
    train_clf_ds = TensorDataset(X_train_ge_clf, X_train_cna_clf, y_train)
    val_ge_ds = TensorDataset(X_val_ge)
    val_cna_ds = TensorDataset(X_val_cna)
    val_clf_ds = TensorDataset(X_val_ge, X_val_cna, y_val)
    test_clf_ds = TensorDataset(X_test_ge, X_test_cna, y_test)

    if 'BRCA' in data_dir:
        batch_size_clf = batch_size*2
    elif 'CRC' in data_dir:
        batch_size_clf = batch_size = int(batch_size/2)
    else:
        batch_size_clf = batch_size
    train_clf_dl = DataLoader(train_clf_ds, batch_size=batch_size_clf, shuffle=True)
    train_ge_AE_dl = DataLoader(train_ge_AE_ds, batch_size=batch_size, shuffle=True)
    train_cna_AE_dl = DataLoader(train_cna_AE_ds, batch_size=batch_size, shuffle=True)

    return (train_clf_dl, train_ge_AE_dl, train_cna_AE_dl), (test_clf_ds, val_clf_ds, val_ge_ds, val_cna_ds), label_weight, ord_enc.categories_[0]


def evaluate(model, testdata, idx2class, result_dir):
    model.eval()
    with torch.no_grad():
        preds = model(testdata[:][0].cuda(), testdata[:][1].cuda())
        preds = F.softmax(preds, dim=1)
        _, preds_label = torch.max(preds.data, dim=-1)

    preds = preds.cpu()
    preds_label = preds_label.data.cpu()
    if len(idx2class) == 2:    
        print('\nTest AUC:\n', metrics.roc_auc_score(testdata[:][-1], preds.data[:,1]))
    clf_report = metrics.classification_report(testdata[:][-1], 
                                            preds_label, 
                                            target_names=idx2class, 
                                            digits=4, 
                                            zero_division=0, 
                                            output_dict=True)
    clf_df = pd.DataFrame(clf_report)
    clf_df.loc[['precision', 'recall'],'accuracy']=np.nan
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(13)
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(testdata[:][-1], preds_label), 
                                display_labels=idx2class).plot(cmap='Blues', ax=ax1)
    sns.heatmap(clf_df.iloc[:-1, :].T, annot=True, cmap='Blues', robust=True, ax=ax2, fmt='.2%')
    
    plt.savefig(os.path.join(result_dir, 'test_results.png'))

def train_test(data_dir, result_dir, batch_size, lr_geAE, lr_cnaAE, lr_AE, lr_clf, wd_AE, wd_clf, patience, n_epoch_geAE, n_epoch_cnaAE):

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    print('Loading data...')
    loader, dataset, label_weight, idx2class = prepare_data(data_dir, batch_size)
    train_clf_dl, train_ge_AE_dl, train_cna_AE_dl = loader
    test_clf_ds, val_clf_ds, val_ge_ds, val_cna_ds = dataset
    print('Create result directory...')
    try:
        os.makedirs(result_dir)
    except:
        print('Result directory already exist!')
    print('\nTraining DAE for GE data...\n')
    geAE = GEautoencoder(len(val_clf_ds[0][0]))
    geAE.to(device)
    train_his_ge, val_his_ge = train_AE(geAE, train_ge_AE_dl, val_ge_ds, n_epoch_geAE, lr_geAE)
    plt.plot(train_his_ge, label='train')
    plt.plot(val_his_ge, label='validation')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'ge_training.png'))
    plt.clf()
    print('\nTraining DAE for CNA data...\n')
    cnaAE = CNAautoencoder(len(val_clf_ds[0][1]))
    cnaAE.to(device)
    train_his_cna, val_his_cna = train_AE(cnaAE, train_cna_AE_dl, val_cna_ds, n_epoch_cnaAE, lr_cnaAE)
    plt.plot(train_his_cna, label='train')
    plt.plot(val_his_cna, label='validation')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'cna_training.png'))
    plt.clf()
    print('\nTraining fusion model...\n')
    clf = Subtyping_model(ge_encoder=geAE.encoder, 
                        cna_encoder=cnaAE.encoder,
                        subtypes=len(label_weight))
    clf.to(device)
    label_weight = label_weight.to(device)

    clf_train_his, clf_val_his = train_clf(clf, label_weight, 
                                        train_clf_dl, val_clf_ds, 
                                        200, patience, lr_clf, 
                                        lr_AE, wd_clf, wd_AE)
    plt.plot(clf_train_his[0], label='train loss')
    plt.plot(clf_val_his[0], label='validation loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'fusion_loss.png'))
    plt.clf()
    plt.plot(clf_train_his[1], label='train acc')
    plt.plot(clf_val_his[1], label='validation acc')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'fusion_acc.png'))
    plt.clf()
    evaluate(clf, test_clf_ds, idx2class, result_dir)
    # save model weights
    torch.save(clf.state_dict(), os.path.join(result_dir, 'checkpoint.pt'))
    print('Please check results in your result folder!')