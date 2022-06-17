import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0.001, path='checkpoint.pt'):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        self.acc_max = 0.0
        self.delta = delta

    def __call__(self, acc, model):

        score = acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc, model)
            self.counter = 0

    def save_checkpoint(self, acc, model):
        
        self.best_weights = copy.deepcopy(model.state_dict())
        self.acc_max = acc

def train_clf(model, class_weight, train_loader, val_dataset, epoch, patience, lr_clf, lr_ae, wd_clf, wd_ae):

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    opt = optim.Adam([
                        {'params':model.classifier.parameters(), 'lr':lr_clf, 'weight_decay':wd_clf},
                        {'params':model.ge_repr.parameters()},
                        {'params':model.cna_repr.parameters()}
                     ], lr=lr_ae, weight_decay=wd_ae)
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    
    train_loss_his = []
    train_acc_his = []
    val_loss_his = []
    val_acc_his = []
    for ep in range(epoch):
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        nb = 0
        for x1b, x2b, yb in train_loader:
            x1b = x1b.cuda()
            x2b = x2b.cuda()
            yb = yb.cuda()

            preds = model(x1b, x2b)
            loss = loss_fn(preds, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

            _, preds_label = torch.max(preds.data, dim=-1)
            train_loss += loss.item()
            train_acc += (preds_label == yb.data).sum().item()/yb.size(0)
            nb += 1

        train_loss_his.append(train_loss/nb)
        train_acc_his.append(train_acc/nb)

        model.eval()
        with torch.no_grad():
            x1b, x2b, yb = val_dataset[:][0].cuda(), val_dataset[:][1].cuda(), val_dataset[:][-1].cuda()
            val_preds = model(x1b, x2b)
            val_loss = loss_fn(val_preds, yb)

            _, preds_label = torch.max(val_preds.data, dim=-1)
            val_acc = (preds_label == yb.data).sum().item()/yb.size(0)

        val_loss_his.append(val_loss)
        val_acc_his.append(val_acc)

        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_weights)

    return (train_loss_his, train_acc_his), (val_loss_his, val_acc_his)
    
def train_AE(model, train_loader, val_dataset, epoch, lr):

    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    train_loss_his = []
    val_loss_his = []
    for ep in range(epoch):
        model.train()
        train_loss = 0.0
        nb = 0
        for xb in train_loader:
            xb = xb[0].cuda()

            preds = model(xb)
            loss = loss_fn(preds, xb)

            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss += loss.item()
            nb += 1

        train_loss_his.append(train_loss/nb)

        model.eval()
        with torch.no_grad():
            xb = val_dataset[:][0].cuda()

            val_preds = model(xb)
            val_loss = loss_fn(val_preds, xb)


        val_loss_his.append(val_loss)
    return train_loss_his, val_loss_his
    