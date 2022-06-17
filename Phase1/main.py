import sys
from utils import train_test

if __name__ == "__main__":

    data_dir = str(sys.argv[1])
    result_dir = str(sys.argv[2])

    lr_geAE = 1e-4
    n_epoch_geAE = 40
    lr_cnaAE = 1e-4
    n_epoch_cnaAE = 40
    lr_AE = 1e-6
    lr_clf = 1e-5
    patience = 20
    batch_size = 32
    wd_AE = 0.0

    if 'BRCA' in data_dir:
        wd_clf = 5e-1
    else:
        wd_clf = 1e-3

    train_test(data_dir, result_dir, batch_size, lr_geAE, lr_cnaAE, lr_AE, lr_clf, wd_AE, wd_clf, patience, n_epoch_geAE, n_epoch_cnaAE)