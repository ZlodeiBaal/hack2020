import os
import glob
import numpy as np
import pandas as pd


def prepare_submission(y_pred, root_dir):
    y_df = pd.read_csv(os.path.join(root_dir, "OpenPart.csv"))

    for i in range(1, 4):
        X = [np.load(_) for _ in sorted(glob.glob(os.path.join(root_dir, "sample_" + str(i) + "/*")))]
        X_ids = [_.split("/")[-1] for _ in sorted(glob.glob(os.path.join(root_dir, "sample_" + str(i) + "/*")))]

        y_ids = y_df.sort_values(by="Case")["Case"].values

        X_not_ids = [X_ids[i] for i in range(len(X)) if X_ids[i].split(".")[0] + ".png" not in y_ids]

    secretPartNames = [_.replace(".npy", ".png") for _ in X_not_ids]
    
    SecretPart_427 = pd.DataFrame()

    SecretPart_427["Case"] = secretPartNames
    SecretPart_427["Sample 1"] = y_pred[:40]
    SecretPart_427["Sample 2"] = y_pred[40:80]
    SecretPart_427["Sample 3"] = y_pred[80:]
    
    return SecretPart_427


def data_generator_train(root_dir):
    
    y_df = pd.read_csv(os.path.join(root_dir, "OpenPart.csv"))
    
    full_X = []
    full_Xy = []
    full_y = []

    for i in range(1, 4):
        # X = np.array([plt.imread(_) for _ in sorted(glob.glob("../data/sample_1/*"))]).astype(bool)
        X = [np.load(_) for _ in sorted(glob.glob(os.path.join(root_dir, "sample_" + str(i) + "/*")))]
        # X_ids = [_.split("/")[-1] for _ in sorted(glob.glob("../data/sample_1/*"))]
        X_ids = [_.split("/")[-1] for _ in sorted(glob.glob(os.path.join(root_dir, "sample_" + str(i) + "/*")))]
        # Xy = np.array([plt.imread(_) for _ in glob.glob("../data/after/*")]).astype(bool)
        Xy = [np.load(_) for _ in sorted(glob.glob(os.path.join(root_dir, "after/*")))]
        # Xy_ids = [_.split("/")[-1] for _ in sorted(glob.glob("../data/after/*"))]
        Xy_ids = [_.split("/")[-1] for _ in sorted(glob.glob(os.path.join(root_dir, "after/*")))]

        y = y_df.sort_values(by="Case")["Sample " + str(i)].values
        y_ids = y_df.sort_values(by="Case")["Case"].values

        X_not_ids = [X_ids[i] for i in range(len(X)) if X_ids[i].split(".")[0] + ".png" not in y_ids]

        X = np.array([X[i] for i in range(len(X)) if X_ids[i].split(".")[0] + ".png" in y_ids])
        Xy = np.array([Xy[i] for i in range(len(Xy)) if Xy_ids[i].split(".")[0] + ".png" in y_ids])

        full_X = list(full_X) + list(X)
        full_Xy = list(full_Xy) + list(Xy)
        full_y = list(full_y) + list(y)

    X = np.array(full_X)
    Xy = np.array(full_Xy)
    y = np.array(full_y)
    
    return X, Xy, y


def data_generator_test(root_dir):
    
    y_df = pd.read_csv(os.path.join(root_dir, "OpenPart.csv"))
    
    full_X = []
    full_Xy = []
    full_y = []

    for i in range(1, 4):
        # X = np.array([plt.imread(_) for _ in sorted(glob.glob("../data/sample_1/*"))]).astype(bool)
        X = [np.load(_) for _ in sorted(glob.glob(os.path.join(root_dir, "sample_" + str(i) + "/*")))]
        # X_ids = [_.split("/")[-1] for _ in sorted(glob.glob("../data/sample_1/*"))]
        X_ids = [_.split("/")[-1] for _ in sorted(glob.glob(os.path.join(root_dir, "sample_" + str(i) + "/*")))]
        # Xy = np.array([plt.imread(_) for _ in glob.glob("../data/after/*")]).astype(bool)
        Xy = [np.load(_) for _ in sorted(glob.glob(os.path.join(root_dir, "after/*")))]
        # Xy_ids = [_.split("/")[-1] for _ in sorted(glob.glob("../data/after/*"))]
        Xy_ids = [_.split("/")[-1] for _ in sorted(glob.glob(os.path.join(root_dir, "after/*")))]

        y = y_df.sort_values(by="Case")["Sample " + str(i)].values
        y_ids = y_df.sort_values(by="Case")["Case"].values

#         X_not_ids = [X_ids[i] for i in range(len(X)) if X_ids[i].split(".")[0] + ".png" not in y_ids]

        X = np.array([X[i] for i in range(len(X)) if X_ids[i].split(".")[0] + ".png" not in y_ids])
        Xy = np.array([Xy[i] for i in range(len(Xy)) if Xy_ids[i].split(".")[0] + ".png" not in y_ids])

        full_X = list(full_X) + list(X)
        full_Xy = list(full_Xy) + list(Xy)
        full_y = list(full_y) + list(y)

    X = np.array(full_X)
    Xy = np.array(full_Xy)
#     y = np.array(full_y)
    
    return X, Xy