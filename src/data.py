import os
import glob
import numpy as np
import pandas as pd
import math 
from PIL import Image, ImageDraw 

def data_generator(root_dir):
    
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

def datasetDecomposition(input_path="../data/DX_TEST_RESULT_FULL.csv",
                         output_path="../data/ellipse",
                         shape=(1024,1024)):
    textData = pd.read_csv(input_path)
    users = sorted(textData[' user_name'].unique())
    
    data = {}
    for k in users:
        data[k]=[]

    cases_order = []    

    for case in textData.file_name.unique():
        for user in users:
            subsample = textData[(textData.file_name == case) & (textData[' user_name'] == user)]

            samples_list = []
            for i, row in subsample.iterrows():
                img = Image.new("RGB", shape)

                img1 = ImageDraw.Draw(img)  
                img1.ellipse([(row[' xcenter']-row[' rhorizontal'],
                               row[' ycenter']-row[' rvertical']),
                              (row[' xcenter']+row[' rhorizontal'],
                               row[' ycenter']+row[' rvertical'])], fill ="white")
                
                sample = np.asarray(img)[:,:,1]
                samples_list.append(sample.astype(bool))

            if len(subsample) == 0:
                samples_array = np.array([np.zeros(shape,dtype=bool)])
            else:
                samples_array = np.stack([sum(samples_list).astype(bool)]+ samples_list)

            data[user].append(samples_array)
        cases_order.append(case)
    
    os.mkdir(output_path)
    os.mkdir(output_path+'/after')
    os.mkdir(output_path+'/sample_1')
    os.mkdir(output_path+'/sample_2')
    os.mkdir(output_path+'/sample_3')
    
    for i in range(100):
        np.save(output_path+f'/after/{cases_order[i]}.npy',data['Expert'][i])
        np.save(output_path+f'/sample_1/{cases_order[i]}.npy',data['sample_1'][i])
        np.save(output_path+f'/sample_2/{cases_order[i]}.npy',data['sample_2'][i])
        np.save(output_path+f'/sample_3/{cases_order[i]}.npy',data['sample_3'][i])