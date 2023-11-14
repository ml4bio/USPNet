import sys

import  pandas as pd
import os
import torch as torch
import numpy as np
import argparse
from utils_tools.utils import *
import esm

data_dir = 'data_processed'
# Set up argument parser
parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--data_dir', nargs='?', default='data_processed', help='data directory')
parser.add_argument('--group_info', nargs='?', default='default', help='group information provided or not')


# Parse arguments
args = parser.parse_args()

dic = {'NO_SP': 0, 'SP': 1, 'LIPO': 2, 'TAT': 3, 'TATLIPO' : 4, 'PILIN' : 5}
dic2 = {0: 'NO_SP', 1: 'SP', 2: 'LIPO', 3: 'TAT', 4: 'TATLIPO', 5: 'PILIN'}
kingdom_dic = {'EUKARYA':0, 'ARCHAEA':1, 'POSITIVE':2, 'NEGATIVE': 3}

#Load ESM1b model
#esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#Load ESM2 model
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = (esm_model).cuda("cuda:0")
batch_converter = alphabet.get_batch_converter()
def trans_data_esm(str_array):

    # Process batches
    batch_labels, batch_strs, batch_tokens = batch_converter(str_array)
    batch_tokens = batch_tokens.cuda()

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, (_, seq) in enumerate(str_array):
        temp_tensor = token_representations[i, 1: len(seq) + 1]
        sequence_representations.append(temp_tensor.mean(0).detach().cpu().numpy())

    result = torch.tensor(np.array(sequence_representations))

    return result

def trans_data_esm_in_batches(str_array, split=100, path="./test_data/embedding/test_feature_esm.npy"):
    if(os.path.exists(path)):
        embedding_result = np.load(path)
        print("feature shape:")
        print(embedding_result.shape)
    else:
        divide_num = int(len(str_array)/split)
        results=[]

        for i in range(1, divide_num+1):
            print("process batch "+str(i)+":")
            results.append(trans_data_esm(str_array[(i-1)*split:i*split]))

        if (len(str_array) % split != 0):
            print("process batch " + str(1) + ":")
            results.append(trans_data_esm(str_array[divide_num * split:len(str_array)]))

        embedding_result = torch.cat(results).detach().cpu().numpy()
        print("feature shape:")
        print(embedding_result.shape)
        np.save(path, embedding_result)
    return embedding_result

def trans_data(str1, padding_length):
    # Translates amino acids into numbers
    a = []
    trans_dic = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':0}
    for i in range(len(str1)):
        if (str1[i] in trans_dic.keys()):
            a.append(trans_dic.get(str1[i]))
        else:
            print("Unknown letter:" + str(str1[i]))
            a.append(trans_dic.get('X'))
    while(len(a)<padding_length):
        a.append(0)

    return a

def trans_label(str1):
    # Translates labels into numbers
    if((str1) in dic.keys()):
        a = dic.get(str1)
    else:
        print(str1)
        raise Exception('Unknown category!')

    return a

def createTestData(data_path='./test_data/data_list.txt',
                    kingdom_path='./test_data/kingdom_list.txt',
                   maxlen=70, test_path="./embedding/test_feature_esm.npy"
                   ):
    # Initialize
    data_list = []
    kingdom_list=[]
    raw_data=[]
    # Load data
    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = np.array(trans_data(line.strip('\n')[0:70], maxlen))
            data_list.append(str)

    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = line.strip('\n\t')[0:70]
            raw_data.append(("protein", str))

    features = trans_data_esm_in_batches(raw_data, path=test_path)

    with open(kingdom_path, 'r') as kingdom_file:
        for line in kingdom_file:
            if args.group_info == 'no_group_info':
                kingdom_list.append([0, 0, 0, 0])
            else:
                kingdom_list.append(np.eye(len(kingdom_dic.keys()))[kingdom_dic[line.strip('\n\t')]])

    data_file.close()
    kingdom_file.close()

    X = np.array(data_list)
    kingdoms= np.array(kingdom_list)

    X = np.concatenate((X,kingdoms, features), axis=1)
    return X

def trans_output(str1):
    # Translates numbers into labels
    if((str1) in dic2.keys()):
        a = dic2.get(str1)
    else:
        print(str1)
        raise Exception('Unknown category!')

    return a

if __name__ == '__main__':
    device = torch.device("cuda:0")
    # read file names if provided
    if args.data_dir != 'data_processed':
        data_dir = args.data_dir

    if args.group_info == 'no_group_info':
        model = torch.load("../data/mdl/USPNet_fast_no_group_info.pth", map_location=device)
    else:
        model = torch.load("../data/mdl/USPNet_fast.pth", map_location=device)

    if isinstance(model, torch.nn.DataParallel):
        # access the model inside the DataParallel wrapper
        model = model.module
    model = model.to(device)
    model.eval()

    filename_list = ["data_list.txt",
                     "kingdom_list.txt",
                     "test_feature_esm.npy",
                     ]

    for i in range(len(filename_list)):
        filename_list[i] = os.path.join(data_dir, filename_list[i])

    X_test = createTestData(data_path=filename_list[0],
                            kingdom_path=filename_list[1],
                            test_path=filename_list[2])
    output = []
    output_aa = []
    aux_test = []

    X_test = torch.tensor(X_test)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=256)
    for i, input in enumerate(test_loader):
        input = input.cuda()
        aux = input[:, 70:74]
        aux = aux.cpu().detach().numpy()
        aux_test.extend(aux)
        o1, o_aa = model(input)
        output.extend(o1.cpu().detach().numpy())
        output_aa.extend(o_aa.cpu().detach().numpy())
    output = torch.tensor(np.array(output))
    results = pred(output).cpu().detach().numpy()
    # For Eurkaryota:
    for i in range(len(aux_test)):
        if (aux_test[i][0] == 1 and results[i] != 1 and results[i] != 0):
            results[i] = 0

    output_aa = torch.argmax(torch.tensor(np.array(output_aa)), dim=2).reshape(-1, 1)
    results_aa = output_aa.cpu().detach().numpy()
    output_aa_ = results_aa.reshape(-1, 70).copy()

    indexes_ = np.where(output_aa_ == 1)
    output_aa_[indexes_] = 100

    indexes_1 = np.where(output_aa_ == 3)
    indexes_2 = np.where(output_aa_ == 0)

    output_aa_[indexes_1] = 1
    output_aa_[indexes_2] = 1

    indexes_0 = np.where(output_aa_ != 1)
    output_aa_[indexes_0] = 0
    indexes_pos = np.where(output_aa_ == 1)

    predicted_type = []
    for result in results:
        predicted_type.append(trans_output(result))

    indexes1= indexes_pos[0].copy().tolist()
    indexes2 = indexes_pos[1].copy().tolist()


    predicted_cleavage = []
    count = 0
    data_list = []
    with open(filename_list[0], 'r') as data_file:
        for line in data_file:
            data_list.append(line.strip('\n'))
    data_file.close()
    for result in results:
        if result==0:
            predicted_cleavage.append('')
        else:
            try:
                index = indexes1.index(count)
            except:
                predicted_cleavage.append(data_list[count])
            else:
                index = indexes1.index(count)
                index2 = indexes2[index]
                sq=data_list[count]
                predicted_cleavage.append(sq[:index2+1])
        count = count + 1

    df = pd.DataFrame({'sequence': data_list, 'predicted_type': predicted_type, 'predicted_cleavage': predicted_cleavage})
    df.to_csv(data_dir + '/results.csv', index=False)