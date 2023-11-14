import torch as torch
import numpy as np
import argparse
import esm
from utils_tools.utils import *

# Set up argument parser
parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('group_info', nargs='?', default='default', help='group information provided or not')

# Parse arguments
args = parser.parse_args()

cls_names=['LIPO', 'NO_SP', 'SP', 'TAT', 'TATLIPO', 'PILIN']
metrics=['acc', 'F1_score', 'MCC']
metric_ad_aa = ['recall', 'precision', 'F1_score']
kingdom_dic = {'EUKARYA':0, 'ARCHAEA':1, 'POSITIVE':2, 'NEGATIVE': 3}
dic2 = {0: 'NO_SP', 1: 'SP', 2: 'LIPO', 3: 'TAT', 4: 'TATLIPO', 5: 'PILIN'}
# position specific class encoder
position_specific_classes_enc = preprocessing.LabelEncoder()
position_specific_classes_enc.fit(
    np.array(PositionSpecificLetter.values()).reshape((len(PositionSpecificLetter.values()), 1))
)

#Load ESM1b model
#esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#Load ESM2 model
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = (esm_model).cuda("cuda:0")
batch_converter = alphabet.get_batch_converter()

def relabel(y, label_test, keep, mode):
    y_ = y.tolist()
    label_ = label_test
    if(mode=="part"):
        new_y=[]
        new_label=[]
        for index, i in enumerate(label_):
            if i==0:
                new_label.append(i)
                new_y.append(y_[index])
        for index, i in enumerate(label_):
            if i==keep:
                new_label.append(i)
                new_y.append(y_[index])
        new_y=np.array(new_y)
        new_label=np.array(new_label)
        new_y[np.where(new_y != keep)] = 0
        new_y[np.where(new_y==keep)]=1
        new_label[np.where(new_label != keep)] = 0
        new_label[np.where(new_label == keep)] = 1


    elif (mode == "all"):
        new_y = np.array(y)
        new_label = np.array(label_test)
        new_y[np.where(new_y != keep)] = 0
        new_y[np.where(new_y==keep)]=1
        new_label[np.where(new_label != keep)] = 0
        new_label[np.where(new_label == keep)] = 1

    return new_y, new_label
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

def trans_data_msa_in_batches(str_array, split=100, path="./test_data/embedding/test_feature_esm.npy"):
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
def createTestData(data_path='./test_data/data_list.txt', label_path="./test_data/target_list.txt",
                    kingdom_path='./test_data/kingdom_list.txt', aa_path = "./test_data/aa_list.txt",
                   maxlen=70, test_path="./test_data/embedding/test_feature_esm.npy"
                   ):
    # Initialize
    data_list = []
    label_list = []
    kingdom_list=[]
    aa_list=[]
    raw_data=[]
    # Load data
    with open(data_path, 'r') as data_file:
        for line in data_file:
            data_list.append(np.array(trans_data(line.strip('\n'), maxlen)))

    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = line.strip('\n\t')
            raw_data.append(("protein", str))

    features = trans_data_msa_in_batches(raw_data, path=test_path)

    with open(label_path, 'r') as label_file:
        for line in label_file:
            label_list.append(trans_label(line.strip('\n')))

    with open(kingdom_path, 'r') as kingdom_file:
        for line in kingdom_file:
            if args.group_info == 'no_group_info':
                kingdom_list.append([0, 0, 0, 0])
            else:
                kingdom_list.append(np.eye(len(kingdom_dic.keys()))[kingdom_dic[line.strip('\n\t')]])

    count = 0
    with open(aa_path, 'r') as aa_file:
        for line in aa_file:
            aa_list.append(classes_sequence_from_ann_sequence(line.strip("\n\t"), position_specific_classes_enc))
            count+=1

    data_file.close()
    label_file.close()
    kingdom_file.close()
    aa_file.close()

    X = np.array(data_list)
    labels = np.array(label_list)
    kingdoms= np.array(kingdom_list)
    aas = np.array(aa_list)

    X = np.concatenate((X,kingdoms, features), axis=1)
    labels = labels.reshape(labels.shape[0], 1)
    labels = np.concatenate((labels, aas), axis=1)
    return X, labels

def evaluate(X, label, mode):
    test_dataset = SPDataset(X, label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)
    print("Total dataset evaluation:")

    output = []
    output_aa = []
    labels_test = []
    labels_test_aa = []

    for i, (input, target) in enumerate(test_loader):
        target_test = target[:, 0].reshape(target.shape[0])
        target_aa = target[:, 1:]
        input = input.cuda()
        target_test = target[:, 0].reshape(target.shape[0]).cuda()
        target_aa = target_aa.cuda()
        o1, o_aa= model(input)
        if mode == "best path":
            o_aa = np.array(model.crf.decode(o_aa.permute(1, 0, 2)))
        else:
            o_aa = model.crf.decode_based_on_prob(o_aa.permute(1, 0, 2), reduce=True)

        output.extend(o1.cpu().detach().numpy())
        output_aa.extend(o_aa)
        labels_test.extend(target_test.cpu().detach().numpy())
        labels_test_aa.extend(target_aa.cpu().detach().numpy())

    output = torch.tensor(np.array(output))
    output_aa = torch.tensor(np.array(output_aa)).reshape(-1, 1)
    labels_test = np.array(labels_test)
    labels_test_aa = np.array(labels_test_aa).reshape(-1, 1)

    return pred(output).cpu(), output_aa, labels_test, labels_test_aa

#cleavage site prediction test
def aaTest(output_aa_origin, labels_test_aa_origin, labels_test_origin, testType):
    if(testType == "SP"):
        tag=1
    elif(testType == "LIPO"):
        tag=2
    elif (testType == "TAT"):
        tag=3
    elif (testType == "TATLIPO"):
        tag=4

    labels_test_origin_torch = torch.Tensor(labels_test_origin)
    output_aa = output_aa_origin.reshape(-1, 70).clone()
    labels_test_aa = labels_test_aa_origin.reshape(-1, 70).copy()
    output_aa = output_aa[torch.where(labels_test_origin_torch==tag)].reshape(-1, 1)
    labels_test_aa = labels_test_aa[np.where(labels_test_origin==tag)].reshape(-1, 1)

    print("aa type:"+testType+":================")

    indexes_ = torch.where(output_aa == 1)
    output_aa[indexes_] = 100

    indexes_1 = torch.where(output_aa == 3)
    indexes_2 = torch.where(output_aa == 0)

    output_aa[indexes_1] = 1
    output_aa[indexes_2] = 1

    indexes_0 = torch.where(output_aa != 1)
    output_aa[indexes_0] = 0

    indexes_ = np.where(labels_test_aa == 1)
    labels_test_aa[indexes_] = 100

    indexes_1 = np.where(labels_test_aa == 3)
    indexes_2 = np.where(labels_test_aa == 0)

    labels_test_aa[indexes_1] = 1
    labels_test_aa[indexes_2] = 1

    indexes_0 = np.where(labels_test_aa != 1)
    labels_test_aa[indexes_0] = 0


    y_pred_aa = (output_aa).cpu()

    output_aa_ = output_aa.detach().cpu().numpy()
    indexes_pos = np.where(labels_test_aa == 1)
    p = np.sum(np.equal(output_aa_[indexes_pos], 1))
    s = indexes_pos[0].shape[0]
    print("cleavage site acc: " + str(p / s) + "=" + str(p) + "/" + str(s))

    indexes_neg = np.where(labels_test_aa == 0)
    p = np.sum(np.equal(output_aa_[indexes_neg], 0))
    s = indexes_neg[0].shape[0]
    print("non cleavage site acc: " + str(p / s) + "=" + str(p) + "/" + str(s))

    print("")
    for m in metric_ad_aa:
        result_ad = metric_advanced(m, y_pred_aa, labels_test_aa)

if __name__ == '__main__':

    # crf has two ways to predict: prob/best path
    device = torch.device("cuda:0")
    mode = "best path"

    if args.group_info == 'no_group_info':
        model = torch.load("../data/mdl/USPNet_fast_no_group_info.pth", map_location=device)
    else:
        model = torch.load("../data/mdl/USPNet_fast.pth", map_location=device)


    if isinstance(model, torch.nn.DataParallel):
        # access the model inside the DataParallel wrapper
        model = model.module
    model = model.to(device)
    model.eval()

    X_test, labels_test = createTestData(data_path='./test_data/data_list.txt',
                                         label_path="./test_data/target_list.txt",
                                         kingdom_path='./test_data/kingdom_list.txt',
                                         aa_path="./test_data/aa_list.txt",
                                         test_path="./test_data/embedding/test_feature_esm.npy")

    X_test_cls = {'EUKARYA':[],'ARCHAEA':[],'POSITIVE':[],'NEGATIVE':[] }
    labels_test_cls = {'EUKARYA': [], 'ARCHAEA': [], 'POSITIVE': [], 'NEGATIVE': []}

    target_files = {'EUKARYA': "./test_data/target_list_EUKARYA.txt",
                    'ARCHAEA': "./test_data/target_list_ARCHAEA.txt",
                    'POSITIVE': "./test_data/target_list_POSITIVE.txt",
                    'NEGATIVE': "./test_data/target_list_NEGATIVE.txt"}

    data_files = {'EUKARYA': "./test_data/data_list_EUKARYA.txt",
                  'ARCHAEA': "./test_data/data_list_ARCHAEA.txt",
                  'POSITIVE': "./test_data/data_list_POSITIVE.txt",
                  'NEGATIVE': "./test_data/data_list_NEGATIVE.txt"}

    kingdom_files = {'EUKARYA': "./test_data/kingdom_list_EUKARYA.txt",
                     'ARCHAEA': "./test_data/kingdom_list_ARCHAEA.txt",
                     'POSITIVE': "./test_data/kingdom_list_POSITIVE.txt",
                     'NEGATIVE': "./test_data/kingdom_list_NEGATIVE.txt"}

    aa_files = {'EUKARYA': "./test_data/aa_list_EUKARYA.txt",
                'ARCHAEA': "./test_data/aa_list_ARCHAEA.txt",
                'POSITIVE': "./test_data/aa_list_POSITIVE.txt",
                'NEGATIVE': "./test_data/aa_list_NEGATIVE.txt"}

    feature_files = {'EUKARYA': "./test_data/embedding/feature_EUKARYA_esm.npy",
                        'ARCHAEA': "./test_data/embedding/feature_ARCHAEA_esm.npy",
                        'POSITIVE': "./test_data/embedding/feature_POSITIVE_esm.npy",
                        'NEGATIVE': "./test_data/embedding/feature_NEGATIVE_esm.npy"}
    y_pred, output_aa, labels_test, labels_test_aa = evaluate(X_test, labels_test, mode)
    m = 'MCC'

    result_ad = metric_advanced(m, y_pred, labels_test)

    for key in X_test_cls.keys():
        m = "MCC"
        print(key+" MCC:")
        print()
        X_test_cls[key], labels_test_cls[key] = createTestData(data_files[key], target_files[key], kingdom_files[key],
                                                                   aa_files[key], test_path=feature_files[key])

        y_pred, output_aa, labels_test, labels_test_aa = evaluate(X_test_cls[key], labels_test_cls[key], mode)

        print()
        print("Test SP Type:")
        print("SP VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 1, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 1, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        print("LIPO VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 2, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 2, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        print("TAT VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 3, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 3, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        print("TATLIPO VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 4, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 4, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        #aaTest(output_aa, labels_test_aa, labels_test, "SP")
        #if (key != 'EUKARYA'):
        #    aaTest(output_aa, labels_test_aa, labels_test, "LIPO")
        #    aaTest(output_aa, labels_test_aa, labels_test, "TAT")
        #    aaTest(output_aa, labels_test_aa, labels_test, "TATLIPO")
