from torch.utils.data import Dataset
import torch
import numpy as np
import math
import os
from sklearn import metrics
from enum import Enum
from sklearn import preprocessing
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

dic = {'NO_SP': 0, 'SP': 1, 'LIPO': 2, 'TAT': 3, 'TATLIPO' : 4, 'PILIN' : 5}

labels=[0, 1, 2, 3, 4, 5]
# position specific class encoder

def normalize(list):
    sum = 0
    for i in list:
        sum = sum+i*i
    sum = math.sqrt(sum)
    new_list = [i/sum for i in list]
    return new_list

class SPDataset(Dataset):
    """
        Complete data initialization, transformation, etc.
    """

    def __init__(self, X, Y ):
        self.x_data = torch.tensor(X)
        self.y_data = torch.tensor(Y).squeeze(1)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def accuracy(output ,target):
    y_true = target.numpy()
    y_pred = torch.argmax(output, axis=-1).numpy()
    acc = np.equal(y_true, y_pred)

    acc = float(np.sum(acc))/(float(y_true.shape[0]))
    return acc

def mcc(TP, TN, FP, FN):
    return ((TP*TN-TP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

def pred(output):
    return torch.argmax(output, axis=-1)


def metric(mode, output, labels, cls=None):
    #print(X)

    # measure metrics
    if mode=='acc':

        all_preds=np.argmax(output.detach().cpu().numpy(), 1)
        if ( cls!=None):
            index_pred = np.argwhere(np.array(all_preds) == dic[cls]).squeeze(1).tolist()
            index_true = np.argwhere(np.array(labels) == dic[cls]).squeeze(1).tolist()
            if(float(len(index_true))==0.0):
                acc = 0.0
            else:
                acc = float(len([i for i in index_pred if i in index_true]))/float(len(index_true))

        else:
            acc = float(np.sum(np.equal(all_preds, np.array(labels))))/float(len(labels))

        return acc

    elif mode=='F1_score':

        all_preds = np.argmax(output.detach().cpu().numpy(), 1)
        index_pred = np.argwhere(np.array(all_preds) == dic[cls]).squeeze(1).tolist()
        index_true = np.argwhere(np.array(labels) == dic[cls]).squeeze(1).tolist()
        # TP
        TP = float(len([i for i in index_pred if i in index_true]))
        # FP
        FP = float(len(index_pred))-TP
        FN = float(len(index_true))-TP

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2.0*precision*recall/(precision+recall)

        return F1_score

    else :
        # MCC score

        all_preds = np.argmax(output.detach().cpu().numpy(), 1)
        index_pred = np.argwhere(np.array(all_preds) == dic[cls]).squeeze(1).tolist()
        index_true = np.argwhere(np.array(labels) == dic[cls]).squeeze(1).tolist()

        index_pred_n = np.argwhere(np.array(all_preds) != dic[cls]).squeeze(1).tolist()
        index_true_n = np.argwhere(np.array(labels) != dic[cls]).squeeze(1).tolist()
        # TP
        TP = float(len([i for i in index_pred if i in index_true]))
        TN = float(len([i for i in index_pred_n if i in index_true_n]))
        # FP
        FP = float(len(index_pred)) - TP
        FN = float(len(index_true)) - TP

        MCC = mcc(TP, TN, FP, FN)

        return MCC

def metric_advanced(mode, y_pred, labels):

    # micro average/mactro average
    # options for state:
    #   macro, micro, weighted

    y_test = labels

    result=0.0
    print()
    if (mode=="F1_score"):
        result = metrics.f1_score(y_test, y_pred, labels=labels)

    elif (mode=="precision"):
        result = metrics.precision_score(y_test, y_pred, labels=labels)

    elif (mode=="recall"):
        result = metrics.recall_score(y_test, y_pred, labels=labels)

    elif (mode=="MCC"):
        result = metrics.matthews_corrcoef(y_test, y_pred)

    elif (mode=="AUC_ROC"):
        result = metrics.roc_auc_score(y_test, y_pred)

    elif (mode == "Kappa"):
        result = metrics.cohen_kappa_score(y_test, y_pred)

    elif (mode == "SN"):
        confusion = metrics.confusion_matrix(y_test, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        result=(TP / float(TP + FN))

    elif (mode == "SP"):
        confusion = metrics.confusion_matrix(y_test, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        result=(TN / float(TN + FP))

    elif (mode == "balanced_accuracy"):
        result = metrics.balanced_accuracy_score(y_test, y_pred)

    return result

class PositionSpecificLetter(Enum):
    """Position-specfic annotation of a letter (= amino acid).

    In addition to the general letter annotation, it also adds position-specific
    information based on the neighboring letters, e.g. whether a transmembrane amino acid
    is entering or leaving the membrane.
    """
    # 9
    SIGNAL_SEC_SP3 = "P"
    # 6
    SIGNAL_SEC_SP1 = "S"
    # 8
    SIGNAL_SEC_SP2 = "Z"
    # 7
    SIGNAL_TAT_SP1 = "T"
    # 0
    CLEAVAGE_SITE_SP1 = "C"
    # 3
    CLEAVAGE_SITE_SP2 = "K"
    # 5
    OUTER = "O"
    # 2
    INNER = "I"
    # 4
    TRANSMEMBRANE_IN_OUT = "L"
    # 1
    TRANSMEMBRANE_OUT_IN = "E"
    #10
    EMPTY = "z"

    @classmethod
    def values(cls):
        return [e.value for e in cls]

class PositionSpecificLetter_binary(Enum):
    """Position-specfic annotation of a letter (= amino acid).

    In addition to the general letter annotation, it also adds position-specific
    information based on the neighboring letters, e.g. whether a transmembrane amino acid
    is entering or leaving the membrane.
    """

    SIGNAL_SEC_SP1 = "N"
    SIGNAL_SEC_SP2 = "N"
    SIGNAL_TAT_SP1 = "N"
    CLEAVAGE_SITE_SP1 = "C"
    CLEAVAGE_SITE_SP2 = "C"
    OUTER = "N"
    INNER = "N"
    TRANSMEMBRANE_IN_OUT = "N"
    TRANSMEMBRANE_OUT_IN = "N"

    @classmethod
    def values(cls):
        return ["C", "N"]

class AnnotationLetter(Enum):
    INNER = "I"
    OUTER = "O"
    TRANSMEMBRANE = "M"
    # Includes cleavage site AA
    SIGNAL_SEC_SP1 = "S"
    # Includes cleavage site AA
    SIGNAL_SEC_SP2 = "L"
    # Includes cleavage site AA
    SIGNAL_TAT_SP1 = "T"
    SIGNAL_SEC_SP3 = "P"

def classes_sequence_from_ann_sequence(sequence, enc):
    #  print("SEQ", sequence)
    #  if sequence == "IIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMOOOOOMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIII":
    #  print("SEQ_!", sequence)

    classes_sequence = []
    prev_inner_outer = None
    for i in range(70):
        if (len(sequence) < i + 1):

            letter = None
        else:
            letter = sequence[i]
        if letter is None:

            position_specific_class = PositionSpecificLetter.EMPTY

            transformed = enc.transform([position_specific_class.value])
            classes_sequence.append(transformed)

            continue

        prev_letter = AnnotationLetter(sequence[i - 1]) if i > 0 else None
        next_letter = AnnotationLetter(sequence[i + 1]) if i + 1 < len(sequence) else None

        position_specific_class = None
        letter = AnnotationLetter(letter)

        if letter == AnnotationLetter.INNER:
            position_specific_class = PositionSpecificLetter.INNER
            prev_inner_outer = AnnotationLetter.INNER
        elif letter == AnnotationLetter.OUTER:
            position_specific_class = PositionSpecificLetter.OUTER
            prev_inner_outer = AnnotationLetter.OUTER
        elif letter == AnnotationLetter.TRANSMEMBRANE:
            if prev_letter == AnnotationLetter.TRANSMEMBRANE:
                if prev_inner_outer == AnnotationLetter.INNER:
                    position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_IN_OUT
                elif prev_inner_outer == AnnotationLetter.OUTER:
                    position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_OUT_IN
            elif (
                prev_letter == AnnotationLetter.INNER or prev_inner_outer == AnnotationLetter.INNER
            ):
                position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_IN_OUT
            elif (
                prev_letter == AnnotationLetter.OUTER or prev_inner_outer == AnnotationLetter.OUTER
            ):
                position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_OUT_IN
        elif letter == AnnotationLetter.SIGNAL_SEC_SP1:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP1:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP1
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1
        elif letter == AnnotationLetter.SIGNAL_SEC_SP2:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP2:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP2
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP2
        elif letter == AnnotationLetter.SIGNAL_TAT_SP1:

            if next_letter == AnnotationLetter.SIGNAL_TAT_SP1:
                position_specific_class = PositionSpecificLetter.SIGNAL_TAT_SP1
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1
        elif letter == AnnotationLetter.SIGNAL_SEC_SP3:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP3:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP3
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1

        if position_specific_class is None:
            print("Unexpected case", prev_letter, letter, next_letter)

        transformed = enc.transform([position_specific_class.value])
        classes_sequence.append(transformed)

    seq_tensor = np.array(classes_sequence).reshape((70))
    #  if sequence == "IIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMOOOOOMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIII":
    #  print("SE!", classes_sequence_to_letters(seq_tensor, enc))

    return seq_tensor

def classes_sequence_from_ann_sequence_binary(sequence, enc):
    #  print("SEQ", sequence)
    #  if sequence == "IIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMOOOOOMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIII":
    #  print("SEQ_!", sequence)

    classes_sequence = []
    prev_inner_outer = None
    for i in range(70):
        if(len(sequence)<i+1):

            letter = None
        else:
            letter = sequence[i]
        if letter is None:
            #  placeholder_encoding = [np.zeros(len(enc.classes_[0]))]
            placeholder_encoding = [0]
            classes_sequence.append(placeholder_encoding)
            continue

        prev_letter = AnnotationLetter(sequence[i - 1]) if i > 0 else None
        next_letter = AnnotationLetter(sequence[i + 1]) if i + 1 < len(sequence) else None

        position_specific_class = None
        letter = AnnotationLetter(letter)

        if letter == AnnotationLetter.INNER:
            position_specific_class = PositionSpecificLetter_binary.INNER
            prev_inner_outer = AnnotationLetter.INNER
        elif letter == AnnotationLetter.OUTER:
            position_specific_class = PositionSpecificLetter_binary.OUTER
            prev_inner_outer = AnnotationLetter.OUTER
        elif letter == AnnotationLetter.TRANSMEMBRANE:
            if prev_letter == AnnotationLetter.TRANSMEMBRANE:
                if prev_inner_outer == AnnotationLetter.INNER:
                    position_specific_class = PositionSpecificLetter_binary.TRANSMEMBRANE_IN_OUT
                elif prev_inner_outer == AnnotationLetter.OUTER:
                    position_specific_class = PositionSpecificLetter_binary.TRANSMEMBRANE_OUT_IN
            elif (
                prev_letter == AnnotationLetter.INNER or prev_inner_outer == AnnotationLetter.INNER
            ):
                position_specific_class = PositionSpecificLetter_binary.TRANSMEMBRANE_IN_OUT
            elif (
                prev_letter == AnnotationLetter.OUTER or prev_inner_outer == AnnotationLetter.OUTER
            ):
                position_specific_class = PositionSpecificLetter_binary.TRANSMEMBRANE_OUT_IN
        elif letter == AnnotationLetter.SIGNAL_SEC_SP1:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP1:
                position_specific_class = PositionSpecificLetter_binary.SIGNAL_SEC_SP1
            else:
                position_specific_class = PositionSpecificLetter_binary.CLEAVAGE_SITE_SP1
        elif letter == AnnotationLetter.SIGNAL_SEC_SP2:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP2:
                position_specific_class = PositionSpecificLetter_binary.SIGNAL_SEC_SP2
            else:
                position_specific_class = PositionSpecificLetter_binary.CLEAVAGE_SITE_SP2
        elif letter == AnnotationLetter.SIGNAL_TAT_SP1:
            if next_letter == AnnotationLetter.SIGNAL_TAT_SP1:

                position_specific_class = PositionSpecificLetter_binary.SIGNAL_TAT_SP1
            else:
                position_specific_class = PositionSpecificLetter_binary.CLEAVAGE_SITE_SP1

        if position_specific_class is None:
            print("Unexpected case", prev_letter, letter, next_letter)

        transformed = enc.transform([position_specific_class.value])
        classes_sequence.append(transformed)

    seq_tensor = np.array(classes_sequence).reshape((70))

    return seq_tensor

if __name__ == '__main__':
    position_specific_classes_enc = preprocessing.LabelEncoder()
    position_specific_classes_enc.fit(
        np.array(PositionSpecificLetter.values()).reshape((len(PositionSpecificLetter.values()), 1))
    )