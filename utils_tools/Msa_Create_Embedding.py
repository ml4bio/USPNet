from __future__ import print_function

import torch
import os
from Bio import SeqIO
import esm
from typing import List, Tuple
import string
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

#Delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_sequence(filename: str) -> Tuple[str, str]:
    #Read the first (reference) sequences from a fasta or MSA file
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


def remove_insertions(sequence: str) -> str:
    #Removes any insertions into the sequence. Needed to load aligned sequences in an MSA.
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    #Reads sequences from an MSA file, removes insertions.
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


# Select 128 sequences from the MSA to maximize the hamming distance
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

# Load MSA model
msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval()
msa_transformer = msa_transformer.cuda(0)
msa_batch_converter = msa_alphabet.get_batch_converter()

NB_seqs_per_msa = 128

def createDatasetEmbedding(data_path, save_path):
    MSAs = os.listdir(data_path)
    MSAs.sort(key=lambda x:int(x[:-4]))
    print(len(MSAs))
    embeddings = []
    for i in tqdm(range(len(MSAs))):
        msa_data = read_msa(data_path + MSAs[i])
        msa_data = greedy_select(msa_data, num_seqs=NB_seqs_per_msa)
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter([msa_data])
        torch.cuda.empty_cache()
        msa_batch_tokens = msa_batch_tokens.cuda(0)
        with torch.no_grad():
            results = msa_transformer(msa_batch_tokens, repr_layers=[12], return_contacts=True)

        embeddings.append(results["representations"][12][:, 0, 1:, :].clone().mean(1).detach().cpu().numpy())

    embedding_result = np.squeeze(embeddings, axis=1)
    print("MSA feature shape:")
    print(np.shape(embedding_result))

    np.save(save_path, embedding_result)
    return embedding_result


if __name__ == '__main__':
    #Input: MSA files, Output: MSA embedding
    #It's recommended to generate MSA files from protein sequences that are limited to a maximum of 70 residues.
    #MSA files should be names as 1.a3m, 2.a3m, 3.a3m, ..., in numerical order
    createDatasetEmbedding('../a3m_files/', "../msa_feature.npy")