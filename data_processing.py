# convert fasta files into usable dataset file
import os
import argparse
import Bio.SeqIO as SeqIO
from utils_tools.Msa_Create_Embedding import *

def main(args):
    filename_list = ['test_set.fasta', 'target_list.txt', 'data_list.txt', 'kingdom_list.txt', 'aa_list.txt']

    data_dir = args.data_processed_dir
    msa_dir = args.msa_dir

    data_list = []
    aa_list = []
    target_list = []
    kingdom_list = []

    for record in SeqIO.parse(args.fasta_file, "fasta"):
        feature_parts = record.id.split("|")
        if args.fasta_file == 'test_set.fasta':
            half_len = int(len(record) / 2)
            sequence = str(record.seq[:half_len])
            ann_sequence = str(record.seq[half_len: int(len(record))])
            (uniprot_id, kingdom, sp_type) = feature_parts
            data_list.append((sequence))
            aa_list.append((ann_sequence))
            kingdom_list.append(kingdom)
            target_list.append(sp_type)
        else:
            sequence = str(record.seq)
            data_list.append((sequence))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in filename_list[1:]:
        if os.path.exists(data_dir + '/' + file):
            os.remove(data_dir + '/' + file)

    # The following saving steps can be encapsulated into functions if you like, to make it more modular
    if len(target_list) != 0:
        with open(data_dir + '/' + filename_list[1], 'w') as outf1:
            for target in target_list:
                outf1.write(target)
                outf1.write('\n')

    if len(data_list) != 0:
        with open(data_dir + '/' + filename_list[2], 'w') as outf2:
            for data in data_list:
                outf2.write(data)
                outf2.write('\n')

    if len(kingdom_list) != 0:
        with open(data_dir + '/' + filename_list[3], 'w') as outf3:
            for kingdom in kingdom_list:
                outf3.write(kingdom)
                outf3.write('\n')
    else:
        with open(data_dir + '/' + filename_list[3], 'w') as outf3:
            for data in data_list:
                outf3.write('\n')

    if len(aa_list) != 0:
        with open(data_dir + '/' + filename_list[4], 'w') as outf4:
            for aa in aa_list:
                outf4.write(aa)
                outf4.write('\n')

    if msa_dir is not None:
        createDatasetEmbedding(msa_dir + '/', data_dir + '/' + 'test_feature.npy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert fasta files into usable dataset files')
    parser.add_argument('--fasta_file', type=str, default='test_set.fasta', help='Path to the fasta file to be processed')
    parser.add_argument('--data_processed_dir', type=str, default='data_processed',
                        help='Directory where the processed data will be saved')
    parser.add_argument('--msa_dir', type=str, help='Directory for storing MSA files (.a3m). MSA files in the directory should be named in numerical order: 1.a3m, 2.a3m, 3.a3m, ...')

    args = parser.parse_args()
    main(args)