import argparse
from tqdm import tqdm
import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from DataSchiz import data

def get_stats(X):

    return [
        np.mean(X), 
        np.std(X),
        np.argmin(X) / len(X), 
        np.argmax(X) / len(X)
    ]

def get_columns(prefix):
    return [
        f'{prefix}_{func}' for func in
        ['mean', 'std', 'min_position', 'max_position']
    ]

def calculate_features(idx, folder_path):
    
    file_path = os.path.join(folder_path, f'{idx}.json')

    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

        # Get the ranks of the real word of the real text
        real_word_ranks = [
            d['real_word_rank']
            for d in json_data['data']
        ]

        # Get the BERT probability of each word of the original text
        # i.e. How probable is each word of the text to be part of the text
        real_word_probs = [
            d['real_word_probability']
            for d in json_data['data']
        ]
        
        # Get the BERT top N probabilities for each position in the original text
        # i.e. How probable are each of the N top most probable words to appear in each 
        # position of the text if the rest of the text is the original one (mask model)
        # This data is N x |text|, then I calculate the min, max, mean, std to get |text| agregated points 
        bert_word_probs = np.array([
            bert_word['probability']
            for d in json_data['data']
            for bert_word in d['bert_words']
        ])

        n_cpu = max(1, cpu_count()-2)
        pool_max = Pool(n_cpu)
        bert_word_probs_max = list(pool_max.map(np.max, bert_word_probs))

        pool_min = Pool(n_cpu)
        bert_word_probs_min = list(pool_min.map(np.min, bert_word_probs))

        pool_mean = Pool(n_cpu)
        bert_word_probs_mean = list(pool_mean.map(np.mean, bert_word_probs))

        pool_std = Pool(n_cpu)
        bert_word_probs_std = list(pool_std.map(np.std, bert_word_probs))

    return [
        *get_stats(real_word_ranks),
        *get_stats(real_word_probs),
        *get_stats(bert_word_probs_max),
        *get_stats(bert_word_probs_min),
        *get_stats(bert_word_probs_mean),
        *get_stats(bert_word_probs_std)
    ]

def load_data_and_feature_extraction(data_path):
    class_labels = np.array(['control', 'schizophrenics'])
    y = (data.data['experimental group'].values == class_labels[1]).astype(int)
    
    result = {
        'y': y,
        'class_labels': class_labels,
        'features': [
            *get_columns('real_word_ranks'),
            *get_columns('real_word_prob'),
            *get_columns('bert_word_prob_max'),
            *get_columns('bert_word_prob_min'),
            *get_columns('bert_word_prob_mean'),
            *get_columns('bert_word_prob_std')
        ]
    }

    languages = ['single_language', 'multilingual']
    contexts_sizes = [5,11,17]
    masked_idxs = ['middle', 'last']
    indexes = data.data.index.tolist()

    for bert_lang in tqdm(languages, desc='Bert lang', leave=False):
        for context_size in tqdm(contexts_sizes, desc='Context size', leave=False):
            for masked_word_position in tqdm(masked_idxs, desc='Masked words', leave=False):
                
                sub_folder = os.path.join(
                    bert_lang,
                    str(context_size),
                    masked_word_position
                )
                
                folder_path = os.path.join(data_path, sub_folder)
                data_key = sub_folder.replace('/','__')

                calculate_file_features = partial(
                    calculate_features,
                    folder_path=folder_path
                )
                
                result[f'{data_key}__X'] = np.array([
                    calculate_file_features(idx=idx)
                    for idx in tqdm(indexes, desc='Text', leave=False)
                ])

    return result
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess bert features to use in classifier models'
    )

    parser.add_argument(
        'data_path',
        type=str,
        help='Path to the root directory where bert feature files are stored'
             'in the correct dir structure (<bert_lang>/<context_size>/<masked_word_position>/<idx>.json)'
    )

    args = parser.parse_args()

    load_data_and_feature_extraction(args.data_path)
