from coherence_transformers import coherence_transformers
from DataSchiz import data
import numpy as np

import argparse
import json
import os
from pathlib import Path

from .utils import chained_tqdm

def encode_defaults(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def data_schema(speech):
    return {
        'corpus': speech['corpus'],
        'path': speech['path'],
        'file_path': speech['file_path'],
        'experimental group': speech['experimental group'],
        'language': speech['language'],
        'text': speech['text'],
        'total words': speech['total words'],
        'data': []
    }

def calculate_transformer_probas(output_path):

    print('Load bert multilingual...')
    bert_multilingual = coherence_transformers(
        model='bert-base-cased',
        verbose=False
    )
    print('Load bert spanish...')
    bert_spanish = coherence_transformers(
        model='bert-base-cased',
        verbose=False
    )
    print('Load bert english...')
    bert_english = coherence_transformers(
        model='dccuchile/bert-base-spanish-wwm-cased',
        verbose=False
    )

    pbar_bl, pbar_cs, pbar_mw, pbar_dt, pbar_wd = chained_tqdm(
        [3,2,2,len(data.data), 1],
        ['Bert lang', 'Context size', 'Masked words', 'Text', 'Calculate Bert']
    )
    
    for bert_lang in ['single_language', 'multilingual']:
        pbar_cs.refresh()
        pbar_cs.reset()
        pbar_cs.set_description('Context size')

        # Seteo los modelos BERT a usar (multi o single language)
        if bert_lang == 'single_language':
            models = {
                'es': bert_spanish,
                'en': bert_english
            }
        elif bert_lang == 'multilingual':
            # Si uso bert multilingual, entonces para ambos idiomas es el mismo modelo
            models = {
                'es': bert_multilingual,
                'en': bert_multilingual
            }
        

        # Para conseguir ventanas simetricas con 2, 5 y 8 palabras de cada lado
        for context_size in [5,11,17]:
            pbar_mw.refresh()
            pbar_mw.reset()
            pbar_mw.set_description('Masked words')

            for masked_word_position in ['middle', 'last']:
                pbar_dt.refresh()
                pbar_dt.reset()
                pbar_dt.set_description('Text')

                # Set prediction context
                masked_words = [False for i in range(context_size)]
                if masked_word_position == 'middle':
                    masked_words[context_size // 2 + 1] = True
                else:
                    masked_words[-1] = True

                dir_path = os.path.join(
                    output_path,
                    bert_lang,
                    str(context_size),
                    masked_word_position
                )
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                for idx, speech in data.data.iterrows():
                    pbar_wd.refresh()
                    pbar_wd.reset()
                    pbar_wd.set_description('Calculate Bert')

                    res_list = models[speech['language']].extract_coherence_features(
                        speech['text'],
                        masked_words=masked_words,
                        n_top_probs=1000,
                        tqdm_bar=pbar_wd
                    )['res']

                    result = data_schema(speech)

                    pbar_wd.refresh()
                    pbar_wd.reset(total=len(res_list)*1000)
                    pbar_wd.set_description('Compile results')
                    for res in res_list:

                        result['data'].append({
                            'word_position': res['hwin_index'], 
                            'real_word': res['real_word_list'][0],
                            'real_word_probability': res['real_word_prob_list'][0],
                            'real_word_rank': res['real_word_rank_list'][0],
                            'bert_words': []
                        })

                        # TODO: Checkear el formato que devuelve BERT
                        for top_i_prob in res['top_n_most_probable_word_lists']:
                            result['data'][-1]['bert_words'].append({
                                'word': top_i_prob['model_predicted_word_list'][0],
                                'probability': top_i_prob['model_predicted_word_prob_list'][0],
                            })
                            pbar_wd.update()

                    file_path = os.path.join(dir_path, f'{idx}.json')
                    json.dump(
                        result,
                        open(file_path, 'w', encoding='utf-8'),
                        indent=6,
                        default=encode_defaults
                    )

                    pbar_dt.update()
                pbar_mw.update()
            pbar_cs.update()
        pbar_bl.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Calculates a set of BERT probabilities to' 
                    'use as features to clasify schizophrenia disease diagnosis.'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        help='The output path where to store the resulting JSON files'
    )

    args = parser.parse_args()

    calculate_transformer_probas(args.output_path)