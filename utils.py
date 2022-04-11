import random
import os
import numpy as np
import torch
import json
from tqdm import tqdm
import pandas as pd
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("pretrained_model")


def set_seed(seed=56):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# truncate text from either head or tail part
def trunc_text(text, trunc_pos, length):

    text_ids = tokenizer.encode(text)[1:-1]

    if trunc_pos == 'head':
        text_trunc_ids = text_ids[:length]
    elif trunc_pos == 'tail':
        text_trunc_ids = text_ids[-length:]

    text_trunc_tokens = tokenizer.convert_ids_to_tokens(text_trunc_ids)
    text_trunc_back_sent = ''.join([x.replace('▁', ' ') for x in text_trunc_tokens])[:-1]

    return text_trunc_back_sent

# extract the title and text parts of a single news by id
def extract_news_byID(raw_data_root_dir, id: str):

    file_path = f"{raw_data_root_dir}/{id[-2:]}/{id}.json"

    if os.path.exists(file_path):

        with open(file_path, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
            news_text = f"{json_file['title']} {json_file['text']}"
            news_truncated_text = f"{trunc_text(news_text, 'head', 200)} {trunc_text(news_text, 'tail', 56)}"
            return news_truncated_text
    else:
        return None
        

def extract_data_from_raw(data_link_filepath, raw_data_root_dir, manual_crawl_file, dataset_save_filepath):
    
    # read the news missed by the tool provided by the organizers and crawled manually
    with open(manual_crawl_file, 'r', encoding='utf-8') as f:
        manual_crawl_dict = json.load(f)

    final_data = []
    final_columns = ['pair_id', 'lang1', 'lang2', 'text1', 'text2', 'Geography', 'Entities', 'Time', 'Narrative', 'Overall', 'Style', 'Tone']

    for _, row in tqdm(pd.read_csv(data_link_filepath).iterrows()):
        
        id1, id2 = row['pair_id'].strip().split('_')
        text1, text2 = extract_news_byID(raw_data_root_dir, id1), extract_news_byID(raw_data_root_dir, id2)

        if not text1: text1 = manual_crawl_dict[f"{row['pair_id']}"]['text1']
        if not text2: text2 = manual_crawl_dict[f"{row['pair_id']}"]['text2']

        cur_data = [row['pair_id'], row['lang1'], row['lang2'], text1, text2, row['Geography'], row['Entities'], row['Time'], row['Narrative'], row['Overall'], row['Style'], row['Tone']]
        final_data.append(cur_data)

    pd.DataFrame(final_data, columns=final_columns).to_csv(dataset_save_filepath)
