# !/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
import collections
import numpy as np


Data = collections.namedtuple('Data_set', ['did', 'year', 'category', 'descriptor', 'text'])
Data_set = collections.namedtuple('Data_sets', ['train', 'validation', 'test'])


def load_data(filename, column_delimiter='\t', item_delimiter=';'):
    with open(filename, 'r', encoding='UTF8') as csv_file:
        data_file = csv.reader(csv_file, delimiter=column_delimiter)
        did, year, category, descriptor, text = [], [], [], [], []
        for row in data_file:
            did.append(row[0].strip())
            year.append(row[1].strip())
            category.append(row[2].strip())
            descriptor.append(np.array(map(str.strip, row[3].split(item_delimiter)), dtype=np.str))
            text.append(row[4].strip())
        did = np.array(did, dtype=np.int32)
        year = np.array(year, dtype=np.int32)
        category = np.array(category, dtype=np.str)
        descriptor = np.array(descriptor)
        text = np.array(text, dtype=np.str)
        return Data(did=did, year=year, category=category, descriptor=descriptor, text=text)


def extract_data_by_year(data, begin, end):
    mask = (data.year >= begin) & (data.year <= end)
    did = data.did[mask]
    year = data.year[mask]
    category = data.category[mask]
    descriptor = data.descriptors[mask]
    text = data.text[mask]
    return Data(did=did, year=year, category=category, descriptor=descriptor, text=text)
    

def extract_data_by_selected_category(data, categories):
    mask = np.isin(data.category, categories)
    did = data.did[mask]
    year = data.year[mask]
    category = data.category[mask]
    descriptor = data.descriptor[mask]
    text = data.text[mask]
    return Data(did=did, year=year, category=category, descriptor=descriptor, text=text)


def covert_label_to_idx(data):
    label_idx = {}
    for idx, name in enumerate(set(data.category)):
        label_idx[name] = idx
    category = []
    for name in data.category:
        category.append(label_idx[name])
    category = np.array(category, dtype=np.int32)
    return Data(did=data.did, year=data.year, category=category, descriptor=data.descriptor, text=data.text), label_idx


def convert_text_to_bow(data):
    token_idx, tokens = {}, {}
    idx = 0
    for row_idx, text in enumerate(data.text):
        tokens[row_idx] = collections.defaultdict(lambda: 0)
        for token in text.split():
            if token not in token_idx:
                token_idx[token] = idx
                idx = idx + 1
            tokens[row_idx][token_idx[token]] += 1
    bow = np.zeros(shape=(len(data.text), len(token_idx)))
    for row_idx in tokens:
        for column_idx in tokens[row_idx]:
            bow[row_idx, column_idx] = tokens[row_idx][column_idx]
    return bow, token_idx


if __name__ == '__main__':    
    file_path = 'data161207.txt'
    all_data = load_data(file_path)
    
    selected_categories = ['검색모형/기법', '계량정보학', '기록관리/보존', '데이터베이스', '도서관/정보센터경영', '도서관사', '디지털도서관', '문헌정보학일반', '분류', '정보검색', '정보교육', '정보서비스', '정보/도서관정책', '정보자료/미디어', '자동분류/클러스터링', '자동색인/요약', '전문용어/시소러스', '편목/메타데이터']
    selected_data = extract_data_by_selected_category(all_data, selected_categories)
    
    train_begin_year = 2002
    train_end_year = 2011
    train = extract_data_by_year(all_data, train_begin_year, train_end_year)
    
    test_begin_year = 2013
    test_end_year = 2015
    test = extract_data_by_year(all_data, test_begin_year, test_end_year)

    print(set(selected_data.category))
    print(len(set(selected_data.category)))
    
    Data_set(train=train, validation=None, test=test)


