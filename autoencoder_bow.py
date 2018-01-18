# !/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import preprocess as p
import numpy as np


'''
load data
'''
file_path = 'data161207.txt'
all_data = p.load_data(file_path)
selected_categories = ['검색모형/기법', '계량정보학', '기록관리/보존', '데이터베이스', '도서관/정보센터경영', '도서관사', '디지털도서관', '문헌정보학일반', '분류', '정보검색',
                       '정보교육', '정보서비스', '정보/도서관정책', '정보자료/미디어', '자동분류/클러스터링', '자동색인/요약', '전문용어/시소러스', '편목/메타데이터']
selected_data = p.extract_data_by_selected_category(all_data, selected_categories)
bow, idx = p.convert_text_to_bow(selected_data)

'''
set autoencoder
'''
# parameter
num_of_documents = bow.shape[0]
num_of_keywords = bow.shape[1]
num_of_encoder_nodes = 500
learning_rate = 0.01
# input
X = tf.placeholder('float', [num_of_documents, num_of_keywords])
# encoder
encoder_weights = tf.Variable(tf.random_normal([num_of_keywords, num_of_encoder_nodes]))
encoder_bias = tf.Variable(tf.random_normal([num_of_encoder_nodes]))
encoder_operation = tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_weights), encoder_bias))
# decoder
decoder_weights = tf.Variable(tf.random_normal([num_of_encoder_nodes, num_of_keywords]))
decoder_bias = tf.Variable(tf.random_normal([num_of_keywords]))
decoder_operation = tf.nn.sigmoid(tf.add(tf.matmul(encoder_operation, decoder_weights), decoder_bias))
# cost
cost = tf.reduce_mean(tf.pow(decoder_operation - X, 2))
# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

'''
train autoencoder
'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
_, cost_val = sess.run([optimizer, cost], feed_dict={X: bow})

'''
output
'''
output = sess.run(decoder_operation, feed_dict={X: bow})
np.savetxt('original.txt', bow, delimiter=' ', fmt='%s')
np.savetxt('autoencoder.txt', output, delimiter=' ', fmt='%s')
