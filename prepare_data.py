
#TODO: 

import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import random
import logging
import librosa

FLAGS = None

np.set_printoptions(edgeitems=12, linewidth=10000, precision=4, suppress=True)

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def audio_example(input, factors, is_real_example):
    record = {
        'input': tf.train.Feature(int64_list=tf.train.Int64List(value=input)),
        'factors': tf.train.Feature(int64_list=tf.train.Int64List(value=np.reshape(factors, [-1]))),
        'is_real_example': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(is_real_example)]))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

def create_records_0(reviews_file, games_file, tfrecords_file):

  import json
  import ast
  import operator
  import pandas as pd
  import numbers
  import re
  import itertools
  
  games = {}  
  sentiments = {}  
  developers = {}  

  with open(games_file, encoding='utf8') as f:
    pattern = re.compile('\d+.?\d*')
    for line in f:
      game = ast.literal_eval(line)

      if not 'id' in game:
        continue

      #if not 'genres' in game:
      #  continue

      if not 'sentiment' in game:
        sentiment = 'UNKNOWN'
      else:
        sentiment = game['sentiment']
      if not sentiment in sentiments:
        sentiments[sentiment] = len(sentiments)+1

      if not 'price' in game:
        price = -1.0
      else:
        if isinstance(game['price'], numbers.Real):
          price = game['price']
        else:
          match = re.search(pattern, game['price'])
          if match:
            price = float(match.group())
          else:
            price = 0.0
      if (price < 0):
        price = 0
      elif price == 0:
        price = 1
      elif price < 1:
        price = 2
      elif price < 10:
        price = 3
      elif price < 100:
        price = 4
      else:
        price = 5

      if not 'developer' in game:
        developer = 'UNKNOWN'
      else:
        developer = game['developer']
      if not developer in developers:
        developers[developer] = len(developers)+1

      games[game['id']] = {'popularity': sentiments[sentiment], 'price':price, 'developer':developers[developer]}

  users = {}
  interaction = []

  with open(reviews_file, encoding='utf8') as f:
    for line in f:
      review = ast.literal_eval(line)

      if not review['username'] in users:
        users[review['username']] = len(users)+1

      user_id = users[review['username']]

      #logger.info ("{} {}".format(review['username'], review['product_id']))
      #print (review['username'])

      if review['product_id'] in games:
        interaction.append((user_id, review['product_id'], review['date']))

  core_interaction = interaction
  remove_users = []

  for i in range(100):
    sorted_interaction = sorted(core_interaction, key=operator.itemgetter(0))

    core_interaction = []
    for k, g in itertools.groupby(sorted_interaction, key=operator.itemgetter(0)):
      gr = list(g)
      if len(gr) >= 5 and not k in remove_users:
        core_interaction.extend(gr)
    print ("step user", i, len(core_interaction))

    length = len(core_interaction)

    sorted_interaction = sorted(core_interaction, key=operator.itemgetter(1))

    core_interaction = []
    remove_users = []
    for k, g in itertools.groupby(sorted_interaction, key=operator.itemgetter(1)):
      gr = list(g)
      if len(gr) >= 5:
        core_interaction.extend(gr)
      else:
        remove_users.extend(gr)
    print ("step item", i, len(core_interaction))
    
    if length == len(core_interaction) and len(remove_users) == 0 :
      break

    #since some items are removed, specific user interaction is incomplete, so these users have to be removed
    remove_interaction = sorted(remove_users, key=operator.itemgetter(0))
    remove_users = [k for k, g in itertools.groupby(remove_interaction, key=operator.itemgetter(0))]

  interacted_games = [k for k, g in itertools.groupby(core_interaction, key=operator.itemgetter(1))]

  products = {}
  product_list = []
  i = 0
  for k in games:
    if k in interacted_games:
      products[k] = (i, sentiments[sentiment], price, developers[developer])
      product_list.append((sentiments[sentiment], price, developers[developer]))
      i = i + 1

  import pickle
  f = open("data/products.pkl", "wb")
  pickle.dump(product_list, f)
  f.close()

  print ("num_sentiment", len(sentiments))
  print ("num_price", 6)
  print ("num_developer", len(developers))
  print ("num_items", len(products))

  return

  sorted_interaction = sorted(core_interaction, key=operator.itemgetter(0, 2))

  interaction = []
  for item in sorted_interaction:
    interaction.append((item[0], products[item[1]][0], products[item[1]][1], products[item[1]][2], products[item[1]][3]))

  interation_lengths = []
  with tf.io.TFRecordWriter(tfrecords_file) as writer:
    for k, g in itertools.groupby(interaction, key=operator.itemgetter(0)):
      group = list(g)
      input_tensor = np.array(group)[:, 1].reshape(-1)
      factors = np.array(group)[:, 2:].transpose([1, 0])
      
      interation_lengths.append(input_tensor.shape[0])

      #print (factors)
      print (factors.shape)

      if (input_tensor.shape[0]<FLAGS.max_seq_length):
        input_tensor = np.concatenate((np.zeros((FLAGS.max_seq_length-input_tensor.shape[0]), dtype=int), input_tensor),axis=0)
        factors = np.concatenate((np.zeros((factors.shape[0], FLAGS.max_seq_length-factors.shape[1]), dtype=int), factors),axis=1)
      elif (input_tensor.shape[0]>FLAGS.max_seq_length):
        input_tensor = input_tensor[input_tensor.shape[0]-FLAGS.max_seq_length:]
        factors = factors[:, factors.shape[1]-FLAGS.max_seq_length:]

      print (input_tensor.shape)
      #print (factors)
      print (factors.shape)

      tf_example = audio_example(input_tensor, factors, True)
      #tf_example = audio_example(name, input_tensor, len(transcript), input_mask, input_durations, mel, mel_len)

      writer.write(tf_example.SerializeToString())

  print ("min/max/mean: ", np.min(interation_lengths), np.max(interation_lengths), np.mean(interation_lengths))

def main():
    #record_count, input_long_count, mel_long_count, max_input_len, max_mel_len = create_records(FLAGS.metadata_file, FLAGS.audio_files, FLAGS.durations_file, FLAGS.tfrecords_file)

    create_records_0(FLAGS.activity, FLAGS.items, FLAGS.tfrecords_file)

    #logging.info ("record_count {} inputr_long {} label_long {} max_input_len {} max_mel_len {}".format(record_count, input_long_count, mel_long_count, max_input_len, max_mel_len))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_rate', type=int, default=22050,
            help='Signal will be resampled to this rate.')
    parser.add_argument('--num_mels', type=int, default=80,
            help='This is number of mel filter banks as per Deep Speech 1 article.')
    parser.add_argument('--winlen', type=float, default=0.020,
            help='Audio frame window size as per Deep Speech 1 article.')
    parser.add_argument('--winstep', type=float, default=0.010,
            help='Audio frame sliding as per Deep Speech 1 article.')
    parser.add_argument('--max_seq_length', type=int, default=20,
            help='Max length of output strings in characters will shorter strings filled with zeros.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--activity', type=str, default='/work/datasets/Steam_Games/steam_reviews.json',
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--items', type=str, default='/work/datasets/Steam_Games/steam_games.json',
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--durations_file', type=str, default=None,
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--audio_files', type=str, default=None,
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--tfrecords_file', type=str, default='data/{}.tfrecords',
            help='tfrecords output file. It will be used as a prefix if split.')

    FLAGS, unparsed = parser.parse_known_args()

    logger.setLevel(FLAGS.logging)

    logger.debug ("Running with parameters: {}".format(FLAGS))

    main()

