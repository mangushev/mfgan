
import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=4, suppress=True)

import collections
import re
import argparse
import sys
import os
import pickle
import tensorflow as tf

from prepare_data import audio_example
from model import MfGAN, get_shape_list

FLAGS = None

def make_input_fn_(filename, is_training, drop_reminder):
  """Returns an `input_fn` for train and eval."""

  def input_fn(params):
    def parser(serialized_example):
      example = tf.io.parse_single_example(
          serialized_example,
          features={
              "input": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
              "factors": tf.io.FixedLenFeature([FLAGS.num_factors, FLAGS.max_seq_length], tf.int64),
              "is_real_example": tf.io.FixedLenFeature((), tf.int64),
          })
      
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.to_int32(t)
        example[name] = t

      return example

    dataset = tf.data.TFRecordDataset(
      filename, buffer_size=FLAGS.dataset_reader_buffer_size)
    
    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        parser, batch_size=params["batch_size"],
        num_parallel_batches=8,
        drop_remainder=drop_reminder))
    return dataset

  return input_fn

def make_input_fn(filename, is_training, drop_reminder):
  """Returns an `input_fn` for train and eval."""

  filenames = filename.split(",")
  cycle_length = len(filenames)

  def input_fn(params):
    def parser(serialized_example):
      example = tf.io.parse_single_example(
          serialized_example,
          features={
              "input": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
              "factors": tf.io.FixedLenFeature([FLAGS.num_factors, FLAGS.max_seq_length], tf.int64),
              "is_real_example": tf.io.FixedLenFeature((), tf.int64),
          })
      
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.to_int32(t)
        example[name] = t

      return example

    #dataset = tf.data.TFRecordDataset(
    #  filename, buffer_size=FLAGS.dataset_reader_buffer_size)

    dataset = tf.data.Dataset.from_tensor_slices(filenames).interleave(tf.data.TFRecordDataset, cycle_length=cycle_length, block_length=32)

    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        parser, batch_size=params["batch_size"],
        num_parallel_batches=8,
        drop_remainder=drop_reminder))
    return dataset

  return input_fn

def model_fn_builder(init_checkpoint, learning_rate, num_train_steps, use_tpu):

  def model_fn(features, labels, mode, params):

    input = features["input"]
    factors = features["factors"]
    factor_bin_sizes = np.asarray(params["factor_bin_sizes"].split(","))
    is_real_example = features["is_real_example"]

    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

    model = MfGAN(input,
      factors,
      num_items=params["num_items"],
      factor_bin_sizes=factor_bin_sizes,
      hidden_size=params["hidden_size"],
      generator_hidden_layers=params["generator_hidden_layers"],
      discriminator_hidden_layers=params["discriminator_hidden_layers"],
      num_attention_heads=params["num_attention_heads"],
      initializer_range=params["initializer_range"],
      activation_fn=tf.nn.relu,
      dropout_prob=params["dropout_prob"],
      use_generator=params["use_generator"],
      is_training=is_training)

    if mode == tf.estimator.ModeKeys.TRAIN:

      tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

      if params["training_task"] == "pretrain-generator":

        #a_l = tf.Print(model.per_example_alignment_loss, [model.encoder_embedding], "_embeddings before grads", summarize=100)

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "input_embeddings")
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "input_positions"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "generator_block"))
        tvars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "prediction_layer"))

        #log likelihood for multinomunal distribution. There is one item - next_item
        #[B, I] --> [B]
        per_example_loss = -tf.math.reduce_sum(model.log_G, axis=-1, keepdims=False)
        #[B] --> [1]
        total_loss = tf.reduce_mean(per_example_loss)

      elif params["training_task"] == "pretrain-discriminator":

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        #[B] --> [B]
        per_example_loss = -(is_real_example*tf.math.log(model.Q)+(1-is_real_example)*tf.math.log(1-model.Q))
        #[B] --> [1]
        total_loss = tf.reduce_mean(per_example_loss)

      elif params["training_task"] == "gan-generator":

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        loss = model.Q
      elif params["training_task"] == "gan-discriminator":

        tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        loss = model.Q

      else:
        print ('unknows training')
        sys.exit(1)

      #print ("all tvars")
      #for i, v in enumerate(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)):
      #  tf.logging.info("{}: {}".format(i, v))

      #print ("selected tvars")
      for i, v in enumerate(tvars):
        tf.logging.info("{}: {}".format(i, v))

      grads = tf.gradients(total_loss, tvars, name='gradients')

      #for index in range(len(grads)):
      #  if grads[index] is not None:
      #    gradstr = "\n g_nan/g_inf/v_nan/v_inf/guid/grad [%i] | tvar [%s] =" % (index, tvars[index].name) 
      #    grads[index] = tf.Print(grads[index], [tf.reduce_any(tf.is_nan(grads[index])), tf.reduce_any(tf.is_inf(grads[index])), tf.reduce_any(tf.is_nan(tvars[index])), tf.reduce_any(tf.is_inf(tvars[index])), guid, grads[index], tvars[index]], gradstr, summarize=100)

      if (FLAGS.clip_gradients > 0):
        gradients, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
      else:
        gradients = grads

      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
      if FLAGS.use_tpu:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer) #, reduction=alignmnt_loss.Reduction.MEAN)

      train_op = optimizer.apply_gradients(zip(gradients, tvars), global_step=tf.compat.v1.train.get_global_step())

      training_hooks = None
      if not FLAGS.use_tpu:
        logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "step": tf.train.get_global_step()}, every_n_iter=1)
        training_hooks = [logging_hook]

      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode, predictions=None, loss=total_loss, train_op=train_op, eval_metrics=None,
        export_outputs=None, scaffold_fn=scaffold_fn, host_call=None, training_hooks=training_hooks,
        evaluation_hooks=None, prediction_hooks=None)

#    elif mode == tf.estimator.ModeKeys.EVAL:
#
#      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
#        #predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
#        #accuracy = tf.metrics.accuracy(
#        #    labels=label_ids, predictions=predictions, weights=is_real_example)
#        loss = tf.metrics.mean(values=per_example_loss, weights=None)
#        return {
#            "eval_accuracy": 0,
#            "eval_loss": loss,
#        }
#
#      eval_metrics = (metric_fn,
#                      [per_example_loss, 0, 0, 0])
#      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
#          mode=mode,
#          loss=total_loss,
#          eval_metrics=eval_metrics,
#          scaffold_fn=None)

    else:

      spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={'generated_sample': model.generated_sample,
                     'factors': factors
                    })
      return spec 

  return model_fn   

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  #tf.logging.set_verbosity(tf.logging.ERROR)

  tpu_cluster_resolver = None

  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=None,
      job_name='worker',
      coordinator_name=None,
      coordinator_address=None,
      credentials='default', 
      service=None,
      discovery_url=None
    )

  tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
    iterations_per_loop=FLAGS.iterations_per_loop, 
    num_cores_per_replica=FLAGS.num_tpu_cores,
    per_host_input_for_training=True 
  )

  run_config = tf.compat.v1.estimator.tpu.RunConfig(
    tpu_config=tpu_config,
    evaluation_master=None,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    master=None,
    cluster=tpu_cluster_resolver,
    **{
      'save_checkpoints_steps': FLAGS.save_checkpoints_steps,
      'tf_random_seed': FLAGS.random_seed,
      'model_dir': FLAGS.output_dir, 
      'keep_checkpoint_max': FLAGS.keep_checkpoint_max,
      'log_step_count_steps': FLAGS.log_step_count_steps
    }
  )

  use_generator = True
  if FLAGS.action == 'PREDICT':
    if FLAGS.prediction_task == 'generate_samples':
      use_generator = False

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    model_fn=model_fn_builder(FLAGS.init_checkpoint, FLAGS.learning_rate, FLAGS.num_train_steps, FLAGS.use_tpu),
    use_tpu=FLAGS.use_tpu,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size,
    predict_batch_size=FLAGS.batch_size,
    config=run_config,
    params={
        "hidden_size": FLAGS.hidden_size,
        "generator_hidden_layers": FLAGS.generator_hidden_layers,
        "discriminator_hidden_layers": FLAGS.discriminator_hidden_layers,
        "num_attention_heads": FLAGS.num_attention_heads,
        "num_items": FLAGS.num_items,
        "factor_bin_sizes": FLAGS.factor_bin_sizes,
        "initializer_range": FLAGS.initializer_range,
        "dropout_prob": FLAGS.dropout_prob,
        "use_tpu": FLAGS.use_tpu,
        "use_generator": use_generator,
        "training_task": FLAGS.training_task
    })

  if FLAGS.action == 'TRAIN':
    if FLAGS.training_task == "pretrain-discriminator":
      training_files = FLAGS.train_file.split(",")
      estimator.train(input_fn=make_input_fn(training_files, is_training=True, drop_reminder=True), max_steps=FLAGS.num_train_steps)
    else:
      estimator.train(input_fn=make_input_fn(FLAGS.train_file, is_training=True, drop_reminder=True), max_steps=FLAGS.num_train_steps)
  
  if FLAGS.action == 'EVALUATE':
    eval_drop_remainder = True if FLAGS.use_tpu else False
    results = estimator.evaluate(input_fn=make_input_fn(FLAGS.test_file, is_training=False, drop_reminder=eval_drop_remainder), steps=None)

    for key in sorted(results.keys()):
      tf.logging.info("  %s = %s", key, str(results[key]))

  if FLAGS.action == 'PREDICT':
    predict_drop_remainder = True if FLAGS.use_tpu else False
    results = estimator.predict(input_fn=make_input_fn(FLAGS.test_file, is_training=False, drop_reminder=predict_drop_remainder))

    if FLAGS.prediction_task == 'generate_samples':
      f = open("data/products.pkl", "rb")
      products = pickle.load(f)

      #print (repr(products))

      with tf.io.TFRecordWriter(FLAGS.output_file) as writer:
        for prediction in results:
          #print (repr(prediction["generated_sample"]))
          #print (prediction["generated_sample"].shape)

          sample_interaction = []
          for i in prediction["generated_sample"]:
            sample_interaction.append((products[i][0], products[i][1], products[i][2]))

          factors = np.array(sample_interaction).transpose([1, 0])

          tf_example = audio_example(prediction["generated_sample"], factors, False)

          writer.write(tf_example.SerializeToString())

    elif FLAGS.prediction_task == 'EVALUATE':
      labels = []
      anomalies = []
      for prediction in results:
        labels.append(prediction["label"])
        anomalies.append(prediction["predicted"])

      metrics = calculate_metrics(anomalies, labels, True)

      tf.logging.info("  %s = %s", "threshold", FLAGS.threshold)
      for key in sorted(metrics.keys()):
        tf.logging.info("  %s = %s", key, str(metrics[key]))
    else:
      output_predict_file = os.path.join("./", "Anomaly.csv")
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        for prediction in results:
          writer.write(str(prediction["anomaly"]) + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='gs://recommender/mfgan/output',
            help='Model directrory in google storage.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
            help='This will be checkpoint from previous training phase.')
    parser.add_argument('--train_file', type=str, default='gs://anomaly_detection/mtad_tf/data/train.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--test_file', type=str, default='gs://anomaly_detection/mtad_tf/data/test.tfrecords',
            help='Test file location in google storage.')
    parser.add_argument('--output_file', type=str, default='gs://anomaly_detection/mtad_tf/data/output.tfrecords',
            help='Output file location in google storage.')
    parser.add_argument('--dropout_prob', type=float, default=0.0,
            help='This used for all dropouts.')
    parser.add_argument('--num_train_steps', type=int, default=78000,
            help='Number of steps to run trainer.')
    parser.add_argument('--iterations_per_loop', type=int, default=1000,
            help='Number of iterations per TPU training loop.')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--log_step_count_steps', type=int, default=1000,
            help='Number of step to write logs.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
    parser.add_argument('--dataset_reader_buffer_size', type=int, default=100,
            help='input pipeline is I/O bottlenecked, consider setting this parameter to a value 1-100 MBs.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=24000,
            help='Items are read from this buffer.')
    parser.add_argument('--use_tpu', default=False, action='store_true',
            help='Train on TPU.')
    parser.add_argument('--tpu', type=str, default='node-1-15-2',
            help='TPU instance name.')
    parser.add_argument('--num_tpu_cores', type=int, default=8,
            help='Number of cores on TPU.')
    parser.add_argument('--tpu_zone', type=str, default='us-central1-c',
            help='TPU instance zone location.')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
            help='Optimizer learning rate.')
    parser.add_argument('--clip_gradients', type=float, default=-1.,
            help='Clip gradients to deal with explosive gradients.')
    parser.add_argument('--random_seed', type=int, default=1234,
            help='Random seed to initialize values in a grath. It will produce the same results only if data and grath did not change in any way.')
    parser.add_argument('--initializer_range', type=float, default=0.02,
            help='.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN','EVALUATE','PREDICT'],
            help='An action to execure.')
    parser.add_argument('--training_task', choices=['pretrain-generator', 'pretrain-discriminator', 'gan-generator', 'gan-discriminator'],
            help='Training phase.')
    parser.add_argument('--prediction_task', default='next_item', choices=['generate_samples', 'next_item'],
            help='Values to predict.')
    parser.add_argument('--restore', default=False, action='store_true',
            help='Restore last checkpoint.')
    parser.add_argument('--hidden_size', type=int, default=50,
            help='dimension of each network in the Feed-Forward Transformer is all set to 768.')
    parser.add_argument('--generator_hidden_layers', type=int, default=2,
            help='Feed-Forward  Transformer  contains  6  FFT  blocks.')
    parser.add_argument('--discriminator_hidden_layers', type=int, default=1,
            help='Feed-Forward  Transformer  contains  6  FFT  blocks.')
    parser.add_argument('--num_attention_heads', type=int, default=2,
            help='number of attention head is set to 2 in all FFT block.')
    parser.add_argument('--num_items', type=int, default=10482,
            help='Computer instance metrics.')
    parser.add_argument('--factor_bin_sizes', type=str,
            help='Computer instance metrics.')
    parser.add_argument('--max_seq_length', type=int, default=20,
            help='Computer instance metrics.')
    parser.add_argument('--num_factors', type=int, default=3,
            help='Computer instance metrics.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
