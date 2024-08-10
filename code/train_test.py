import argparse
import multiprocessing
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.tools import tokenization
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Get BERT layer and tokenizer:
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# This provides a function to convert row to input features and label
label_list = [0, 1] # Label categories
max_seq_length = 128 # maximum length of (token) input sequences
train_batch_size = 16

def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
    example = classifier_data_lib.InputExample(guid=None,
                                               text_a=text.numpy(),
                                               text_b=None,
                                               label=label.numpy())
    feature = classifier_data_lib.convert_single_example(0, example, label_list,
                                                         max_seq_length, tokenizer)

    return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)


def to_feature_map(text, label):
    input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label],
                                Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

    # py_func doesn't set the shape of the returned tensors.
    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    label_id.set_shape([])

    x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
        }
    return (x, label_id)

# Building the model
def create_model():
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                             name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                         name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                             name="input_type_ids")
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
    drop = tf.keras.layers.Dropout(0.4)(pooled_output)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(drop)
    model = tf.keras.Model(
          inputs={
              'input_word_ids': input_word_ids,
              'input_mask': input_mask,
              'input_type_ids': input_type_ids
          },
          outputs=output)
    return model


def train_test(q, df, train, test, rt, fold_no):
    try:
        with tf.device('/cpu:0'):

            train_data = tf.data.Dataset.from_tensor_slices((df.comments.values[train], df.target.values[train]))
            test_data = tf.data.Dataset.from_tensor_slices((df.comments.values[test], df.target.values[test]))
            real_test_data = tf.data.Dataset.from_tensor_slices((rt.comments.values, rt.target.values))
            # train
            train_data = (train_data.map(to_feature_map,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
                          # .cache()
                          .shuffle(1000)
                          .batch(16, drop_remainder=True)
                          .prefetch(tf.data.experimental.AUTOTUNE))
            # test
            test_data = (test_data.map(to_feature_map,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         .batch(16, drop_remainder=True)
                         .prefetch(tf.data.experimental.AUTOTUNE))
            # real test
            real_test_data = (real_test_data.map(to_feature_map,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
                              .batch(16, drop_remainder=True)
                              .prefetch(tf.data.experimental.AUTOTUNE))

        model = create_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        model.summary()

        # Train model
        epochs = 4
        history = model.fit(train_data,
                            epochs=epochs,
                            verbose=1)
        # Evaluate the model
        loss, accuracy, precision, recall = model.evaluate(test_data)

        # Evaluate the model by using real data
        loss1, accuracy1, precision1, recall1 = model.evaluate(real_test_data)
        model.save('../models/results/'+str(fold_no))
    except Exception as e:
        loss = None
        accuracy = None
        precision = None
        recall = None
        loss1 = None
        accuracy1 = None
        precision1 = None
        recall1 = None

    q.put((loss, accuracy, precision, recall, loss1, accuracy1, precision1, recall1))

def train():
    num_folds = 5
    df = pd.read_csv('../dataset/Ground_Truth_Dataset.csv',on_bad_lines='skip', encoding="utf-8")
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=5)
    rt = pd.read_csv('../dataset/Test_Dataset.csv',on_bad_lines='skip', encoding="utf-8")

    # Define per-fold score containers
    acc_per_fold = []
    pre_per_fold = []
    loss_per_fold = []
    recall_per_fold = []
    acc_per_fold1 = []
    pre_per_fold1 = []
    loss_per_fold1 = []
    recall_per_fold1 = []

    fold_no = 0
    q = multiprocessing.Queue()
    for train, test in kfold.split(df.comments.values, df.target.values):
        fold_no = fold_no + 1
        p = multiprocessing.Process(target=train_test, args=[q, df, train, test, rt, fold_no])
        p.start()
        loss, accuracy, precision, recall, loss1, accuracy1, precision1, recall1 = q.get()
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy1}')
        print(f'Precision: {precision1}')
        print(f'Recall: {recall1}')
        acc_per_fold.append(accuracy * 100)
        pre_per_fold.append(precision*100)
        loss_per_fold.append(loss)
        recall_per_fold.append(recall * 100)
        acc_per_fold1.append(accuracy1 * 100)
        pre_per_fold1.append(precision1*100)
        loss_per_fold1.append(loss1)
        recall_per_fold1.append(recall1 * 100)
        print('------------------------------------------------------------------------')
        p.join()

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - Precision: {pre_per_fold[i]}% - Recall: {recall_per_fold[i]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Precision: {np.mean(pre_per_fold)} (+- {np.std(pre_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print(f'> Recall: {np.mean(recall_per_fold)} (+- {np.std(recall_per_fold)})')

def test():
    rt = pd.read_csv('../dataset/Test_Dataset.csv',on_bad_lines='skip', encoding="utf-8")
    real_test_data = tf.data.Dataset.from_tensor_slices((rt.comments.values, rt.target.values))
    real_test_data = (real_test_data.map(to_feature_map,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
                      .batch(16, drop_remainder=True)
                      .prefetch(tf.data.experimental.AUTOTUNE))
    model = tf.keras.models.load_model('../models/best')
    loss, accuracy, precision, recall = model.evaluate(real_test_data)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

def parse_args():
    parse = argparse.ArgumentParser(description="Train and test code")
    parse.add_argument('--train', help="Train Process", action="store_true")
    parse.add_argument('--test', help="Test Process", action="store_true")
    args = parse.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    if args.train:
        train()
    if args.test:
        test()
