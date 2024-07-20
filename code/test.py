import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization
import pandas as pd

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

if __name__ == '__main__':
    test()
