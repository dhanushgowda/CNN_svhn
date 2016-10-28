import os

import tensorflow as tf
from PIL import Image
import graph
import input
import scipy.io as sio
import input
BATCH_SIZE = 1
DS_SIZE = 58240
VALID_SIZE = 14976

CHECKPOINT_DIR = "checkpoints/"

def _read_svhn_file(filepath):
    print("Reading", filepath)
    datadict = sio.loadmat(filepath)
    y = datadict['y'].reshape(datadict['y'].shape[0], )
    X, Y = (datadict['X'].transpose((3, 0, 1, 2)), y)
    i = 0
    file_names = []
    if filepath == "data/test_32x32.mat":
        for data in X[0:20]:
            i += 1
            img = Image.fromarray(data, 'RGB')
            file_name = 'static/images/svhn' + str(i) + '.png'
            img.save(file_name)
            file_names.append(file_name)

    return X[0:20], Y[0:20], file_names

X_test, y_test, file_names = _read_svhn_file("data/test_32x32.mat")
test = input.DataSet(X_test, y_test)

def run_prediction(data, file_names):
    ret = []
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 32, 32, 3])
        labels_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = graph.inference(images_pl)

        saver = tf.train.Saver(tf.all_variables())

        init = tf.initialize_all_variables()
        sess = tf.Session()

        saver.restore(sess, "checkpoints/-500")
        print("Model restored.")

        sess.run(init)

        images, labels = (data.images, data.labels)

        for example in range(data.num_examples):

            feed_dict = {
                images_pl: [images[example]],
                labels_pl: [labels[example]]
            }

            logits_op, actual = sess.run([logits,labels_pl], feed_dict=feed_dict)

            predicted = logits_op[0].argmax()

            print(logits_op[0], predicted, actual)
            ret.append([predicted, actual, file_names[example]])

        return ret

# data, file_names = input.get_data(num_training=DS_SIZE, num_validation=VALID_SIZE)
# run_prediction(test, file_names)

def get_predictions():
    prediction_op = run_prediction(test, file_names)
    return prediction_op, file_names

if __name__=="__main__":
    get_predictions()