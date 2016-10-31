import tensorflow as tf
import time

import graph
import input

BATCH_SIZE = 128
DS_SIZE = 58240
VALID_SIZE = 14976  # GySm7gI$Jd4FDa3D
N_EPOCH = 10
SUMMARY_DIR = "summary"
CHECKPOINT_DIR = "checkpoints/"


def do_eval(sess, eval_correct, images_pl, labels_pl, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_pl, labels_pl)
        true_count += sess.run(eval_correct, feed_dict)
    accuracy = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
          (num_examples, true_count, accuracy))


def fill_feed_dict(data_set, images_pl, labels_pl):
    images, labels = data_set.next_batch(BATCH_SIZE)
    return {
        images_pl: images,
        labels_pl: labels
    }


def run_evaluation(data):
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 32, 32, 3])
        labels_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = graph.inference(images_pl)
        eval_correct = graph.evaluate(logits, labels_pl)

        saver = tf.train.Saver(tf.all_variables())

        # init = tf.initialize_all_variables()
        sess = tf.Session()

        saver.restore(sess, "checkpoints/-4500")
        print("Model restored.")

        # sess.run(init)
        do_eval(sess, eval_correct, images_pl, labels_pl, data.test)


start = time.time()

data = input.get_data(num_training=DS_SIZE, num_validation=VALID_SIZE)
run_evaluation(data)

end = time.time()
print("Total time:", (end - start), "secs")

