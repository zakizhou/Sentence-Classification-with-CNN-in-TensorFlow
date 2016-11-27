from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf
from inputs import Inputs
from model import Config, TextCNN


def main(*args, **kwargs):
    inputs = Inputs()
    config = Config()
    with tf.variable_scope("inference") as scope:
        model = TextCNN(config, inputs)
        scope.reuse_variables()
        validation_model = TextCNN(Config, inputs)

    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)
    try:
        index = 0
        while not coord.should_stop():
            _, loss_value = sess.run([model.train, model.loss])
            index += 1
            print("step: %d, loss: %f" % (index, loss_value))
    except tf.errors.OutOfRangeError:
        print("Done traing:-------Epoch limit reached")
    except KeyboardInterrupt:
        print("keyboard interrput detected, stop training")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    del sess


if __name__ == "__main__":
    main()
