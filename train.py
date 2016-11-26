from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf
from inputs import Inputs, records_inputs, read_and_decode
from model import Config, TextCNN


def main(*args, **kwargs):
    filename_queue = tf.train.string_input_producer(["inputs/train.tfrecords"])
    input, label = read_and_decode(filename_queue)
    inputs, labels = records_inputs(input, label)
    model_inputs = Inputs(inputs, labels)
    config = Config()
    model = TextCNN(config, model_inputs)

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
