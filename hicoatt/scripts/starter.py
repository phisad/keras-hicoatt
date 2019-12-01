# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf;


def main():
    tf.enable_eager_execution();
    result = tf.reduce_sum(tf.random_normal([1000, 1000]))
    print(result)
    print("Setup TF", tf.__version__, "successful")


if __name__ == "__main__":
    main()
