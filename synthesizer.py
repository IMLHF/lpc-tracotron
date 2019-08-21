import numpy as np
import time
import tensorflow as tf
from hparams import hparams
from models import create_model
from text import text_to_sequence
import lpcnet


class Synthesizer:
    def load(self, checkpoint_path, model_name='tacotron'):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        with tf.variable_scope('model'):
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs, input_lengths, is_training=False)

        print('Loading checkpoint: %s' % checkpoint_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
        }

        toc = time.time()
        features = self.session.run(self.model.lpc_outputs, feed_dict=feed_dict)
        print('[{:<10}]: generating lpc feature escaped '.format('tacotron'), time.time() - toc)

        toc = time.time()
        feature = features[0]
        wav = lpcnet.Synthesizer().synthesis(feature)
        print('[{:<10}]: generating wavform escaped '.format('vocoder'), time.time() - toc)
        return wav, features
