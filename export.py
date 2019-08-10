import argparse
from datetime import datetime
import os
import tensorflow as tf

from hparams import hparams
from models import create_model
log = infolog.log


def add_stats(model):
    with tf.variable_scope('stats'):
        tf.summary.histogram('linear_outputs', model.linear_outputs)
        tf.summary.histogram('linear_targets', model.linear_targets)
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('loss_mel', model.mel_loss)
        tf.summary.scalar('loss_linear', model.linear_loss)
        # tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('stop_token_loss', model.stop_token_loss)
        tf.summary.scalar('learning_rate', model.learning_rate)
        tf.summary.scalar('loss', model.loss)
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
        return tf.summary.merge_all()


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def freeze(log_dir, args, export_path='export/1'):
    ckpt_path = 'logs-tacotron/model.ckpt-{}'.format(320000)

    print('Constructing model')
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
        model = create_model('tacotron', hparams)
        model.initialize(inputs, input_lengths)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    mel_outputs = model.mel_outputs
    wav_outputs = audio.inv_spectrogram_tensorflow(model.linear_outputs[0])

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    sig_inputs = tf.saved_model.utils.build_tensor_info(inputs)
    sig_input_lengths = tf.saved_model.utils.build_tensor_info(input_lengths)
    sig_outputs = tf.saved_model.utils.build_tensor_info(wav_outputs)
    sig_mels = tf.saved_model.utils.build_tensor_info(mel_outputs)


    signature_wav = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'inputs': sig_inputs,
                'input_lengths': sig_input_lengths
            },
            outputs={
                'wavs': sig_outputs
            },
            method_name='tensorflow/serving/predict'))

    signature_mel = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'inputs': sig_inputs,
                'input_lengths': sig_input_lengths
            },
            outputs={
                'mels': sig_mels
            },
            method_name='tensorflow/serving/predict'))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_mels':
                signature_mel,
            'predict_wavs':
                signature_wav
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    builder.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('.'))
    parser.add_argument('--input', default='training/train.txt')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=bool, default=True,
                        help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=100,
                        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints.')
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1,
                        help='Tensorflow C++ log level.')
    parser.add_argument('--git', action='store_true',
                        help='If set, verify that the client is clean.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    hparams.parse(args.hparams)
    freeze(log_dir, args)


if __name__ == '__main__':
    main()
