import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
log = infolog.log


def get_git_commit():
    # Verify client is clean
    subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
    log('Git commit: %s' % commit)
    return commit


def add_stats(model):
    with tf.variable_scope('stats'):
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


def train(log_dir, args):
    commit = get_git_commit() if args.git else 'None'
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    args.base_dir = './'
    input_path = os.path.join(args.base_dir, args.input)
    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_path)
    log('Using model: %s' % args.model)
    log(hparams_debug_string())

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = DataFeeder(coord, input_path, hparams)

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('model') as scope:
        model = create_model(args.model, hparams)
        model.initialize(feeder.inputs, feeder.input_lengths,
                         feeder.lpc_targets, feeder.stop_token_targets)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_stats(model)

    # Bookkeeping:
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    # Train!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            if args.restore_step:
                # Restore from a checkpoint if the user requested it.
                checkpoint_state = tf.train.get_checkpoint_state(log_dir)
                # restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                if checkpoint_state is not None:
                    saver.restore(sess, checkpoint_state.model_checkpoint_path)
                    log('Resuming from checkpoint: %s at commit: %s' % (
                        checkpoint_state.model_checkpoint_path, commit), slack=True)
            else:
                log('Starting new training run at commit: %s' %
                    commit, slack=True)

            if args.restore_decoder:
                models = [f for f in os.listdir(
                    'pretrain') if f.find('.meta') != -1]
                decoder_ckpt_path = os.path.join(
                    'pretrain', models[0].replace('.meta', ''))

                global_vars = tf.global_variables()
                var_list = []
                valid_scope = [
                    'model/inference/decoder',
                    'model/inference/post_cbhg',
                    'model/inference/dense',
                    'model/inference/memory_layer']
                for v in global_vars:
                    if v.name.find('attention') != -1:
                        continue
                    if v.name.find('Attention') != -1:
                        continue
                    for scope in valid_scope:
                        if v.name.startswith(scope):
                            var_list.append(v)
                decoder_saver = tf.train.Saver(var_list)
                decoder_saver.restore(sess, decoder_ckpt_path)
                print('restore pretrained decoder ...')

            feeder.start_in_session(sess)

            while not coord.should_stop():
                start_time = time.time()
                step, loss, opt = sess.run(
                    [global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
                    step, time_window.average, loss, loss_window.average)
                log(message, slack=(step % args.checkpoint_interval == 0))

                if loss > 100 or math.isnan(loss):
                    log('Loss exploded to %.05f at step %d!' %
                        (loss, step), slack=True)
                    raise Exception('Loss Exploded')

                if step % args.summary_interval == 0:
                    log('Writing summary at step: %d' % step)
                    summary_writer.add_summary(sess.run(stats), step)

                if step % args.checkpoint_interval == 0:
                    log('Saving checkpoint to: %s-%d' %
                        (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)
                    log('Saving audio and alignment...')
                    input_seq, lpc_targets, alignment = sess.run(
                        [model.inputs[0], model.lpc_outputs[0], model.alignments[0]])
                    plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                                        info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
                    np.save(os.path.join(log_dir, 'step-%d-lpc.npy' %
                                         step), lpc_targets)
                    log('Input: %s' % sequence_to_text(input_seq))

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('.'))
    parser.add_argument('--input', default='training/train.txt')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument(
        '--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=bool, default=True,
                        help='Global step to restore from checkpoint.')
    parser.add_argument('--restore_decoder', type=bool,
                        default=False, help='if set, restore the decoder weights')
    parser.add_argument('--summary_interval', type=int, default=100,
                        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints.')
    parser.add_argument(
        '--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int,
                        default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--git', action='store_true',
                        help='If set, verify that the client is clean.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
    hparams.parse(args.hparams)
    train(log_dir, args)


if __name__ == '__main__':
    main()
