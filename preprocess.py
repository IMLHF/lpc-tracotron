import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import ljspeech
from hparams import hparams


def preprocess_ljspeech(args):
    in_dir = os.path.join(args.base_dir, 'DATA', 'LJSpeech-1.1')
    out_dir = os.path.join(args.base_dir, 'DATA', 'LJSpeech-1.1', args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(
        in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

# def preprocess_biaobei_10000sentences(args):
#     in_dir = os.path.join(args.base_dir, 'DATA', 'BZNSYP')
#     out_dir = os.path.join(args.base_dir, 'DATA', 'BZNSYP', args.output)
#     os.makedirs(out_dir, exist_ok=True)
#     metadata = ljspeech_and_bznsyp.build_from_path(
#         in_dir, out_dir, args.num_workers, tqdm=tqdm)
#     write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
        frames = sum([m[1] for m in metadata])
        hours = frames * hparams.frame_shift_ms / (3600 * 1000)
        print('Wrote %d utterances, %d frames (%.2f hours)' %
              (len(metadata), frames, hours))
        print('Max input length:  %d' % max(len(m[2]) for m in metadata))
        print('Max output length: %d' % max(m[1] for m in metadata))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('.'))
    parser.add_argument('--output', default='training')
    parser.add_argument('--dataset', default='bznsyp',
                        choices=['bznsyp', 'ljspeech', 'blizzard', 'thchs30'])
    # TODO: test 'blizzard', 'thchs30'
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    # parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    preprocess_fn = {
        # 'bznsyp': preprocess_biaobei_10000sentences,
        'ljspeech': preprocess_ljspeech
    }[args.dataset]

    preprocess_fn(args)


if __name__ == "__main__":
    main()
