import argparse
import os
import re
import numpy as np
import soundfile as sf
from textnorm import get_pinyin
from functools import partial
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
    '美国主持人听到“中国”就插话',
    '勉励乡亲们为过上更加幸福美好的生活继续团结奋斗。',
    '中国基建领域又来了一款“神器”, 哪里不平平哪里',
    '违反中央八项规定精神和廉洁纪律，违规出入私人会所和打高尔夫球',
    '陪审团未能就其盗窃和藏匿文物罪名作出裁决',
    '于美国首都华盛顿国家记者俱乐部召开的新闻发布会上说',
    '杭州市卫健委某直属单位一名拟提副处级干部刘某公示期间，纪检监察组照例对其个人重大事项进行抽查',
    '我国森林面积、森林蓄积分别增长一倍左右，人工林面积居全球第一',
    '打打打打打打打打打打打',
    '卡尔普陪外孙玩滑梯。',
    '假语村言别再拥抱我。',
    '宝马配挂跛骡鞍，貂蝉怨枕董翁榻。',
    '中国地震台网速报,'
    '中国地震台网正式测定,',
    '06月04日17时46分在台湾台东县海域（北纬22.82度，东经121.75度）发生5.8级地震',
    '中国地震台网速报，中国地震台网正式测定：06月04日17时46分在台湾台东县海域（北纬22.82度，东经121.75度）发生5.8级地震',
    '震源深度9千米，震中位于海中，距台湾岛最近约47公里。',
    '刚刚,台湾发生5.8级地震,与此同时,泉州厦门漳州震感明显,',
    '此次台湾地震发生后,许多网友为同胞祈福,愿平安,'
]

text2pinyin = partial(get_pinyin, std=False, pb=True)
sentences = [' '.join(text2pinyin(sent)) for sent in sentences]
print(sentences)


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print(' ')
        print('[{:<10}]: {}'.format('processing', path))
        wav, feature = synth.synthesize(text)
        sf.write(path, wav, 16000)
        np.save(path.replace('.wav', '.npy'), feature)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']= ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
