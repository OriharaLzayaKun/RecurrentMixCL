import stanza           # 导入stanza库，用于自然语言处理
import spacy             # 导入spacy库，用于命名实体识别
import string
from nltk import word_tokenize      # 从nltk库中导入word_tokenize函数，用于分词
import numpy as np          # 导入numpy库，用于数值计算
from tqdm import tqdm       # 导入tqdm库，用于显示进度条
import json
from transformers import pipeline
import sys
import torch                # 导入torch库，用于深度学习计算


class MixUp:
    def __init__(self):
        #初始化 MixUp 类的实例，加载 Spacy NER 模型，并初始化缓存。
        self.nlp = None      # 初始化nlp为None，用于加载stanza处理器
        self.ner = spacy.load("en_core_web_sm") # 加载spacy的小型英文模型，用于命名实体识别
        self.bert = None     # 初始化bert为None
        self.cache = {}      # 初始化cache为空字典，用于缓存结果

    def get_spans(self, node, layer=0):
        '''
        获取句法树片段
        功能: 递归地获取句法树节点的 span（文本和标签）。
        实现: 如果是叶子节点，返回节点文本；否则递归处理子节点，拼接结果。
        '''
        if layer > 996:     # 避免递归层次过深
            return '', []
        if node.is_leaf():  # 如果节点是叶子节点，返回节点的字符串和空列表
            return str(node), []
        res = [self.get_spans(child, layer=layer + 1) for child in node.children]   # 递归获取子节点的span
        head = [child[0] for child in res]      # 获取子节点的字符串
        spans = [child[1] for child in res]     # 递归获取子节点的span
        if node.label not in ['ROOT', 'S']:     # 如果节点不是ROOT或S，返回节点字符串和子节点的span
            return ' '.join(head), [[node.label, ' '.join(head)]] + sum(spans, [])  # 返回拼接的字符串和span列表
        return ' '.join(head), sum(spans, [])   # 返回拼接的字符串和span列表

    def spanning(self, text):
        '''
        MixUp.spanning("Barack Obama was born in Hawaii.")
        输出 ： Barack Obama was born in Hawaii .
              [['NP', 'Barack Obama'], ['VP', ' born in Hawaii'], ['PP', ' in Hawaii']]
        '''
        if self.nlp is None:
            # 加载stanza处理器
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False, use_gpu=True)
        text = ' '.join(text.split()[:64])      # 限制文本长度为64个词
        key = text
        if key in self.cache:   # 如果缓存中有处理结果，直接返回
            return self.cache[key]
        spans = [self.get_spans(sent.constituency, layer=0) for sent in self.nlp(text).sentences] # 获取每个句子的句法树片段
        sub = sum([sent[1] for sent in spans], [])   # 合并所有的span
        sub = [s for s in sub if s[1] not in string.punctuation]  # 去掉标点符号的span
        text = ' '.join([sent[0] for sent in spans])    # 拼接处理后的文本
        length = len(text.split())   # 获取文本的长度
        sub = [s for s in sub if max(length * 0.1, 1) < len(s[1].split()) < length * 0.5]  # 筛选合适长度的span
        sub = [[s[0], ' ' + s[1]] if ' ' + s[1] in text else s for s in sub]  # 调整span的格式
        self.cache[key] = (text, sub)  # 缓存处理结果
        torch.cuda.empty_cache()       # 清空CUDA缓存
        return text, sub               # 返回处理过的文本及其语法结构信息

    def nering(self, text):
        '''
        nering("Barack Obama was born in Hawaii.")
        输出 ：[['PERSON', 'Barack Obama'], ['GPE', ' Hawaii']]        #'GPE' : Countries, cities, states
        '''
        # 使用spaCy进行命名实体识别，返回识别结果及其标签
        return [[ent.label_, ' ' + ent.text if ' ' + ent.text in text else ent.text] for ent in self.ner(text).ents]

    def span_mix(self, text1, text2, type_matter=True, cached=False):
        # 混合两个文本的句法树片段
        if cached:  # 如果使用缓存
            text1, sub1 = text1
            sub2 = text2
        else:       # 否则重新处理文本
            text1, sub1 = self.spanning(text1)
            text2, sub2 = self.spanning(text2)
        if len(sub1) == 0 or len(sub2) == 0:    # 如果任一文本没有span，返回原文本
            return [text1], [1]
        type2 = [s[0] for s in sub2]            # 获取第二个文本的span类型
        common_type = [s[0] for s in sub1 if s[0] in type2]     # 获取公共的span类型
        if type_matter:
            if len(common_type) == 0:   # 如果没有公共类型，返回原文本
                return [text1], [1]
            chosen_type = np.random.choice(common_type)     # 随机选择一个公共类型
            chosen_sub1 = np.random.choice([s[1] for s in sub1 if s[0] == chosen_type])  # 随机选择对应类型的span
            chosen_sub2 = np.random.choice([s[1] for s in sub2 if s[0] == chosen_type])  # 随机选择对应类型的span
        else:
            chosen_sub1 = np.random.choice([s[1] for s in sub1])    # 随机选择任何类型的span
            chosen_sub2 = np.random.choice([s[1] for s in sub2])    # 随机选择任何类型的span
        text1 = text1.split(chosen_sub1)    # 拆分文本
        text1 = [text1[0], chosen_sub2, chosen_sub1.join(text1[1:])]    # 替换为新的span
        sign = [1, -1, 1]    # 标记替换的部分
        text1, sign = zip(*[[a, b] for a, b in zip(text1, sign) if len(a) > 0])  # 移除空段
        return list(text1), list(sign)  # 返回替换后的文本及其符号列表

    def ent_mix(self, text1, text2, ratio=1, type_matter=False, cached=False):
        # 混合两个文本的命名实体
        if cached:
            ent1 = text1    # 如果使用缓存
            ent2 = text2
        else:                # 否则重新处理文本
            text1 = ' '.join(word_tokenize(text1))
            text2 = ' '.join(word_tokenize(text2))
            ent1 = self.nering(text1)
            ent2 = self.nering(text2)
        if type_matter:     # 如果考虑类型
            type2 = [s[0] for s in ent2]    # 获取第二个文本的span类型
            ent1 = [s for s in ent1 if s[0] in type2]    # 获取公共的span类型
        num = max(int(len(ent1) * ratio), 1)
        np.random.shuffle(ent1)
        text1 = [text1]
        sign = [1]
        for ent in ent1[:num]:
            new_text = []
            new_sign = []
            for seg, s in zip(text1, sign):
                if s < 0 or ent[1] not in seg:
                    new_text.append(seg)
                    new_sign.append(s)
                    continue
                seg = seg.split(ent[1])
                np.random.shuffle(ent2)
                if type_matter:
                    replace = [x for x in ent2 if x[0] == ent[0] and x[1] != ent[1]][0][1]
                else:
                    replace = [x for x in ent2 if x[1] != ent[1]][0][1]
                seg = [seg[0], replace, ent[1].join(seg[1:])]
                new_text.extend(seg)
                new_sign.extend([1, -1, 1])
            text1 = new_text
            sign = new_sign
        text1, sign = zip(*[[a, b] for a, b in zip(text1, sign) if len(a) > 0])
        return list(text1), list(sign)


def build_cache(data):
    # 构建知识缓存
    model = MixUp()
    knowledge_cache = []
    for example in tqdm(data):
        if example['title'] == 'no_passages_used':   # 跳过无用的示例
            continue
        collect = []    # 初始化收集列表
        text = example['title'] + ' ' + example['checked_sentence'] # 合并标题和句子
        collect.append(model.nering(text))
        for k in example['knowledge']:      # 跳过与示例相同的知识
            for s in example['knowledge'][k]:
                if k == example['title'] or s == example['checked_sentence']:
                    continue
                text = k + ' ' + s  # 合并知识标题和内容
                collect.append(model.nering(text))   # 对知识文本进行命名实体识别并收集结果
        knowledge_cache.append(collect)     # 添加收集结果到知识缓存
    return knowledge_cache      # 返回知识缓存


def main():
    data = json.load(open('dataset/wizard/train.json'))
    model = MixUp()
    knowledge_cache = []
    data = [[i, x] for i, x in enumerate(data)]       # 添加索引
    data.sort(key=lambda x: x[1]['chosen_topic'])     # 按主题排序

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True)  # 添加命令行参数
    args = parser.parse_args()
    ids = int(args.id)  # 获取id参数
    processes = 8   # 设置进程数
    length = len(data) // processes + 1   # 计算每个进程处理的数据量
    data = data[ids * length:(ids + 1) * length]  # 切分数据

    for index, example in tqdm(data):   # 逐个处理数据
        if example['title'] == 'no_passages_used':   # 合并标题和检查句子
            continue
        collect = []
        text = example['title'] + ' ' + example['checked_sentence']
        collect.append(model.nering(text))  # 提取命名实体
        for k in example['knowledge']:      # 处理知识段
            for s in example['knowledge'][k]:
                if k == example['title'] or s == example['checked_sentence']:
                    continue
                text = k + ' ' + s
                collect.append(model.nering(text))
        knowledge_cache.append([index, collect])    # 添加到缓存中
    # knowledge_cache = mp(build_cache, data, processes=1)
    json.dump(knowledge_cache, open(f'new_tmp/ner{ids}.json', 'w'))     # 保存缓存


def ner_main():
    import sys
    # 并行处理输入数据，生成命名实体识别的缓存文件。
    sys.path += ['./']
    from utils.mp import mp
    data = json.load(open('dataset/wizard/train.json'))
    knowledge_cache = mp(build_cache, data, processes=40)
    json.dump(knowledge_cache, open(f'new_tmp/ner.json', 'w'))


def test():
    # 测试 MixUp 类的 spanning 方法，用一个示例文本进行测试
    model = MixUp()
    text = 'Science fiction (often shortened to SF or sci-fi) is a \"genre\" of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.'
    text = text + text
    print(model.spanning(text))


if __name__ == '__main__':
    # test()
    main()
    # ner_main()