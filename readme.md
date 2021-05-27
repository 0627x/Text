# NPL新闻文本分类

# 一、赛题理解

## 1、赛题概况

通过这道赛题可以引导大家走入自然语言处理的世界，带大家接触NLP的预处理、模型构建和模型训练等知识点。

赛题以自然语言处理为背景，要求选手对新闻文本进行分类，这是一个典型的字符识别问题。

## 2、数据概况

赛题以新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。
赛题数据由以下几个部分构成：训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了匿名处理。

处理后的赛题训练数据如下：

![1622134434490](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622134434490.png)



## 3.评测标准

![1622134537122](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622134537122.png)

## 二、数据分析

~~~
train_df.head()
~~~

 ![1622134920981](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622134920981.png)

```
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())
```

![1622134975075](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622134975075.png)

对新闻句子长度的统计可以得出，每个句子平均由872个字符构成，最短的句子长度为64，最长的句子长度为7125。

```
_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")
```

![1622135114517](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622135114517.png)

```
train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
```

![1622135159035](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622135159035.png)

从统计结果可以看出，赛题的数据集类别分布存在较为不均匀的情况。在训练集中体育类新闻最多，其次是股票类新闻，最少的新闻是星座新闻。

## 三、模型训练

## 1.卷积神经网络

CNN由纽约大学的Yann Lecun于1998年提出，其本质是一个多层感知机，是一种具有局部连接、权值共享等特征的深层前馈神经网络。
一方面减少了权值的数量使得网络易于优化
另一方面降低了模型的复杂度，也就是减小了过拟合的风险

卷积神经网络相比一般神经网络在图像理解中的优点：
同时进行特征提取和分类，使得特征提取有助于特征分类
权值共享可以减少网络的训练参数，使得神经网络结构变得简单，适应性更强

![1622135349807](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622135349807.png)

## 2.word2vec词向量

通过word2vec学习词向量。word2vec模型背后的基本思想是对出现在上下文环境里的词进行预测。对于每一条输入文本，我们选取一个上下文窗口和一个中心词，并基于这个中心词去预测窗口里其他词出现的概率。因此，word2vec模型可以方便地从新增语料中学习到新增词的向量表达，是一种高效的在线学习算法（online learning）。word2vec的主要思路：通过单词和上下文彼此预测
word2vec的主要思路：通过单词和上下文彼此预测，对应的两个算法分别为：Skip-grams (SG)：预测上下文、Continuous Bag of Words (CBOW)：预测目标单词
Word2vec是词嵌入（ word embedding) 的一种。判断一个词的词性，是动词还是名词。用机器学习的思路，我们有一系列样本(x,y)，这里 x 是词语，y 是它们的词性，我们要构建 f(x)->y 的映射，但这里的数学模型 f（比如神经网络、SVM）只接受数值型输入，而 NLP 里的词语，是人类的抽象总结，是符号形式的（比如中文、英文、拉丁文等等），所以需要把他们转换成数值形式，或者说——嵌入到一个数学空间里，这种嵌入方式，就叫词嵌入（word embedding)。

![1622135477039](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622135477039.png)

## 3.TextCNN模型

 TextCNN网络是2014年提出的用来做文本分类的卷积神经网络，由于其结构简单、效果好，在文本分类、推荐等NLP领域应用广泛，我自己在工作中也有探索其在实际当中的应用，今天总结一下。

TextCNN的网络结构
数据预处理
再将TextCNN网络的具体结构之前，先讲一下TextCNN处理的是什么样的数据以及需要什么样的数据输入格式。假设现在有一个文本分类的任务，我们需要对一段文本进行分类来判断这个文本是是属于哪个类别：体育、经济、娱乐、科技等。训练数据集如下示意图：
￼
第一列是文本的内容，第二列是文本的标签。首先需要对数据集进行处理，步骤如下：
\- 分词 中文文本分类需要分词，有很多开源的中文分词工具，例如Jieba等。分词后还会做进一步的处理，去除掉一些高频词汇和低频词汇，去掉一些无意义的符号等。
\- 建立词典以及单词索引 建立词典就是统计文本中出现多少了单词，然后为每个单词编码一个唯一的索引号，便于查找。如果对以上词典建立单词索引，上面的词典表明，“谷歌”这个单词，可以用数字 0 来表示，“乐视”这个单词可以用数字 1 来表示。
\- 将训练文本用单词索引号表示 在上面的单词-索引表示下，训练示例中的第一个文本样本可以用如下的一串数字表示，到这里文本的预处理工作基本全部完成，将自然语言组成的训练文本表示成离散的数据格式，是处理NLP工作的第一步。

TextCNN结构
TextCNN的结构比较简单，输入数据首先通过一个embedding layer，得到输入语句的embedding表示，然后通过一个convolution layer，提取语句的特征，最后通过一个fully connected layer得到最终的输出，整个模型的结构如下图：
￼![1622135887619](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622135887619.png)
上图是论文中给出的视力图，下面分别介绍每一层。
\- embedding layer：即嵌入层，这一层的主要作用是将输入的自然语言编码成distributed representation，具体的实现方法可以参考word2vec相关论文，这里不再赘述。可以使用预训练好的词向量，也可以直接在训练textcnn的过程中训练出一套词向量，不过前者比或者快100倍不止。如果使用预训练好的词向量，又分为static方法和no-static方法，前者是指在训练textcnn过程中不再调节词向量的参数，后者在训练过程中调节词向量的参数，所以，后者的结果比前者要好。更为一般的做法是：不要在每一个batch中都调节emdbedding层，而是每个100个batch调节一次，这样可以减少训练的时间，又可以微调词向量。
\- convolution layer：这一层主要是通过卷积，提取不同的n-gram特征。输入的语句或者文本，通过embedding layer后，会转变成一个二维矩阵，假设文本的长度为|T|，词向量的大小为|d|，则该二维矩阵的大为|T|x|d|，接下的卷积工作就是对这一个|T|x|d|的二维矩阵进行的。卷积核的大小一般设定为n是卷积核的长度，|d|是卷积核的宽度，这个宽度和词向量的维度是相同的，也就是卷积只是沿着文本序列进行的，n可以有多种选择，比如2、3、4、5等。对于一个|T|x|d|的文本，如果选择卷积核kernel的大小为2x|d|，则卷积后得到的结果是|T-2+1|x1的一个向量。在TextCNN网络中，需要同时使用多个不同类型的kernel，同时每个size的kernel又可以有多个。如果我们使用的kernel size大小为2、3、4、5x|d|，每个种类的size又有128个kernel，则卷积网络一共有4x128个卷积核。￼
上图是从google上找到的一个不太理想的卷积示意图，我们看到红色的横框就是所谓的卷积核，红色的竖框是卷积后的结果。从图中看到卷积核的size=1、2、3， 图中上下方向是文本的序列方向，卷积核只能沿着“上下”方向移动。卷积层本质上是一个n-gram特征提取器，不同的卷积核提取的特征不同，以文本分类为例，有的卷积核可能提取到娱乐类的n-gram，比如范冰冰、电影等n-gram；有的卷积核可能提取到经济类的n-gram，比如去产能、调结构等。分类的时候，不同领域的文本包含的n-gram是不同的，激活对应的卷积核，就会被分到对应的类。
\- max-pooling layer：最大池化层，对卷积后得到的若干个一维向量取最大值，然后拼接在一块，作为本层的输出值。如果卷积核的size=2，3，4，5，每个size有128个kernel，则经过卷积层后会得到4x128个一维的向量（注意这4x128个一维向量的大小不同，但是不妨碍取最大值），再经过max-pooling之后，会得到4x128个scalar值，拼接在一块，得到最终的结构—512x1的向量。max-pooling层的意义在于对卷积提取的n-gram特征，提取激活程度最大的特征。
\- fully-connected layer：这一层没有特别的地方，将max-pooling layer后再拼接一层，作为输出结果。实际中为了提高网络的学习能力，可以拼接多个全连接层。 

## 4.HAN

 (HAN)基于层级注意力，在单词和句子级别分别编码并基于注意力获得文档的表示，然后经过Softmax进行分类。其中word encoder的作用是获得句子的表示，可以替换为上节提到的TextCNN和TextRNN，也可以替换为下节中的BERT。

## 5.训练word2vec

```
import logging
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed 
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)
fold_num = 10
data_file = './Text/train_set.csv'
import pandas as pd


def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]

    total = len(labels)

    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


fold_data = all_data2fold(10)
```

## 6.基于TextCNN文本表示

```
word2vec_path = './Text/word2vec.txt'
dropout = 0.15


class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100
        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)
        
        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))

        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, word_ids, extword_ids):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks: sen_num x sent_len
        sen_num, sent_len = word_ids.shape

        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        batch_embed.unsqueeze_(1)  # sen_num x 1 x sent_len x 100

        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            filter_height = sent_len - self.filter_sizes[i] + 1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)  # sen_num x out_channel x filter_height x 1

            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width)
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)  # sen_num x out_channel x 1 x 1 -> sen_num x out_channel

            pooled_outputs.append(pooled)

        reps = torch.cat(pooled_outputs, dim=1)  # sen_num x total_out_channel

        if self.training:
            reps = self.dropout(reps)

        return reps


# build sent encoder
sent_hidden_size = 256
sent_num_layers = 2


class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens
```

```
import time
from sklearn.metrics import classification_report

clip = 5.0
epochs = 1
early_stops = 1
log_interval = 600

test_batch_size = 2
train_batch_size = 2

save_model = './Text/1/cnn.bin'
save_test = './Text/1/cnn.csv'

class Trainer():
    def __init__(self, model, vocab):
        self.model = model
        self.report = True

        self.train_data = get_examples(train_data, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        self.dev_data = get_examples(dev_data, vocab)
        self.test_data = get_examples(test_data, vocab)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = vocab.target_names

        # optimizer
        self.optimizer = Optimizer(model.all_parameters)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training...')
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)

            dev_f1 = self._eval(epoch)

            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(save_model))
        self._eval(self.last_epoch + 1, test=True)
    
    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / log_interval,
                        elapsed / log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses,
                                                                                  during_time))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time
            
            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(save_test, index=False, sep=',')
            else:
                logging.info(
                    '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1,
                                                                              during_time))
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    logging.info('\n' + report)

        return f1

    def batch2tensor(self, batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1

        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels
```

![1622137109488](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622137109488.png)

![1622137122736](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622137122736.png)

## 7.提交结果

![1622137209582](C:\Users\Ccy\AppData\Roaming\Typora\typora-user-images\1622137209582.png)

