# -*- encoding:utf-8 -*-
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel

# 配置
class Config(object):
    # 词向量维度
    embed_size = 50
    # 批量数据大小
    batch_size = 64
    # 时间步骤数
    num_steps = 10
    # 隐层大小
    hidden_size = 100
    # 最大迭代次数
    max_epochs = 16
    # 用于获取迭代最优点
    early_stopping = 2
    # 弃权值
    dropout = 0.9
    # 学习率
    lr = 0.001

    # 设定GPU的性质,允许将不能在GPU上处理的部分放到CPU
    # 设置log打印
    cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # 只占用20%的GPU内存
    cf.gpu_options.per_process_gpu_memory_fraction = 0.2


# RNN语言模型
class RNNLM_Model(LanguageModel):
    def __init__(self, config):
        # 载入配置
        self.config = config
        # 载入数据
        self.load_data()
        # 加入输入的容器
        self.add_placeholders()
        # 得到输入的词向量
        self.inputs = self.add_embedding()
        # 得到rnn的输出
        self.rnn_outputs = self.add_model(self.inputs)
        # 得到每个rnn输出的softmax层输出
        self.outputs = self.add_projection(self.rnn_outputs)
        # 得到预测值,shape=(num_step, batch_size, len(vocab))
        self.predictions = [tf.nn.softmax(tf.cast(o, tf.float64)) for o in self.outputs]

        # outputs原来为长度为num_step的List,每个元素为(batch_size, len(vocab))
        # 要把outputs转化为(num_step*batch_size, len(vocab))
        # 即每一列都是以num_step为单位,共batch_size个num_step
        output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])

        # 计算损失
        self.calculate_loss = self.add_loss_op(output)
        # 得到训练节点
        self.train_step = self.add_training_op(self.calculate_loss)

    # 加载数据
    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data."""
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        # 存的是词的id,encoded_train里面是大量句子组成的一维list,每个句子用<eos>来分隔
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('train')],
            dtype=np.int32)
        # 和train一样
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
            dtype=np.int32)
        # 和train一样
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('test')],
            dtype=np.int32)
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    # 加入placeholders
    def add_placeholders(self):
        # (None, num_steps)
        self.input_placeholder = tf.placeholder(tf.int32,
                                    [None, self.config.num_steps], 'input')
        # (None, num_steps)
        self.labels_placeholder = tf.placeholder(tf.int32,
                                    [None, self.config.num_steps], 'labels')
        # (float)
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

    # 将输入的ID转为对应的词向量
    def add_embedding(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('Embedding'):
                self.embedding = tf.get_variable('embedding',
                                [len(self.vocab), self.config.embed_size])
                # 经过lookup后的inputs的shape为(batch_size, num_steps, embed_size)
                inputs = tf.nn.embedding_lookup(self.embedding, self.input_placeholder)
                # 将inputs转置,把维度0和维度1交换
                inputs = tf.transpose(inputs, (1,0,2))
                inputs = tf.reshape(inputs, [-1, self.config.embed_size])
                # 在维度0上进行分割,将数据分割成num_steps份,得到inputs列表
                inputs = tf.split(0, self.config.num_steps, inputs)
                # (num_steps, batch_size, embed_size)
                return inputs

    # 模型
    def add_model(self, inputs):
        # 输入弃权
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(input, self.dropout_placeholder) for input in inputs]

        # RNN层循环
        with tf.variable_scope('RNN') as scope:
            self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            rnn_outputs = []
            # 对每个时间步骤的输入进行处理
            for tstep, current_input in enumerate(inputs):
                if tstep>0:
                    scope.reuse_variables()
                RNN_H = tf.get_variable('RNN_H', [self.config.hidden_size, self.config.hidden_size])
                RNN_I = tf.get_variable('RNN_I', [self.config.embed_size, self.config.hidden_size])
                RNN_b = tf.get_variable('b', [self.config.hidden_size])
                state = tf.sigmoid(tf.matmul(current_input, RNN_I)+tf.matmul(state, RNN_H)+RNN_b)
                rnn_outputs.append(state)
            # 最后的一个状态即rnn的最后一个时间步骤的输出
            self.final_state = rnn_outputs[-1]

        # 输出弃权
        with tf.variable_scope('OutputDropout') as scope:
            rnn_outputs = [tf.nn.dropout(rnn_output, self.dropout_placeholder) for rnn_output in rnn_outputs]

        return rnn_outputs

    # 投影层
    def add_projection(self, rnn_outputs):
        with tf.variable_scope('Projection'):
            U = tf.get_variable('U', [self.config.hidden_size, len(self.vocab)], tf.float32)
            proj_b = tf.get_variable('Proj_bias', [len(self.vocab)])
            outputs = [tf.matmul(output, U)+proj_b for output in rnn_outputs]
            return outputs

    # 损失节点
    def add_loss_op(self, output):
        # (batch_size, num_steps)
        all_ones = [tf.ones([self.config.num_steps*self.config.batch_size])]
        # 序列的交叉熵损失,即整个长度为num_step的序列的损失
        # sequence_loss各个参数说明请看源码
        # sequence_loss返回的是平均对数困惑度(log-perplexity),即平均交叉熵
        cross_entropy = sequence_loss(
            [output], [tf.reshape(self.labels_placeholder, [-1])], all_ones
        )
        # 得到所有的误差.如果要考虑L2正则化,可以考虑将RNN_I和RNN_H以及U加入total_loss
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    # 增加训练节点,用的Adam算法
    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return opt

    # 每次迭代训练
    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        # 表示不训练
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        # 总的迭代步骤
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        # 总的损失列表
        total_loss = []
        # 每次跑完所有数据后要初始化状态
        state = self.initial_state.eval()
        # 对每个批量数据
        for step, (x, y) in enumerate(
            ptb_iterator(data, config.batch_size, config.num_steps)):
            # 获取feed_dict
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp}
            # 得到损失和最后的状态
            loss, state, _ = session.run(
                [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)

            # 如果让过程显示可见
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        # 指数化损失,相当于困惑度,当然也可以不用指数化,只是为了容易看出变化
        return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                    stop_length=100, stop_tokens=None, temp=1.0):

    # 初始化状态需要在每次重新输入句子的时候初始化
    state = model.initial_state.eval()
    # 得到starting_text对应ID序列
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    # 最大长度为stop_length
    for i in xrange(stop_length):
        # 只输入最后一个词的ID
        # 初始化模型状态
        # dropout=1
        feed = {model.input_placeholder: [tokens[-1:]],
                model.initial_state: state,
                model.dropout_placeholder: 1}
        # 得到输出结果(state)和预测的词的ID,predictions[-1]的大小为(batch_size, len(vocab))
        state, y_pred = session.run(
            [model.final_state, model.predictions[-1]], feed_dict=feed)
        # y_pred[0]的大小为len(vocab)
        # 得到概率最大的词的ID,作为下一个词
        next_word_idx = sample(y_pred[0], temperature=temp)
        # 加入到句子列表汇总
        tokens.append(next_word_idx)
        # 如果遇到停止词,则跳出
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
            break
    # 形成整个句子
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output

# 生成一个句子
def generate_sentence(session, model, config, *args, **kwargs):
    # 很方便的生成一个句子
    return generate_text(session, model, config, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
    # 用于训练模型的配置
    config = Config()

    # 用于生成词语的配置
    gen_config = deepcopy(config)
    gen_config.batch_size = gen_config.num_steps = 1

    # 训练模型和生成语言模型
    with tf.variable_scope('RNNLM') as scope:
        # 训练模型
        model = RNNLM_Model(config)
        # 使用原有的变量
        scope.reuse_variables()
        # 再建立一个生成语言模型,其变量和训练模型共享
        gen_model = RNNLM_Model(gen_config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session(config=config.cf) as session:
        # 最优解
        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(init)
        # 迭代循环
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            # 获得训练损失
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step)
            # 验证集的损失
            valid_pp = model.run_epoch(session, model.encoded_valid)
            print 'Training perplexity: {}'.format(train_pp)
            print 'Validation perplexity: {}'.format(valid_pp)
            # 更新最优迭代步骤
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
            # early_stopping
            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)

        # 加载变量
        saver.restore(session, 'ptb_rnnlm.weights')
        # 测试
        test_pp = model.run_epoch(session, model.encoded_test)
        print '=-=' * 5
        print 'Test perplexity: {}'.format(test_pp)
        print '=-=' * 5

        # 造句
        starting_text = 'in palo alto'
        while starting_text:
            print ' '.join(generate_sentence(
                session, gen_model, gen_config, starting_text=starting_text, temp=1.0))
            starting_text = raw_input('请输入起头文> ')




if __name__ == '__main__':
    test_RNNLM()