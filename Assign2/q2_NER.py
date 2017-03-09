# —*- encoding:utf8 -*-
import sys
import time
import os

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel

# ————————————————————
"""
这一节的目标是建立一个NER分类网络,例子里共有5种实体类别.
实现细节:
先建立一个模型网络,之后用已经标注好的数据进行训练.
输入为窗口大小的词组组成的词向量的连接,输出为窗口中心词的实体类型
比如每个词的embed_size=50,窗口大小为3,则窗口词向量大小为150
"""
# ————————————————————

# 所有参数用一个Config类表示
class Config():
    # 词向量维度
    embed_size = 50
    # 批量数据大小
    batch_size = 64
    # 标签种类数
    label_size = 5
    # 隐层大小
    hidden_size = 100
    # 最大迭代次数
    max_epochs = 24
    # 用于获取迭代最优点
    early_stopping = 2
    # 弃权值
    dropout = 0.9
    # 学习率
    lr = 0.001
    # l2正则化值
    l2 = 0.001
    # 窗口大小
    window_size = 3

    # 设定GPU的性质,允许将不能在GPU上处理的部分放到CPU
    # 设置log打印
    cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # 只占用20%的GPU内存
    cf.gpu_options.per_process_gpu_memory_fraction = 0.2

# NER模型
class NERModel(LanguageModel):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        window = self.add_embedding()
        y = self.add_model(window)
        self.predictions = tf.nn.softmax(y)
        onehot_pred = tf.argmax(self.predictions, 1)
        real_pred = tf.equal(onehot_pred, tf.argmax(self.labels_placeholder, 1))
        self.correct_predictions = tf.reduce_sum(tf.cast(real_pred, tf.int32))
        self.loss = self.add_loss_op(y)
        self.train_op = self.add_training_op(self.loss)


    # 加载标注的数据
    def load_data(self, debug=False):
        self.wv, word2num, num2word = ner.load_wv(
            'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
        self.wv = self.wv.astype(np.float32)
        tags = ["O", "LOC", "MISC", "ORG", "PER"]
        self.num2tag = dict(enumerate(tags))
        tag2num = dict(zip(self.num2tag.values(), self.num2tag.keys()))
        docs = du.load_dataset('data/ner/train')
        self.X_train, self.y_train = du.docs_to_windows(
            docs, word2num, tag2num, wsize=self.config.window_size
        )
        if debug:
            self.X_train = self.X_train[:1024]
            self.y_train = self.y_train[:1024]

        docs = du.load_dataset('data/ner/dev')
        self.X_dev, self.y_dev = du.docs_to_windows(
            docs, word2num, tag2num, wsize=self.config.window_size)
        if debug:
            self.X_dev = self.X_dev[:1024]
            self.y_dev = self.y_dev[:1024]

        docs = du.load_dataset('data/ner/test.masked')
        self.X_test, self.y_test = du.docs_to_windows(
            docs, word2num, tag2num, wsize=self.config.window_size)

    # 加入输入的容器,和feed对应
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
                tf.int32, [None, self.config.window_size], 'input')
        self.labels_placeholder = tf.placeholder(
                tf.float32, [None, self.config.label_size], 'labels')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

    # 创建feed_dict
    def create_feed_dict(self, input_batch, dropout, label_batch=None):
        feed_dict = {self.input_placeholder:input_batch}
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch

        return feed_dict

    # 将输入的样本变为词向量
    # 输入shape为[batch_size, window_size],输出shape为[batch_size, embed_size*window_size]
    def add_embedding(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('embedding'):
                # 使用预训练词向量,wv类型若为float,则必须为float32,保持统一
                # embedding = tf.get_variable('Embedding', initializer=self.wv)
                # 不使用预训练词向量
                embedding = tf.get_variable('embed', [len(self.wv), self.config.embed_size])
                window = tf.nn.embedding_lookup(embedding, self.input_placeholder)
                window = tf.reshape(window, [-1, self.config.embed_size*self.config.window_size])
                return window

    # 建立模型
    # 所有变量使用xavier_weight_init来初始化,防止神经元过早饱和
    def add_model(self, window):
        with tf.device('/cpu:0'):
            # 第一层的网络
            with tf.variable_scope('Layer1', initializer=xavier_weight_init()):
                W = tf.get_variable('w1',
                    [self.config.embed_size*self.config.window_size, self.config.hidden_size])
                b1 = tf.get_variable('b1', [self.config.hidden_size])
                h = tf.nn.tanh(tf.matmul(window, W)+b1)
                # 加入l2正则化项
                if self.config.l2:
                    # 加入到一个名为'total_loss'的变量里
                    tf.add_to_collection('total_loss', 0.5*self.config.l2*tf.nn.l2_loss(W))

            # 第二层的网络
            with tf.variable_scope('Layer2', initializer=xavier_weight_init()):
                U = tf.get_variable('w2',
                    [self.config.hidden_size, self.config.label_size])
                b2 = tf.get_variable('b2', [self.config.label_size])
                y = tf.matmul(h, U) + b2
                # 加入l2正则化项
                if self.config.l2:
                    tf.add_to_collection('total_loss', 0.5*self.config.l2*tf.nn.l2_loss(U))

            # 输出使用dropout
            output = tf.nn.dropout(y, self.dropout_placeholder)
            return output

    # 加入损失节点
    def add_loss_op(self, pred):
        # 交叉熵损失
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.labels_placeholder))
        # 加入ce到total_loss
        tf.add_to_collection('total_loss', ce)
        # 总损失,add_n对多个形状一样的tensor求和
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    # 加入训练节点
    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return opt

    # 每个循环需要做的事情
    def run_epoch(self, sess, input_data, input_labels, shuffle=True, verbose=True):
        # 所有的数据
        orig_X, orig_y = input_data, input_labels
        dp = self.config.dropout
        # 记录各个循环的误差
        total_loss = []
        # 记录所有训练数据中预测正确的数量
        total_correct_examples = 0
        # 记录所有已经处理的数据
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size
        # 对每份批量数据
        for step, (x, y) in enumerate(
            data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                label_size=self.config.label_size, shuffle=shuffle)):
            # 获取feed字典
            feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
            # 运行计算图,获取损失和正确预测个数
            loss, total_correct, _ = sess.run(
                [self.loss, self.correct_predictions, self.train_op],
                    feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)

            # 若可显示,则输出每个循环/每步迭代的结果
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        # 返回平均误差和正确率
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

    # 在已经训练好的模型下进行预测
    # 注意此时的dropout/keep_prob应该等于1
    # 如果y=None,表示我只需要获取X对应的预测值
    # 如果y!=None,表示我要获取X对应的预测值,以及预测值与真实值比较的损失
    def predict(self, session, X, y=None):
        # dropout
        dp = 1
        # 损失
        losses = []
        # 输出结果
        results = []
        # 判定y是否为None
        if np.any(y):
            data = data_iterator(X, y, batch_size=self.config.batch_size,
                    label_size=self.config.label_size, shuffle=False)
        else:
            data = data_iterator(X, batch_size=self.config.batch_size,
                    label_size=self.config.label_size, shuffle=False)

        # 每步循环
        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            # 如果y非空,获取预测值和损失
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run(
                    [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            # 如果y为空,只获取预测值
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            results.extend(predicted_indices)
        # 返回平均损失和预测结果
        return np.mean(losses), results




# 打印困惑矩阵
def print_confusion(confusion, num_to_tag):
    # 从上到下求和
    total_guessed_tags = confusion.sum(axis=0)
    # 从左到右求和
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        # 查准率
        prec = confusion[i, i] / float(total_guessed_tags[i])
        # 查全率
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)

# 返回困惑矩阵(对角线上数据越大越准确,其余位置越大效果越差),大小为5*5
def calculate_confusion(config, predicted_indices, y_indices):
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

# 保存预测值
def save_predictions(predictions, filename):
    with open(filename, "wb") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")


# 测试NER
# 如果要debug,可以设置max_epochs=1会快速迭代
def test_NER():
    # 获取配置
    config = Config()
    # 建立默认图
    with tf.Graph().as_default():
        # 初始化模型
        model = NERModel(config)
        # 初始化变量
        init = tf.initialize_all_variables()
        # 用于存储变量
        saver = tf.train.Saver()

        with tf.Session(config=config.cf) as session:
            # 存储最小损失
            best_val_loss = float('inf')
            # 存储最小损失对应的迭代次数
            best_val_epoch = 0

            session.run(init)
            # 开始迭代
            for epoch in xrange(config.max_epochs):
                print 'Epoch {}'.format(epoch)
                start = time.time()
                # 获取一次迭代的损失和准确率
                train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                        model.y_train)
                # 验证集的损失和预测                                                                           model.y_train)
                val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
                print 'Training loss: {}'.format(train_loss)
                print 'Training acc: {}'.format(train_acc)
                print 'Validation loss: {}'.format(val_loss)
                # 保存最优值
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")
                    # 保存最优时的变量
                    saver.save(session, './weights/ner.weights')
                # 超过early_stopping,跳出循环
                if epoch - best_val_epoch > config.early_stopping:
                    break
                #计算困惑矩阵
                confusion = calculate_confusion(config, predictions, model.y_dev)
                # 打印
                print_confusion(confusion, model.num2tag)
                # 耗时
                print 'Total time: {}'.format(time.time() - start)

            # 读取之前存储的变量到session里
            saver.restore(session, './weights/ner.weights')
            # 测试
            print 'Test'
            print '=-=-='
            print 'Writing predictions to q2_test.predicted'
            _, predictions = model.predict(session, model.X_test, model.y_test)
            # 保存预测值
            save_predictions(predictions, "q2_test.predicted")

if __name__ == "__main__":
    test_NER()