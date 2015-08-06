# -*- coding: utf-8 -*-
__author__ = 'tan'

import os
import logging
import codecs
import pickle
import numpy as np

class HMMModel:

    def __init__(self, N, M, PI, AA, BB):
        self.n = N
        self.m = M
        self.pi = PI
        self.B = BB
        self.A = AA


    def viterbi(self, T, O):
        '''
        下标都是从0开始的
        :param T:
        :param O:
        :return:
        '''
        # 初始化

        delta = np.zeros((T, self.n))
        psi = np.zeros((T, self.n))

        for i in range(self.n):
            delta[0][i] = self.pi[i]*self.B[i][O[1]]
            psi[0][i] = 0

        # 递推
        for t in range(1, T):
            for i in range(self.n):
                maxDelta = 0.0
                index = 1
                for j in range(self.n):
                    if maxDelta < delta[t-1][j] * self.A[j][i]:
                        maxDelta = delta[t-1][j] * self.A[j][i]
                        index = j

                delta[t][i] = maxDelta * self.B[i][O[t]]
                psi[t][i] = index

        # 终止

        prob = 0
        path = [0 for _ in range(T)]
        path[T-1] = 1
        for i in range(self.n):
            if prob < delta[T-1][i]:
                prob = delta[T-1][i]
                path[T-1] = i

        # 最优路径回溯
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1][path[t+1]]

        return path, prob, delta, psi


class HMMSegment:

    def __init__(self, dictfile="data/dict.utf8.txt"):
        '''

        :param dictfile: 词汇文件名
        :return:
        '''
        self.word2idx = {}
        self.idx2word = {}
        self.hmmmodel = None
        self.outfile = dictfile
        self.inited = False

    def build_dict_file(self, filename):
        f = codecs.open(filename, "rb", encoding="utf-8")
        idx = 1
        words = {}
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            idx += 1
            if idx % 100 == 0:
                print("read %d lines" % idx)

            ws = line.split()
            for word in ws:
                for _, w in enumerate(word):
                    if w not in words:
                        words[w] = 0
                    words[w] += 1

        f.close()

        dicts = sorted(words.items(), key=lambda d:d[1], reverse=True)

        print("writing the words in to file {}".format(self.outfile))
        dictfile = codecs.open(self.outfile, "wb", encoding="utf-8")
        for d in dicts:
            dictfile.write("%s\t%d\n" % (d[0], d[1]))
        dictfile.close()

    def init_paramater(self, load=False, save=False):
        '''
        '''
        if load == True:
            self.word2idx = pickle.load(open("data/word2idx.pkl", "rb"))
            self.idx2word = pickle.load(open("data/idx2word.pkl", "rb"))
        else:
            f = codecs.open(self.outfile, "rb", encoding="utf-8")
            for idx, line in enumerate(f):
                word, _ = line.strip().split("\t")
                self.word2idx[word] = idx + 1
                self.idx2word[idx+1] = word
            f.close()

            if save:
                pickle.dump(self.word2idx, open("data/word2idx.pkl", "wb"))
                pickle.dump(self.idx2word, open("data/idx2word.pkl", "wb"))

    def init_model(self, trainfile=None, load=False, save=False):

        self.init_paramater(load, save)

        if load:
            A = pickle.load(open("data/A.pkl", "rb"))
            B = pickle.load(open("data/B.pkl", "rb"))
            PI = pickle.load(open("data/PI.pkl", "rb"))

        else:
            f = codecs.open(trainfile, "rb", encoding="utf-8")
            lines = f.readlines()
            f.close()

            PI, A, B = self.init_A_B_PI(lines)

            if save:
                pickle.dump(A, open("data/A.pkl", "wb"))
                pickle.dump(B, open("data/B.pkl", "wb"))
                pickle.dump(PI, open("data/PI.pkl", "wb"))

        self.hmmmodel = HMMModel(4, len(self.word2idx), PI, A, B)

    def init_A_B_PI(self, lines):
        '''
         * count matrix:
         *   ALL B M E S
         * B *   * * * *
         * M *   * * * *
         * E *   * * * *
         * S *   * * * *
         *
         * NOTE:
         *  count[2][4] is the total number of complex words
         *  count[3][4] is the total number of single words
        :return:
        :param lines:
        :return:
        '''

        print("Init A B PI paramaters")
        last = ""
        countA = np.zeros((4, 5))
        numwords = len(self.word2idx)
        print(numwords)
        countB = np.zeros((4, numwords+1))

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            phrase = line.split(" ")
            for word in phrase:
                # print(word)
                word = word.strip()
                num = len(word)
                #只有一个单词 S 3
                if num == 1:
                    #countB
                    countB[3][self.word2idx[word]] += 1
                    countB[3][0] += 1

                    ########################
                    countA[3][4] += 1 # 单个词的个数
                    #统计转移值
                    if last != "":
                        #单独词转移过来 S -> S
                        if len(last) == 1:
                            countA[3][3] += 1
                        #是从词尾转移过来 E-> S
                        else:
                            countA[2][3] += 1
                else:
                    countA[2][4] += 1 # 多个词的个数
                    countA[0][4] += 1 # B->任意 统计
                    if num > 2:
                        countA[0][1] += 1 # B -> M
                        countA[1][4] += num - 2 # 统计M转移的个数
                        if num > 3: # M-> M
                            countA[1][1] += num - 3
                        countA[1][2] += 1 # M->E
                    else:
                        countA[0][2] += 1 # B->E

                    if last != "":
                        if len(last) == 1:
                            countA[3][0] += 1 # S-> B
                        else:
                            countA[2][0] += 1 # E -> B

                    ###countB 用于计算B矩阵
                    for idx, w in enumerate(word):
                        if idx == 0:
                            countB[0][self.word2idx[word[idx]]] += 1
                            countB[0][0] += 1
                        elif idx == num - 1:
                            countB[2][self.word2idx[word[idx]]] += 1
                            countB[2][0] += 1
                        else:
                            countB[1][self.word2idx[word[idx]]] += 1
                            countB[1][0] += 1

                last = word

        countA[2][0] += 1 # 最后一个E 设为E->B

        print("The count matrix is:")
        print(countA)
        # print(countB)

        A = np.zeros((4, 4))
        PI = np.array([0.0] * 4)

        B = np.zeros((4, numwords+1))

        allwords = countA[2][4] + countA[3][4]

        PI[0] = countA[0][4] / allwords
        PI[3] = countA[3][4] / allwords

        for i in range(4):
            for j in range(4):
                A[i][j] = countA[i][j] / countA[i][4]

            for j in range(1, numwords+1):
                B[i][j] = (countB[i][j] + 1) / countB[i][0]

        print("A and PI B is ")
        print(PI)
        print(A)
        # print(B)

        return PI, A, B

    def segment_sent(self, sentence):

        if not self.inited:
            self.init_model(load=True)

        O = []
        for w in sentence:
            w = w.strip()
            if len(w) == 0:
                continue
            if w not in self.word2idx:
                num = len(self.word2idx)+1
                self.word2idx[w] = num
                self.idx2word[num] = w
                #初始化未登录词的概率
                self.hmmmodel.B = np.column_stack((self.hmmmodel.B, np.array([0.3, 0.3, 0.3, 0.1])))
            O.append(self.word2idx[w])

        T = len(O)

        path, prob, delta, psi = self.hmmmodel.viterbi(T, O)
        result = ""
        for idx, p in enumerate(path):
            # print(self.idx2word[O[idx]], end="")
            result += self.idx2word[O[idx]]
            if p == 2 or p == 3:
                # print("/ ", end="")
                result +="/ "
        # print("")
        # print(path)
        # print(prob)
        return result

    def cut_sentence_new(self, content):
        start = 0
        i = 0
        sents = []
        punt_list = ',.!?:;~，。！？：；～'
        for word in content:
            if word in punt_list and token not in punt_list: #检查标点符号下一个字符是否还是标点
                sents.append(content[start:i+1])
                start = i+1
                i += 1
            else:
                i += 1
                token = list(content[start:i+2]).pop()  # 取下一个字符
        if start < len(content):
            sents.append(content[start:])
        return sents

    def segment_sentences(self, content):
        result = ""
        sentences = self.cut_sentence_new(content)
        for sent in sentences:
            # print(sent)
            result  += self.segment_sent(sent)
            # print(tmp)
            # result += tmp

        return result



if __name__ == "__main__":
    traingfile = "D:\\research\\icwb2-data\\training\\pku_training.utf8"

    hmmseg = HMMSegment()
    # hmmseg.init_model(traingfile, save=True)
    hmmseg.init_model(traingfile, load=True)
    sent = "哈尔滨工业大学，以下是我用自己实现的算法,hello"
    # hmmseg.segment_sent(sent)
    two = "辛辛苦苦做的PPT光保存是不够哒，做个加密才是双保险啦。"
    # hmmseg.segment_sent(two)

    content = u'''辛辛苦苦做的PPT光保存是不够哒，做个加密才是双保险啦。!你只需在保存文件时点击弹出框的“工具”，
        选择“常规选项”，然后设置“修改权限密码”，就能防止PPT文档被人修改咯~
        另外还可将PPT存为PPS格式，这样只要双击文件即可直接播放幻灯片，一举多得！'''
    print(hmmseg.segment_sentences(content))
    # hmmseg.build_dict_file(traingfile)

    # lines = ["abc a ab", "abc a a ab abcd"]
    # hmmseg.init_A_B_PI(lines)
    # hmmseg.init_B(lines)

