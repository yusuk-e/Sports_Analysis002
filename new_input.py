# -*- coding:utf-8 -*-

import pdb
from time import time
import datetime as dt
import numpy as np
from scipy.special import gammaln
import matplotlib
#matplotlib.use('Agg') #DIPLAYの設定
import matplotlib.pyplot as plt
import resource
import codecs
import random
from collections import defaultdict
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import commands

#-----variable-----

#--dictionary--
action_dic = {}
team_dic = {}
player_dic = {}

#--data--
elem = namedtuple("elem", "t, re_t, team, p, a, x, y, s")
#絶対時刻, ハーフ相対時間，チームID，アクションID, x座標，y座標, success
D = defaultdict(int)#アクションID付きボール位置データ D[counter]
N = 0
Seq_Team_of = defaultdict(lambda: defaultdict(int))
N_Team1_of = -1
N_Team2_of = -1

#--others--
Stoppage = [5, 7, 11, 12]

xmax = -10 ** 14
xmin = 10 ** 14
ymax = -10 ** 14
ymin = 10 ** 14
tmax = -10 ** 14
tmin = 10 ** 14

period1_start = 0
period1_end = 0
period2_start = 0
period2_end = 0
period3_start = 0
period3_end = 0
period4_start = 0
period4_end = 0

period1_start_re_t = 0
period1_end_re_t = 0
period2_start_re_t = 0
period2_end_re_t = 0
period3_start_re_t = 0
period3_end_re_t = 0
period4_start_re_t = 0
period4_end_re_t = 0

#-----------------


def input():
#--Input--

    global xmax, xmin, ymax, ymin
    global start_time, end_time
    global period1_start, period1_end
    global period2_start, period2_end
    global period3_start, period3_end
    global period4_start, period4_end

    global N

    t0 = time()

    counter = 0
    filename = "processed_metadata.csv"
    fin = open(filename)

    tmp = fin.readline().rstrip("\r\n").split(",")
    A = tmp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period1_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period1_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])
    pdb.set_trace()

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period2_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period2_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period3_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period3_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period4_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period4_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    counter = 0
    temp_player0_dic = {}
    temp_player1_dic = {}
    for row in fin:
        temp = row.rstrip("\r\n").split(",")

        A = temp[0].split(".")
        B = A[0].split(":")
        C = A[1]
        t = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

        A = temp[1].split(".")
        B = A[0].split(":")
        C = A[1]
        re_t = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

        team = int(temp[2])
        if team not in team_dic:
            team_dic[team] = len(team_dic)

        player = int(temp[3])
        if team_dic[team] == 0:
            if player not in temp_player0_dic:
                temp_player0_dic[player] = len(temp_player0_dic)
        elif team_dic[team] == 1:
            if player not in temp_player1_dic:
                temp_player1_dic[player] = len(temp_player1_dic)
        else:
            print "err"

        action = int(temp[4])
        if action not in action_dic:
            action_dic[action] = len(action_dic)

        x = float(temp[5])
        if xmin > x:
            xmin = x
        if xmax < x:
            xmax = x
        
        y = float(temp[6])
        if ymin > y:
            ymin = y
        if ymax < y:
            ymax = y

        s = int(temp[7])

        f = elem(t, re_t, team, player, action, x, y, s)
        D[counter] = f
        counter += 1

    fin.close()
    N = counter - 1
    player_dic[0] = temp_player0_dic
    player_dic[1] = temp_player1_dic

    Make_re_t()#各ピリオド開始からの相対時間を生成
    Reverse_Seq()#後半の攻撃を反転
    print "time:%f" % (time()-t0)

def Make_re_t():
#--各ピリオド開始からの相対時間を生成--
    global period1_start_re_t, period1_end_re_t 
    global period2_start_re_t, period2_end_re_t
    global period3_start_re_t, period3_end_re_t 
    global period4_start_re_t, period4_end_re_t

    period1_start_re_t = 0.0
    period1_end_re_t = period1_end
    period2_start_re_t = 0.0
    period2_end_re_t = period2_end - period2_start + 1
    period3_start_re_t = 0.0
    period3_end_re_t = period3_end - period3_start + 1
    period4_start_re_t = 0.0
    period4_end_re_t = period4_end - period4_start + 1

    for n in range(N):
        x = D[n]
        t = x.t
        if period2_start < t and t < period2_end:
            f = elem(t, x.re_t - period2_start, x.team, x.p, x.a, x.x, x.y, x.s)
            D[n] = f
        elif period3_start < t and t < period3_end:
            f = elem(t, x.re_t - period3_start, x.team, x.p, x.a, x.x, x.y, x.s)
            D[n] = f
        elif period4_start < t and t < period4_end:
            f = elem(t, x.re_t - period4_start, x.team, x.p, x.a, x.x, x.y, x.s)
            D[n] = f

        #注意：4ピリオド終了後にアクションがある

def Reverse_Seq():
#--後半反転--
    for n in range(N):
        x = D[n]
        t = x.t
        if t > period3_start:
            f = elem(t, x.re_t, x.team, x.p, x.a, xmax - x.x, ymax - x.y, x.s)
            D[n] = f




#--main--
input()
#データ読み込み

pdb.set_trace()
