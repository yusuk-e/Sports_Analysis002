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
D = defaultdict(lambda: defaultdict(int))#選手位置データ D[seq_id][team_id]
N = 0
Seq_Team1_of = defaultdict(int)
Seq_Team2_of = defaultdict(int)
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
    period1_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period1_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period2_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period2_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period3_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period3_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period4_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period4_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

    counter = 0
    temp_player0_dic = {}
    temp_player1_dic = {}
    flag = 0
    time_fix = -1
    for row in fin:
        temp = row.rstrip("\r\n").split(",")

        A = temp[0].split(".")
        B = A[0].split(":")
        C = A[1]
        t = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

        A = temp[1].split(".")
        B = A[0].split(":")
        C = A[1]
        re_t = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3

        team = int(temp[2])
        if team not in team_dic:
            team_dic[team] = len(team_dic)

        player = int(temp[3])
        if team_dic[team] == 0:
            if player not in temp_player0_dic:
                temp_player0_dic[player] = len(temp_player0_dic)
                player_dic[0] = temp_player0_dic
        elif team_dic[team] == 1:
            if player not in temp_player1_dic:
                temp_player1_dic[player] = len(temp_player1_dic)
                player_dic[1] = temp_player1_dic
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

        if time_fix != t:
            if flag == 1:
                if len(event) == 10:
                    #コート上にいる人数が5,7,9,11人とか有り．とりあえず削除．
                    flag2 = 0
                    for team_id in team_dic.itervalues():
                        ids = np.where(event[:,2] == team_id)[0]
                        if len(ids) == 5:
                            D[counter][team_id] = event[ids,:]
                            flag2 = 1
                    if flag2 == 1:
                        counter += 1

            time_fix = t
            event = np.array([[t, re_t, team_dic[team], player_dic[team_dic[team]]\
                               [player], action_dic[action], x, y, s]])

        elif time_fix == t:
            etemp = np.array([t, re_t, team_dic[team], player_dic[team_dic[team]]\
                              [player], action_dic[action], x, y, s])
            event = np.vstack([event,etemp])
            flag = 1

    fin.close()
    N = counter

    for i in range(len(Stoppage)):#Stoppageもディクショナリの番号に直す
        Stoppage[i] = action_dic[Stoppage[i]]

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
        for team_id in team_dic.values():
            x_team = x[team_id]
            x_team_size = np.shape(x_team)[0]
            t = x_team[0][0]
            for i in range(x_team_size):
                if period2_start < t and t < period2_end:
                    x_team[i][1] = t - period2_start
                elif period3_start < t and t < period3_end:
                    x_team[i][1] = t - period3_start
                elif period4_start < t and t < period4_end:
                    x_team[i][1] = t - period4_start
            D[n][team_id] = x_team
            #注意：4ピリオド終了後にアクションがある


def Reverse_Seq():
#--後半反転--
    for n in range(N):
        x = D[n]
        for team_id in team_dic.values():
            x_team = x[team_id]
            x_team_size = np.shape(x_team)[0]
            t = x_team[0][0]
            if t > period3_start:
                for i in range(x_team_size):
                    x_team[i][5] = xmax - x_team[i][5]
                    x_team[i][6] = ymax - x_team[i][6]
            D[n][team_id] = x_team


def Seq_Team_of():
#--offense ボール軌跡--
    global N_Team1_of, N_Team2_of
    t0 = time()

    n = 0
    while n < N:
        x = D[n]
        pdb.set_trace()




    pre_team = 0
    flag = 0

    while n < N:
        team = D[n].team
        action = D[n].a

        if action in Stoppage:
            flag = 1
            n += 1

        else:
            if pre_team == team:
                if team == 0:
                    if flag == 1:
                        N_Team1_of += 1
                        flag = 0

                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                    n += 1
                    pre_team = team

                elif team == 1:
                    if flag == 1:
                        N_Team2_of += 1
                        flag = 0

                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    n += 1
                    pre_team = team

            else:
                if team == 0:
                    N_Team1_of += 1
                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                    n += 1
                    pre_team = team

                elif team == 1:
                    N_Team2_of += 1
                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    n += 1
                    pre_team = team

    N_Team1_of = len(Seq_Team1_of)
    N_Team2_of = len(Seq_Team2_of)



#--main--
input()
#データ読み込み

Seq_Team_of()
#オフェンス時のボール軌跡データ作成

pdb.set_trace()
