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
from PIL import Image
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

Seq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
#選手位置データ Seq[Seq_id][team_id][event_id]
N_Seq = 0
Offense_team_ids = []
Diffense_team_ids = []

#--others--
Stoppage = [5,11,12]
#5:ファール, 11:サイドプレイ, 12:エンドプレイ
Turnover = [0,1,2,3]#ボール保持チームの判別に使用
#0:2P, 1:3P, 2:ドリブル, 3:パス
#残りは14:移動, 15:スクリーン

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
        if " " in temp:
            print "err"

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
                        D[counter][team_id] = event[ids,:]
                        if len(ids) != 5:#何らかの理由で5vs5になっていないときはスキップ
                            flag2 = 1
                    if flag2 == 0:
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

    for i in range(len(Turnover)):#Turnoverもディクショナリの番号に直す
        Turnover[i] = action_dic[Turnover[i]]

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


def make_sequence():
#--攻撃機会毎のデータ構造を作成--
    global N_Seq
    t0 = time()
    seq_id = -1
    event_id = 0
    prev_possesion_team = -1
    
    n = 0
    while n < N:
        x = D[n]
        tmp = np.vstack([x[0],x[1]])
        action_line = tmp[:,4]
        action_line_list = map(int, action_line.tolist())
        action_posi = np.where(action_line != action_dic[14])[0]#14:移動
        flag =  0
        for i in range(len(action_posi)):
            action_id = action_line_list[action_posi[i]]
            if action_id in Turnover:
                flag = 1
                possesion_event_posi = action_posi[i]
            elif action_id in Stoppage:
                flag = 2
                possesion_event_posi = action_posi[i]
            #else:
            #    action_len = len(action_posi)
            #    if action_len == 1:
            #        print action_line
            #        #15:スクリーンは他のアクションとセット．単体では記録されていない．

        if flag == 0:
            print "error"

        elif flag == 1:#Turnover {0:2P, 1:3P, 2:ドリブル, 3:パス}
            possesion_team = int(tmp[possesion_event_posi][2])            

            if prev_possesion_team != possesion_team:
                seq_id += 1
                event_id = 0
                Offense_team_ids.append(possesion_team)
                Diffense_team_ids.append((1 - possesion_team))

            for team_id in team_dic.itervalues():
                Seq[seq_id][team_id][event_id] = x[team_id]

            event_id += 1
            prev_possesion_team = possesion_team


        elif flag == 2:#Stoppage {5:ファール, 11:サイドプレイ, 12:エンドプレイ}           
            if action_id == action_dic[5]:#5:ファール
                seq_id += 1
                event_id = 0

            else:#11:サイドプレイ, 12:エンドプレイ
                possesion_team = int(tmp[possesion_event_posi][2])            

                seq_id += 1
                event_id = 0
                Offense_team_ids.append(possesion_team)
                Diffense_team_ids.append((1 - possesion_team))

                for team_id in team_dic.itervalues():
                    Seq[seq_id][team_id][event_id] = x[team_id]

                event_id += 1
                prev_possesion_team = possesion_team

        n += 1

    N_Seq = seq_id + 1


def visualize_sequence():

    im = Image.open('court.png')
    im = np.array(im)
    team_dic_inv = {v:k for k, v in team_dic.items()}

    seq_id = 0
    m_seq_id = 0
    while seq_id < N_Seq:

        Sub_Seq = Seq[seq_id][0]
        Leng_Sub_Seq = len(Sub_Seq)
        if Leng_Sub_Seq == 1:

            == 0がある！！


            m_seq_id += 1
        else:
            if seq_id == 19:
                pdb.set_trace()
            fig = plt.figure(figsize=(16,16))
            for team_id in team_dic.itervalues():
                Sub_Seq = Seq[seq_id][team_id]
                Leng_Sub_Seq = len(Sub_Seq)
    
                X = defaultdict(int)
                Y = defaultdict(int)
                for i in range(Leng_Sub_Seq):
                    E = Sub_Seq[i]
                    player_ids = E[:,3]
                    x = E[:,5]
                    y = E[:,6]
                    for j in range(len(player_ids)):
                        if i == 0:
                            player_id = int(player_ids[j])
                            X[player_id] = np.array(x[j])
                            Y[player_id] = np.array(y[j])
                        else:
                            player_id = int(player_ids[j])
                            X[player_id] = np.hstack([X[player_id], x[j]])
                            Y[player_id] = np.hstack([Y[player_id], y[j]])
        
                if team_id == Offense_team_ids[seq_id]:
                    OD = 'Offense'
                    OD_color = 'red'
                else:
                    OD = 'Diffense'
                    OD_color = 'blue'
                    
                if team_dic_inv[team_id] == 8:
                    team_name = 'JPN'
                else:
                    team_name = 'AUS'
        
                plt.subplot(int(str(21)+str(team_id + 1)))
                #plt.title(team_name+'  '+OD, size=30, loc='left')
                plt.title(OD, color=OD_color, size=35, loc='left')
                plt.ylabel(team_name,size=45)
                plt.tick_params(labelbottom='off',labelleft='off')
                plt.imshow(im)
                for j in range(len(player_ids)):
                    player_id = int(player_ids[j])
                    x = X[player_id]
                    y = Y[player_id]
                    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], width=0.002, \
                               headwidth=5, headlength=7, headaxislength=7, scale_units='xy',\
                               angles='xy', scale=1, color='grey')
                    #angles='xy', scale=1, color='darkcyan')
                    #quive(x,y,u,v) (x,y):始点 (u,v):方向ベクトル
                    plt.scatter(x,y,s = 40, color=OD_color,alpha=0.5)
                plt.axis([0, 600, 0, 330])
                    
            plt.savefig('Sequence/Sequence_'+str(int(seq_id - m_seq_id))+'.png', transparent=True)
            plt.close()            
    
        seq_id += 1
    

    print "ok"
    pdb.set_trace()   



#--main--
input()
#データ読み込み

make_sequence()
#攻撃機会毎のデータ構造を作成

visualize_sequence()
#攻撃機会毎のデータを可視化

pdb.set_trace()
