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
from sklearn.cluster import MeanShift, estimate_bandwidth
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

MeanShift_Cluster_no = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
N_Cluster = defaultdict(int)

BoF = defaultdict(int)
BoF_seq_id = defaultdict(int)

#--others--
Stoppage = [11,12]
#11:サイドプレイ, 12:エンドプレイ
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

Time_out = []
Menber_change = []

K = 5
C = ['blue', 'red', 'green', 'grey', 'gold']
#-----------------


def input():
#--Input--

    global xmax, xmin, ymax, ymin
    global N

    t0 = time()

    input_time()#ピリオド開始・終了，タイムアウト，メンバーチェンジの時刻読み込み

    counter = 0
    filename = "processed_metadata.csv"
    fin = open(filename)

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
            print 'error'

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


def input_time():

    global period1_start, period1_end
    global period2_start, period2_end
    global period3_start, period3_end
    global period4_start, period4_end

    global Time_out, Menber_change

    filename = "period.csv"
    fin = open(filename)    

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
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

    fin.close()


    filename = "time_out.csv"
    fin = open(filename)    
    strList = fin.readlines()

    for line in strList:
        temp = line.rstrip("\r\n").split(",")
        A = temp[0].split(".")
        B = A[0].split(":")
        C = A[1]
        Time_out.append(int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3)

    fin.close()


    filename = "menber_change.csv"
    fin = open(filename)
    strList = fin.readlines()

    for line in strList:
        temp = line.rstrip("\r\n").split(",")
        A = temp[0].split(".")
        B = A[0].split(":")
        C = A[1]
        Menber_change.append(int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2]) + int(C) * 10 ** -3)

    fin.close()
        

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
            if t >= period3_start:
                for i in range(x_team_size):
                    x_team[i][5] = xmax - x_team[i][5]
            if t < period3_start:#データの零点が左上なので注意
                for i in range(x_team_size):
                    x_team[i][6] = ymax - x_team[i][6]
            D[n][team_id] = x_team


def make_sequence():
#--攻撃機会毎のデータ構造を作成--
    global N_Seq
    t0 = time()
    seq_id = 0
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
            possesion_event_posi = action_posi[i]
            if action_id in Turnover:
                flag = 1
            elif action_id in Stoppage:
                flag = 2
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
                if len(Seq[seq_id][0]) != 0 and len(Seq[seq_id][0]) != 1 \
                   and len(Seq[seq_id][0]) != 2:
                    #見やすさのため系列数2以下のものを削除
                    seq_id += 1
                    Offense_team_ids.append(prev_possesion_team)
                    Diffense_team_ids.append((1 - prev_possesion_team))
                event_id = 0

            for team_id in team_dic.itervalues():
                Seq[seq_id][team_id][event_id] = x[team_id]

            event_id += 1
            prev_possesion_team = possesion_team


        elif flag == 2:#Stoppage {11:サイドプレイ, 12:エンドプレイ}           
            possesion_team = int(tmp[possesion_event_posi][2])            

            if len(Seq[seq_id][0]) != 0 and len(Seq[seq_id][0]) != 1 \
               and len(Seq[seq_id][0]) != 2:
                #見やすさのため系列数2以下のものを削除
                seq_id += 1
                Offense_team_ids.append(prev_possesion_team)
                Diffense_team_ids.append((1 - prev_possesion_team))
            event_id = 0

            for team_id in team_dic.itervalues():
                Seq[seq_id][team_id][event_id] = x[team_id]

            event_id += 1
            prev_possesion_team = possesion_team

        n += 1

    N_Seq = seq_id
    Quantization()


def Quantization():

    #Location---------------------
    XY = defaultdict(int)
    Labels = defaultdict(int)
    seq_id = 0
    while seq_id < N_Seq:
        offense_team_id = Offense_team_ids[seq_id]
        Sub_Seq = Seq[seq_id][offense_team_id]
        L = len(Sub_Seq)
        for l in range(L):
            x = Sub_Seq[l][:,5]
            y = Sub_Seq[l][:,6]
            xy = np.vstack([x,y]).T
            if np.size(XY[offense_team_id]) == 1:
                XY[offense_team_id] = xy
            else:
                XY[offense_team_id] = np.vstack([XY[offense_team_id], xy])
        seq_id += 1

    fig = plt.figure(figsize=(16,16))
    im = Image.open('court.png')
    im = np.array(im)
    team_dic_inv = {v:k for k, v in team_dic.items()}

    team_id = 0
    if team_dic_inv[team_id] == 8:
        team_name = 'JPN'
    else:
        team_name = 'AUS'
    plt.subplot(211)
    plt.imshow(im)
    plt.scatter(XY[team_id][:,0], XY[team_id][:,1], s = 40, color='blue', alpha=0.5)
    plt.axis([0, 600, 0, 330])
    plt.title(team_name, size=35, loc='left')

    bandwidth = 25
    #bandwidth = estimate_bandwidth(XY[team_id], quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(XY[team_id])
    labels = ms.labels_
    Labels[team_id] = labels
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    N_Cluster[team_id] = n_clusters
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], s = 200, color='black')

    team_id = 1
    if team_dic_inv[team_id] == 8:
        team_name = 'JPN'
    else:
        team_name = 'AUS'
    plt.subplot(212)
    plt.imshow(im)
    plt.scatter(XY[team_id][:,0], XY[team_id][:,1], s = 40, color='red', alpha=0.5)
    plt.axis([0, 600, 0, 330])
    plt.title(team_name, size=35, loc='left')
    #bandwidth = estimate_bandwidth(XY[team_id], quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(XY[team_id])
    labels = ms.labels_
    Labels[team_id] = labels
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    N_Cluster[team_id] = n_clusters
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], s = 200, color='black')

    plt.savefig('Location/Location.png')
    plt.close()
    #Location---------------------

    seq_id = 0
    counter = defaultdict(int)
    counter[0] = 0
    counter[1] = 0
    while seq_id < N_Seq:    
        offense_team_id = Offense_team_ids[seq_id]
        Sub_Seq = Seq[seq_id][offense_team_id]
        L = len(Sub_Seq)
        for l in range(L):        
            Sub_Labels = Labels[offense_team_id]
            Cluster_nos = Sub_Labels[counter[offense_team_id]:counter[offense_team_id]+5]
            MeanShift_Cluster_no[seq_id][offense_team_id][l] = Cluster_nos
            counter[offense_team_id] += 5

        seq_id += 1


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
            m_seq_id += 1
            pdb.set_trace()
        else:
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
                    
            plt.savefig('Sequence/Sequence_'+str(int(seq_id - m_seq_id))+'.png', \
                        transparent=True)
            plt.close()            
    
        seq_id += 1


def make_BoF():

    seq_id = 0
    while seq_id < N_Seq:
        offense_team_id = Offense_team_ids[seq_id]
        N_player = len(player_dic[offense_team_id])
        M = np.zeros([N_player, N_player])
        C = np.zeros(N_Cluster[offense_team_id] + 1)
        Sub_Seq = Seq[seq_id][offense_team_id]
        Sub_MeanShift_Cluster_no = MeanShift_Cluster_no[seq_id][offense_team_id]
        L = len(Sub_Seq)

        #pass----------------------
        for l in range(L - 1):
            Event = Sub_Seq[l]
            action_line = Event[:,4]
            action_posi = (np.where(action_line != action_dic[14])) or \
                          np.where((action_line != action_dic[15]))#14:移動, 15:スクリーン
            now_player_id = int(Event[action_posi[0][0]][3])

            Event = Sub_Seq[l+1]
            action_line = Event[:,4]
            action_posi = (np.where(action_line != action_dic[14])) or \
                          np.where((action_line != action_dic[15]))#14:移動, 15:スクリーン
            next_player_id = int(Event[action_posi[0][0]][3])

            M[now_player_id, next_player_id] += 1

        pass_line = M.reshape((1, N_player * N_player))[0]
        S_pass_line = np.sum(pass_line)
        for i in range(len(pass_line)):
            pass_line[i] = (pass_line[i] - 0.00001) / S_pass_line
        #---------------------------

        #location-------------------
        for l in range(L):
            Clusters = Sub_MeanShift_Cluster_no[l]
            for c in range(len(Clusters)):
                Cluster_id = Clusters[c]
                C[Cluster_id] += 1

        S_C = np.sum(C)
        for i in range(len(C)):
            C[i] = (C[i] - 0.00001) / S_C
        #---------------------------

        line = np.hstack([pass_line, C])
        
        if np.size(BoF[offense_team_id]) == 1:
            BoF[offense_team_id] = line
            BoF_seq_id[offense_team_id] = seq_id
        else:
            BoF[offense_team_id] = np.vstack([BoF[offense_team_id], line])
            BoF_seq_id[offense_team_id] = np.hstack([BoF_seq_id[offense_team_id], seq_id])
        
        seq_id += 1


def Clustering():
    
    PCA_threshold = 0.8

    for team_id in team_dic.itervalues():
        BoF_Team = BoF[team_id]
        
        dim = np.shape(BoF_Team)[0]
        threshold_dim = 0
        for i in range(dim):
            pca = PCA(n_components = i)
            pca.fit(BoF_Team)
            X = pca.transform(BoF_Team)
            E = pca.explained_variance_ratio_
            if np.sum(E) > PCA_threshold:
                thereshold_dim = len(E)
                print 'Team' + str(team_id)+ 'dim:%d' % thereshold_dim
                break
    
        pca = PCA(n_components = thereshold_dim)
        pca.fit(BoF_Team)
        X = pca.transform(BoF_Team)
    
        min_score = 10000
        for i in range(100):
            model = KMeans(n_clusters=K, init='k-means++', max_iter=300, tol=0.0001).fit(X)
            if min_score > model.score(X):
                min_score = model.score(X)
                labels = model.labels_
        print min_score
    
        pca = PCA(n_components = 2)
        pca.fit(BoF_Team)
        X = pca.transform(BoF_Team)
        for k in range(K):
            labels_ind = np.where(labels == k)[0]
            plt.scatter(X[labels_ind,0], X[labels_ind,1], color=C[k])
        plt.legend(['C0','C1','C2','C3','C4'], loc=4)
    
        plt.title('Team' + str(team_id) + '1_PCA_kmeans')
        plt.savefig('Seq_Team' + str(team_id)+ '/Team' + str(team_id) + '_PCA_kmeans.png')
        plt.show()
        plt.close()
        np.savetxt('Seq_Team' + str(team_id) + '/labels_Team' + str(team_id) + '.csv', \
                   labels, delimiter=',')
    

def Visualize_tactical_pattern():
#--kmeansで出力されたラベルに基づいて攻撃パターンを色塗り--

    x_period1 = np.arange(int(period1_end_re_t) + 1)
    y0_period1 = np.zeros(int(period1_end_re_t) + 1)

    x_period2 = np.arange(int(period2_end_re_t) + 1)
    y0_period2 = np.zeros(int(period2_end_re_t) + 1)

    x_period3 = np.arange(int(period3_end_re_t) + 1)
    y0_period3 = np.zeros(int(period3_end_re_t) + 1)

    x_period4 = np.arange(int(period4_end_re_t) + 1)
    y0_period4 = np.zeros(int(period4_end_re_t) + 1)


    for team_id in team_dic.itervalues():

        fig = plt.figure(figsize=(16,4))
        plt.subplots_adjust(hspace=1.5)

        seq_id_set = BoF_seq_id[team_id]
        N_of = len(seq_id_set)
        labels = np.loadtxt('Seq_Team' + str(team_id) + '/labels_Team' + str(team_id) + \
                            '.csv', delimiter=',')
        
        for k in range(K):
            Y_period1_of = np.copy(y0_period1)
            Y_period2_of = np.copy(y0_period2)
            Y_period3_of = np.copy(y0_period3)
            Y_period4_of = np.copy(y0_period4)

            for i in range(N_of):
                if labels[i] == k:
                    seq_id = seq_id_set[i]
                    S = Seq[seq_id][team_id]
                    start_t = int(S[0][0,0])
                    start_re_t = int(S[0][0,1])
                    end_re_t = int(S[len(S) - 1][0,1])
                    period = int(end_re_t - start_re_t)

                    if start_t < period1_end:
                        for j in range(period):
                            Y_period1_of[start_re_t + j] = 1.0
                    elif period2_start < start_t and start_t < period2_end:
                        for j in range(period):
                            Y_period2_of[start_re_t + j] = 1.0
                    elif period3_start < start_t and start_t < period3_end:
                        for j in range(period):
                            Y_period3_of[start_re_t + j] = 1.0
                    elif period4_start < start_t and start_t < period4_end:
                        for j in range(period):
                            Y_period4_of[start_re_t + j] = 1.0
                    
            plt.subplot(4, 1, 1)        
            plt.fill_between(x_period1, y0_period1, Y_period1_of, edgecolor = C[k], facecolor = C[k])

            plt.subplot(4, 1, 2)
            plt.fill_between(x_period2, y0_period2, Y_period2_of, edgecolor = C[k], facecolor = C[k])

            plt.subplot(4, 1, 3)
            plt.fill_between(x_period3, y0_period3, Y_period3_of, edgecolor = C[k], facecolor = C[k])

            plt.subplot(4, 1, 4)
            plt.fill_between(x_period4, y0_period4, Y_period4_of, edgecolor = C[k], facecolor = C[k])


        plt.subplot(4, 1, 1)
        plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
        plt.yticks([])
        plt.title('Team0_period1_offense')

        plt.subplot(4, 1, 2)
        plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
        plt.yticks([])
        plt.title('Team0_period2_offense')

        plt.subplot(4, 1, 3)
        plt.axis([period3_start_re_t, period3_end_re_t, 0, 1])
        plt.yticks([])
        plt.title('Team0_period3_offense')

        plt.subplot(4, 1, 4)
        plt.axis([period4_start_re_t, period4_end_re_t, 0, 1])
        plt.yticks([])
        plt.title('Team0_period4_offense')
        


        pdb.set_trace()
    


    for k in range(K):
        Y_period1_Team1_of = np.copy(y0_period1)
        Y_period2_Team1_of = np.copy(y0_period2)
        Y_period3_Team1_of = np.copy(y0_period3)
        Y_period4_Team1_of = np.copy(y0_period4)

        for i in range(N_Team1_of):
            if labels_Team1[i] == k:
                S = Seq_Team1_of[i]
                start_t = S[0,0]
                start_re_t = S[0,1]
                end_re_t = S[len(S) - 1,1]
                period = int(end_re_t - start_re_t)

                if start_t < period1_end:
                    for j in range(period):
                        Y_period1_Team1_of[start_re_t + j] = 1.0
                elif period2_start < start_t and start_t < period2_end:
                    for j in range(period):
                        Y_period2_Team1_of[start_re_t + j] = 1.0
                elif period3_start < start_t and start_t < period3_end:
                    for j in range(period):
                        Y_period3_Team1_of[start_re_t + j] = 1.0
                elif period4_start < start_t and start_t < period4_end:
                    for j in range(period):
                        Y_period4_Team1_of[start_re_t + j] = 1.0

        plt.subplot(4, 1, 1)        
        plt.fill_between(x_period1, y0_period1, Y_period1_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 2)
        plt.fill_between(x_period2, y0_period2, Y_period2_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 3)
        plt.fill_between(x_period3, y0_period3, Y_period3_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 4)
        plt.fill_between(x_period4, y0_period4, Y_period4_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])



    plt.subplot(4, 1, 1)        
    tempX = np.array(shot_period1_Team1_re_t)
    tempY = np.ones(len(shot_period1_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period1_Team1_re_t)
    tempY = np.ones(len(shot_success_period1_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period1_offense')

    plt.subplot(4, 1, 2)        
    tempX = np.array(shot_period2_Team1_re_t)
    tempY = np.ones(len(shot_period2_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period2_Team1_re_t)
    tempY = np.ones(len(shot_success_period2_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period2_offense')

    plt.subplot(4, 1, 3)        
    tempX = np.array(shot_period3_Team1_re_t)
    tempY = np.ones(len(shot_period3_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period3_Team1_re_t)
    tempY = np.ones(len(shot_success_period3_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period3_offense')

    plt.subplot(4, 1, 4)        
    tempX = np.array(shot_period4_Team1_re_t)
    tempY = np.ones(len(shot_period4_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period4_Team1_re_t)
    tempY = np.ones(len(shot_success_period4_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period4_offense')

    plt.savefig('Seq_Team1/Vis_tactical_pattern_Team1.png')
    plt.show()
    plt.close()







    #--Team2--
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=1.5)


    for k in range(K):
        Y_period1_Team2_of = np.copy(y0_period1)
        Y_period2_Team2_of = np.copy(y0_period2)
        Y_period3_Team2_of = np.copy(y0_period3)
        Y_period4_Team2_of = np.copy(y0_period4)

        for i in range(N_Team2_of):
            if labels_Team2[i] == k:
                S = Seq_Team2_of[i]
                start_t = S[0,0]
                start_re_t = S[0,1]
                end_re_t = S[len(S) - 1,1]
                period = int(end_re_t - start_re_t)

                if start_t < period1_end:
                    for j in range(period):
                        Y_period1_Team2_of[start_re_t + j] = 1.0
                elif period2_start < start_t and start_t < period2_end:
                    for j in range(period):
                        Y_period2_Team2_of[start_re_t + j] = 1.0
                elif period3_start < start_t and start_t < period3_end:
                    for j in range(period):
                        Y_period3_Team2_of[start_re_t + j] = 1.0
                elif period4_start < start_t and start_t < period4_end:
                    for j in range(period):
                        Y_period4_Team2_of[start_re_t + j] = 1.0

        plt.subplot(4, 1, 1)        
        plt.fill_between(x_period1, y0_period1, Y_period1_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 2)
        plt.fill_between(x_period2, y0_period2, Y_period2_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 3)
        plt.fill_between(x_period3, y0_period3, Y_period3_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 4)
        plt.fill_between(x_period4, y0_period4, Y_period4_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])


    plt.subplot(4, 1, 1)        
    tempX = np.array(shot_period1_Team2_re_t)
    tempY = np.ones(len(shot_period1_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period1_Team2_re_t)
    tempY = np.ones(len(shot_success_period1_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period1_offense')

    plt.subplot(4, 1, 2)        
    tempX = np.array(shot_period2_Team2_re_t)
    tempY = np.ones(len(shot_period2_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period2_Team2_re_t)
    tempY = np.ones(len(shot_success_period2_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period2_offense')

    plt.subplot(4, 1, 3)        
    tempX = np.array(shot_period3_Team2_re_t)
    tempY = np.ones(len(shot_period3_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period3_Team2_re_t)
    tempY = np.ones(len(shot_success_period3_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period3_start_re_t, period3_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period3_offense')

    plt.subplot(4, 1, 4)        
    tempX = np.array(shot_period4_Team2_re_t)
    tempY = np.ones(len(shot_period4_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period4_Team2_re_t)
    tempY = np.ones(len(shot_success_period4_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period4_start_re_t, period4_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period4_offense')

    plt.savefig('Seq_Team2/Vis_tactical_pattern_Team2.png')
    plt.show()
    plt.close()



    print "ok"
    pdb.set_trace()   


#--main--
input()
#データ読み込み

make_sequence()
#攻撃機会毎のデータ構造を作成

#visualize_sequence()
#攻撃機会毎のデータを可視化

make_BoF()
#パス系列と量子化された位置情報を含むBag-of-Feature作成

Clustering()
#BoFを入力にして攻撃パターンをクラスタリング

Visualize_tactical_pattern()
#タクティカルパターンの出力

pdb.set_trace()
