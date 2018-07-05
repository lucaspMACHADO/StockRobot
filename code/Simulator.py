import sys
import pickle
from elm import ELMClassifier
from Plotter import plot, plot_stockprice
import time
import numpy as np
import Indicators as id


goers_trn = []
goers_tst = []
mn = []
mx = []
mnm = None
obv = None
bbs = None
m_9 = None
m12 = None
m26 = None


def join_by_min(fname, join_size):
    global goers_trn
    global goers_tst
    start = time.time()
    raw = np.loadtxt(fname, dtype='str', delimiter='\t')
    trn = np.reshape([], (-1, 15))
    tst = np.reshape([], (-1, 15))
    flg = False
    date_aux = raw[0, 1].split(' ')[0]
    date_aux_i = date_aux
    cnt = 0
    op = float(raw[0, 2])
    hi = float(raw[0, 3])
    lo = float(raw[0, 4])
    vl = 0
    aux_ii = len(raw) - 1
    for ii, i in enumerate(raw):
        aux = i[1].split(' ')[0]
        if  aux == '2017-04-03':
            flg = True
        if date_aux_i != aux: # Caso tenha mudado de dia
            if aux:
                goers_tst.append(len(tst))
            else:
                goers_trn.append(len(trn))
            date_aux_i = aux
        if cnt == join_size or aux != date_aux or ii == aux_ii:
            cl = float(raw[ii - 1, 5])
            if flg:
                tst = np.append(tst, [[op, hi, lo, cl, vl, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
            else:
                trn = np.append(trn, [[op, hi, lo, cl, vl, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
            date_aux = aux
            op = float(i[2])
            hi = float(i[3])
            lo = float(i[4])
            vl = float(i[6])
            cnt = 0
        else:
            aux_i = float(i[3])
            if aux_i > hi:
                hi = aux_i
            aux_i = float(i[4])
            if aux_i < lo:
                lo = aux_i
            vl += float(i[6])
            cnt += 1
        sys.stdout.write('\r' + '%6d / %d' % (ii, aux_ii) + '\033[K')
    sys.stdout.write('\r' + '>> %6.2f: Data Join Done!\n' % (time.time() - start) + '\033[K')
    return trn, tst


def join_by_vol(fname, join_size):
    global goers_trn
    global goers_tst
    start = time.time()
    raw = np.loadtxt(fname, dtype='str', delimiter='\t')
    trn = np.reshape([], (-1, 15))
    tst = np.reshape([], (-1, 15))
    flg = False
    date_aux = raw[0, 1].split(' ')[0]
    date_aux_i = date_aux
    op = float(raw[0, 2])
    hi = float(raw[0, 3])
    lo = float(raw[0, 4])
    vl = 0
    aux_ii = len(raw) - 1
    for ii, i in enumerate(raw):
        aux = i[1].split(' ')[0]
        if aux == '2017-04-03':
            flg = True
        if date_aux_i != aux: # Caso tenha mudado de dia
            if flg:
                goers_tst.append(len(tst))
            else:
                goers_trn.append(len(trn))
            date_aux_i = aux
        if vl >= join_size or aux != date_aux or ii == aux_ii:
            cl = float(raw[ii - 1, 5])
            if flg:
                tst = np.append(tst, [[op, hi, lo, cl, vl, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
            else:
                trn = np.append(trn, [[op, hi, lo, cl, vl, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
            date_aux = aux
            op = float(i[2])
            hi = float(i[3])
            lo = float(i[4])
            vl = float(i[6])
        else:
            aux_i = float(i[3])
            if aux_i > hi:
                hi = aux_i
            aux_i = float(i[4])
            if aux_i < lo:
                lo = aux_i
            vl += float(i[6])
        sys.stdout.write('\r' + '%6d / %d' % (ii, aux_ii) + '\033[K')
    sys.stdout.write('\r' + '>> %6.2f: Data Join Done!\n' % (time.time() - start) + '\033[K')
    return trn, tst


def prep_data(trn, tst):
    global mn
    global mx
    global mnm
    mnm = id.Momentum()
    global obv
    obv = id.OBV()
    global bbs
    bbs = id.Bbands()
    global m_9
    m_9 = id.EMA(9)
    global m12
    m12 = id.EMA(12)
    global m26
    m26 = id.EMA(26)
    start = time.time()
    aux_ii = len(trn) - 1
    for idx, val in enumerate(trn):
        if idx == 0:
            trn[idx, 5] = id.log_return(val[0], val[3], val[0], val[3])
        else:
            trn[idx, 5] = id.log_return(val[0], val[3], trn[idx - 1, 0], trn[idx - 1, 3])
        trn[idx, 6] = mnm.get(val[5])
        trn[idx, 7] = obv.get_obv(val[3], val[4])
        aux = bbs.sma(val[3])
        if aux is not None:
            trn[idx, 8], trn[idx, 9] = aux
        aux_9 = m_9.ema(val[3])
        aux12 = m12.ema(val[3])
        aux26 = m26.ema(val[3])
        trn[idx, 10] = aux12 - aux26
        trn[idx, 11] = trn[idx, 10] - aux_9
        sys.stdout.write('\r' + '%6d / %d' % (idx, aux_ii) + '\033[K')
    for idx, val in enumerate(trn[:-1]):
        if trn[idx + 1, 5] > 0.0:
            trn[idx, 12] = 1.0
            trn[idx, 13] = 0.0
        elif trn[idx + 1, 5] < 0.0:
            trn[idx, 12] = 0.0
            trn[idx, 13] = 1.0
        else:
            trn[idx, 12] = 0.0
            trn[idx, 13] = 0.0
    if tst[0, 5] > 0.0:
        trn[-1, 12] = 1.0
        trn[-1, 13] = 0.0
    elif tst[0, 5] < 0.0:
        trn[-1, 12] = 0.0
        trn[-1, 13] = 1.0
    else:
        trn[-1, 12] = 0.0
        trn[-1, 13] = 0.0

    for idx, val in enumerate(tst):
        if idx == 0:
            tst[idx, 5] = id.log_return(val[0], val[3], val[0], val[3])
        else:
            tst[idx, 5] = id.log_return(val[0], val[3], trn[idx - 1, 0], trn[idx - 1, 3])

    for idx, val in enumerate(tst[:-1]):
        if tst[idx + 1, 5] > 0.0:
            tst[idx, 12] = 1.0
            tst[idx, 13] = 0.0
        elif tst[idx + 1, 5] < 0.0:
            tst[idx, 12] = 0.0
            tst[idx, 13] = 1.0
        else:
            tst[idx, 12] = 0.0
            tst[idx, 13] = 0.0
    for i in goers_tst[1:]:
        tst[i - 1, 14] = 1.0
    tst[-1, 14] = 1.0
    for i in range(len(trn[0, :12])):
        mn.append(min(trn[:, i]))
        mx.append(1 / (max(trn[:, i]) - mn[i]))
    for i in enumerate(trn):
        for j in enumerate(i[1][:12]):
            trn[i[0], j[0]] = (j[1] - mn[j[0]]) * mx[j[0]]
    sys.stdout.write('\r' + '>> %6.2f: Data Prep Done!\n' % (time.time() - start) + '\033[K')
    return np.delete(trn, list(range(21)) + goers_trn, axis=0), tst


def simulate(trn, tst):
    start = time.time()
    b_tp = b_fp = b_tn = b_fn = 0
    s_tp = s_fp = s_tn = s_fn = 0
    b_min = s_min = 1000000
    b_max = s_max = 0
    b_money = s_money = 0
    b_money_vec = [0]
    s_money_vec = [0]
    b_gain = s_gain = 0
    b_loss = s_loss = 0
    b_draw = s_draw = 0
    b_gain_vec = []
    s_gain_vec = []
    b_loss_vec = []
    s_loss_vec = []
    b_max_drawdown = s_max_drawdown = 0
    b_pos = s_pos = False
    time_vec = []
    aux_ii = len(tst) - 1

    for t, val in enumerate(tst):
        start_i = time.time()

        if t == 201:
            continue

        if t == 0:
            tst[0, 5] = id.log_return(tst[0, 0], tst[0, 3], trn[-1, 0], trn[-1, 3])
        else:
            tst[t, 5] = id.log_return(tst[t, 0], tst[t, 3], trn[t - 1, 0], trn[t - 1, 3])
        tst[t, 6] = mnm.get(val[5])
        tst[t, 7] = obv.get_obv(val[3], val[4])
        aux = bbs.sma(val[3])
        if aux is not None:
            tst[t, 8], tst[t, 9] = aux
        aux_9 = m_9.ema(val[3])
        aux12 = m12.ema(val[3])
        aux26 = m26.ema(val[3])
        tst[t, 10] = aux12 - aux26
        tst[t, 11] = tst[t, 10] - aux_9

        aux = trn[-1000:]
        aux_i = [(i[1] - mn[i[0]]) * mx[i[0]] for i in enumerate(tst[t, :12])]
        # aux_j = trn[-1000:, :]

        b_elm = ELMClassifier(random_state=0, n_hidden=200, activation_func='sigmoid', alpha=0.0)
        b_elm.fit(aux[:, :12], aux[:, 12])
        b_res = b_elm.predict([aux_i[:12]])
        s_elm = ELMClassifier(random_state=0, n_hidden=200, activation_func='sigmoid', alpha=0.0)
        s_elm.fit(aux[:, :12], aux[:, 13])
        s_res = s_elm.predict([aux_i[:12]])

        if b_res == 1.0:
            if val[12] == 1.0:
                b_tp += 1
            else:
                b_fp += 1
            if not b_pos:
                # Entra
                b_money -= val[3]
                b_pos = True
        else:
            if val[12] == 0.0:
                b_tn += 1
            else:
                b_fn += 1
            if b_pos:
                # Sai
                b_money += val[3]
                b_pos = False
                if b_money < b_money_vec[-1]:
                    b_loss += 1
                    b_loss_vec.append(b_money_vec[-1] - b_money)
                elif b_money > b_money_vec[-1]:
                    b_gain += 1
                    b_gain_vec.append(b_money - b_money_vec[-1])
                else:
                    b_draw += 1
        if val[14] == 1.0:
            # Sai
            b_money += val[3]
            b_pos = False
            if b_money < b_money_vec[-1]:
                b_loss += 1
                b_loss_vec.append(b_money_vec[-1] - b_money)
            elif b_money > b_money_vec[-1]:
                b_gain += 1
                b_gain_vec.append(b_money - b_money_vec[-1])
            else:
                b_draw += 1

        if b_pos:
            b_money_vec.append(b_money_vec[-1])
        else:
            b_money_vec.append(b_money)
            if b_money > b_max:
                b_max = b_money
            if b_money < b_min:
                b_min = b_money

        if s_res == 1.0:
            if val[13] == 1.0:
                s_tp += 1
            else:
                s_fp += 1
            if not s_pos:
                # Entra
                s_money += val[3]
                s_pos = True
        else:
            if val[13] == 0.0:
                s_tn += 1
            else:
                s_fn += 1
            if s_pos:
                # Sai
                s_money -= val[3]
                s_pos = False
                if s_money < s_money_vec[-1]:
                    s_loss += 1
                    s_loss_vec.append(s_money_vec[-1] - s_money)
                elif s_money > s_money_vec[-1]:
                    s_gain += 1
                    s_gain_vec.append(s_money - s_money_vec[-1])
                else:
                    s_draw += 1
        if val[14] == 1.0:
            # Sai
            s_money -= val[3]
            s_pos = False
            if s_money < s_money_vec[-1]:
                s_loss += 1
                s_loss_vec.append(s_money_vec[-1] - s_money)
            elif s_money > s_money_vec[-1]:
                s_gain += 1
                s_gain_vec.append(s_money - s_money_vec[-1])
            else:
                s_draw += 1

        if s_pos:
            s_money_vec.append(s_money_vec[-1])
        else:
            s_money_vec.append(s_money)
            if s_money > s_max:
                s_max = s_money
            if s_money < s_min:
                s_min = s_money

        # print(aux_i + list(tst[t, 12:]))
        trn = np.append(trn, [aux_i + list(tst[t, 12:])], axis=0)
        time_vec.append(time.time() - start_i)
        sys.stdout.write('\r' + '%6d / %d' % (t, aux_ii) + '\033[K')
    sys.stdout.write('\r' + '>> %6.2f: Simulation Done!\n\n' % (time.time() - start) + '\033[K')

    print('#### ' + sys.argv[1] + ' ####')
    print('Tempo médio: %f' % np.mean(time_vec))
    print('Final      : %5.5f | %5.5f' % (b_money, s_money))
    # print('Final      : %5.5f | %5.5f' % (b_money_vec[-1], s_money_vec[-1]))
    print('Minimo     : %5.5f | %5.5f' % (b_min, s_min))
    print('Maximo     : %5.5f | %5.5f' % (b_max, s_max))
    print('Ganho qtd  : %10d | %10d' % (b_gain, s_gain))
    print('Perda qtd  : %10d | %10d' % (b_loss, s_loss))
    print('Empate qtd : %10d | %10d' % (b_draw, s_draw))
    print('Ganho medio: %5.5f | %5.5f' % (np.mean(b_gain_vec), np.mean(s_gain_vec)))
    print('Perda media: %5.5f | %5.5f' % (np.mean(b_loss_vec), np.mean(s_loss_vec)))
    print('TP         : %10d | %10d' % (b_tp, s_tp))
    print('FP         : %10d | %10d' % (b_fp, s_fp))
    print('TN         : %10d | %10d' % (b_tn, s_tn))
    print('FN         : %10d | %10d' % (b_fn, s_fn))

    plot(b_money_vec, s_money_vec, sys.argv[1], tst[:, 3])


if __name__ == '__main__':
    trn, tst = join_by_min('../data/' + sys.argv[1] + '.csv', 5)
    # trn, tst = join_by_vol('../data/' + sys.argv[1] + '.csv', 100000)
    trn, tst = prep_data(trn, tst)
    plot_stockprice(sys.argv[1], tst[:, 3])
    # simulate(trn, tst)



#
# with open('../data/' + sys.argv[1] + '.pkl', 'rb') as file:
#
#     data = pickle.load(file)
#
# # print(list(data[-9791:, 3]))
#
# # for i in range(200):
# #
# #     if data[27650 + i][-1] == 1.0:
# #
# #         print(27650 + i, ':', data[27650 + i][-1])
#
# # Buy-only
# for i in enumerate(data):
#
#     if i[1][-2] > 0:
#
#         data[i[0], -2] = 1
#
#     else:
#
#         data[i[0], -2] = 0
#
# # Sell-only
# # for i in enumerate(data):
# #
# #     if i[1][-2] < 0:
# #
# #         data[i[0], -2] = 0
# #
# #     else:
# #
# #         data[i[0], -2] = 1
#
# din = 0
# vec = [0.0]
# pos = False
# limit = len(data[27672:]) - 1
# time_vec = []
#
# min_mon = None
# max_mon = None
#
# tp = 0
# fp = 0
# tn = 0
# fn = 0
#
# for idx, val in enumerate(data[27672:]):
#
#     start = time.time()
#
#     elmr = ELMRegressor(random_state=0, n_hidden=200, activation_func='sigmoid', alpha=0.0)
#     elmr.fit(data[27672+idx-1000:27672+idx, :-2], data[27672+idx-1000:27672+idx, -2])
#
#     res = elmr.predict([val[:-2]])
#
#     if res > 0.5:
#
#         if val[-2] == 1:
#
#             tp += 1
#
#         else:
#
#             fp += 1
#
#     else:
#
#         if val[-2] > 0.5:
#
#             fn += 1
#
#         else:
#
#             tn += 1
#
#     # time_vec.append(time.time() - start)
#     #
#     # if not pos and res > 0.5:
#     #
#     #     # Entra
#     #     din -= val[3]
#     #     pos = True
#     #
#     # elif (pos and res <= 0.5) or (idx == limit or data[idx+1, -1] == 1.0):
#     #
#     #     #Sai
#     #     din += val[3]
#     #     pos = False
#     #
#     # # if not pos and res > 0.5:
#     # #
#     # #     din += val[3]
#     # #     pos = True
#     # #
#     # # if (pos and res <= 0.5) or (idx == limit or data[idx+1, -1] == 1.0):
#     # #
#     # #     din -= val[3]
#     # #     pos = False
#     # #
#     # # # vec.append(din)
#     sys.stdout.write('\r' + '%4d / %d' % (idx, limit) + '\033[K')
#     #
#     # if not pos:
#     #
#     #     if min_mon is None or din < min_mon:
#     #
#     #         min_mon = din
#     #
#     #     if max_mon is None or din > max_mon:
#     #
#     #         max_mon = din
#     #
#     #     vec.append(din)
#     #
#     # else:
#     #
#     #     vec.append(vec[-1])
#
# print()
# # print('pos:', pos)
# # print('din:', din)
# # print('min:', min_mon)
# # print('max:', max_mon)
# # print('mean_time:', mean(time_vec))
# # print('std_time:', std(time_vec))
# # print(vec)
# print('tp:', tp)
# print('fp:', fp)
# print('tn:', tn)
# print('fn:', fn)
