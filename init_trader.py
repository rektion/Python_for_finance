import numpy
import matplotlib.pyplot as plt
import copy
import random
import csv
from multiprocessing import Pool
from properties import n_children, epsilon, base_capital, first_price, init_arr, coin_name

# Dans cette dernière partie, nous allons créer un "jeu" qui permet d'acheter ou de vendre
# une partie de ses fonds suivant le stock marcket du jour.
# - qui est en fait le stock market des 2 dernières années -

# Puis nous alons générer un nombre très important 'traders' prenant de décisions d'achat et de ventes aléatoires
# Nous allons conserver les historiques de ceux ayant performé

# balance, number of coins, estimated total balance, buy_or_sell_ratio, price


def load_crypto():
    f = open('historical_datas\\Binance_' + coin_name + 'USDT_d.csv', 'rb').readlines()[2:]
    raw = []
    for b_line in f:
        line = b_line.decode("utf-8")
        raw.append(float(line.split(',')[3]))
    raw.reverse()
    return raw 

raw = load_crypto()

# Fonction qui permet de stocker la décision d'un jour dans l'historique de trading

def step_history(buy_or_sell_ratio, price, historyy):
    previous = historyy[len(historyy)-1]
    if buy_or_sell_ratio == 0:
        historyy.append([previous[0], previous[1], previous[2], 0, price])
    else:
        if buy_or_sell_ratio > 0:  # we buy
            amount_in_order = previous[0]*buy_or_sell_ratio
            balance = previous[0] - amount_in_order
            n_barrels = previous[1] + ((amount_in_order*0.999)/price) # 0.999 is binance fee
        else: # we sell
            amount_in_order = previous[1]*(-buy_or_sell_ratio)
            balance = previous[0] + amount_in_order*0.999*price # 0.999 is binance fee
            n_barrels = previous[1] - amount_in_order
        estimated_total_balance = balance + n_barrels*price
        historyy.append([balance, n_barrels, estimated_total_balance, buy_or_sell_ratio, price])
    return historyy

# Fonction qui permet - à partir de 50 traders - de générer
# 200 enfants avec des choix de décisions proches de celles
# des parents. Le nombre d'enfants en modifiable

def calculate_next_gen(historys):
    next_gen = []
    history = [init_arr]
    historys.sort(key=lambda x: x[len(x)-1][2])
    for j in range(0, 49, 2): # Pour chaque couple
        for n in range(0, n_children): # On créé 4 enfants
            for i in range(1,len(historys[0])):
                avg = (historys[j][i][3] + historys[j+1][i][3])/2 # Chaque enfant reprend les gènes des parents
                avg = avg + numpy.random.normal(0, abs(epsilon*avg)) # Avec une petite variation epsilon
                if avg < -1:
                    avg = -1
                elif avg > 1:
                    avg = 1
                step_history(avg, historys[0][i][4], history)
            next_gen.append(copy.copy(history))
            del history
            history = [init_arr]
    del historys
    next_gen.sort(key=lambda x: x[len(x)-1][2])
    next_gen = next_gen[-50:] # Seul les 50 meilleurs sont ajoutés à la prochaine génération
    return next_gen

def init_parent(historys):
    history = [init_arr]
    while True:
        for elem in raw:
            step_history(random.uniform(-1, 1), elem, history)
        if history[len(history)-1][2] > 18000: # On m'appelle la sélection naturelle
            print("1 parent en plus")
            return history
        del history
        history = [init_arr]

# Permet de générer la génération initiale aléatoirement
# Ici la séléction naturelle est de 6000
# Cela prend ~1h sur mon pc mais vous pouvez baisser
# La valeur à 4000 et cela prend quelques minutes

def init_parents():
    historys = [0]*50
    with Pool(8) as p:
        historys = p.map(init_parent, historys)
    return historys
    


# Gènère N générations à partir de rien
# Stoque les résultats dans un csv

def make_n_generation(n):
    next_gen = init_parents()
    for j in range(1,n+1):
        history = [init_arr]
        next_gen = calculate_next_gen(next_gen)
        del history
    for j in range(len(next_gen)):
        with open("trader_data\\" + coin_name + "\\trader" + str(j) + ".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["balance", "n_coins", "estimated_total_balance", "buy_or_sell_ratio", "price"])
            for k in range(len(next_gen[j])):
                writer.writerow([next_gen[j][k][0], next_gen[j][k][1], next_gen[j][k][2], next_gen[j][k][3], next_gen[j][k][4]])


if __name__ == '__main__':
    make_n_generation(6)