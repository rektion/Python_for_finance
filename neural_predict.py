# Le but de cette partie est de créer un reseau de neuronnes
# Qui va tenter de prédire les valeurs d'achat et de vente des
# Traders générés aléatoirement - qui font quand même +2% par mois -
# A partir du stock marcket des 30 derniers jours
# Enfin, on va simuler les achats/ventes du réseau de neuronne sur dix ans
# Et regarder ensuite si son portefeuille a bien évolué

import tensorflow.keras.layers as lyrs
import tensorflow.keras.models as mod
import numpy as np
import matplotlib.pyplot as plt
import copy
from properties import init_arr, coin_name
from os import listdir
from os.path import isfile, join

length = len(open("historical_datas" + "\\" + "Binance_" + coin_name + "USDT_d.csv", 'rb').readlines()[2:])

def find_best_trader():
    onlyfiles = [f for f in listdir("trader_data\\" + coin_name) if isfile(join("trader_data\\" + coin_name, f))]
    final_capital = []
    for i in range(len(onlyfiles)):
        f = open("trader_data\\" + coin_name + "\\" + onlyfiles[i], 'rb').readlines()[-1:]
        line_str = f[0].decode("utf-8")
        float_value = float(line_str.split(',')[2])
        final_capital.append([i, float_value])
    final_capital.sort(key=lambda x: x[1], reverse=True)
    return final_capital[0][0]

best_trader_number = find_best_trader()

def load_signals_crypto():
    # https://www.CryptoDataDownload.com
    onlyfiles = [f for f in listdir("historical_datas") if isfile(join("historical_datas", f)) and f != "Binance_" + coin_name + "USDT_d.csv"]
    signals = []
    for fichier in onlyfiles:
        f = open("historical_datas" + "\\" + fichier, 'rb').readlines()[2:]
        signal = []
        for i in range(length):
            line_str = f[i].decode("utf-8")
            str_value = line_str.split(',')[3]
            signal.append(float(str_value))
        signal.reverse()
        signals.append(signal)
    return signals

def load_signals_5_days_per_week():
    # https://www.investing.com/commodities/gold-historical-data
    onlyfiles = [f for f in listdir("other_signals") if isfile(join("other_signals", f))]
    # signals = []
    for fichier in onlyfiles:
        f = open("other_signals" + "\\" + fichier, 'rb').readlines()[1:]
        signal = []
        for i in range(len(f)):
            line_str = f[i].decode("utf-8")
            str_value = line_str.split(',')[2]
            float_val = float(str_value)
            if (i % 5 == 4):
                signal.append(float_val)
                signal.append(float_val)
            signal.append(float_val)
        signal.reverse()
        # signals.append(signal)
    return signal

def load_trader():
    f = open("trader_data\\" + coin_name + "\\trader" + str(best_trader_number) + '.csv', 'rb').readlines()[1:]
    prices = []
    for b_line in f:
        line = b_line.decode("utf-8")
        price = float(line.split(',')[4])
        prices.append(price)
    ratios = []
    for i in range(0,50):
        f = open("trader_data\\" + coin_name + "\\trader" + str(i) +'.csv', 'rb').readlines()[1:]
        tmp = []
        for b_line in f:
            line = b_line.decode("utf-8")
            ratio = float(line.split(',')[3])
            tmp.append(ratio)
        ratios.append(tmp)
    return prices, ratios

prices, ratios = load_trader()

signals = load_signals_crypto()
signals.append(load_signals_5_days_per_week())

signals = np.array(signals)
prices = np.array(prices)
ratios = np.array(ratios)
TRAIN_SIZE = int(length*0.8)

train = []
test = []
train_target = []
test_target = []

for i in range(30, length):
    tmp = []
    tmp.append(prices[i-30:i])
    for currency in signals:
        tmp.append(currency[i-30:i])
    if i < TRAIN_SIZE:
        train.append(tmp)
        train_target.append(ratios[0][i])
    else:
        test.append(tmp)
        test_target.append(ratios[0][i])

train = np.array(train)
train_target = np.array(train_target)
test = np.array(test)
test_target = np.array(test_target)

files = [f for f in listdir("models\\" + coin_name)]
if len(files) == 0:
    model = mod.Sequential()
    model.add(lyrs.Dense(3000, input_shape = (20, 30)))
    model.add(lyrs.Activation('tanh'))
    model.add(lyrs.Dropout(0.25))
    model.add(lyrs.Dense(1500))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dense(750))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dense(375))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dense(185))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dense(1))
    model.add(lyrs.Activation('linear'))
    model.compile(optimizer="adam", loss='mse')
    model.fit(train, 
          train_target, 
          epochs=120,
          batch_size = len(train), 
          verbose=1, 
          validation_data=(test, test_target))
    model.save("models\\" + coin_name)
else:
    model = mod.load_model("models\\" + coin_name)

tot = []
for element in train:
    tot.append(element)
for element in test:
    tot.append(element)
tot = np.array(tot)

predicted = model.predict(tot)


hist = [init_arr]
for i in range(len(predicted)):
    previous = hist[len(hist)-1]
    prediction = predicted[i][0]
    if prediction == 0:
        hist.append([previous[0], previous[1], previous[2], 0, prices[i+1] ]) # raw[1] = prices
    else:
        if prediction > 0:  # we buy
            amount_in_order = previous[0]*prediction
            balance = previous[0] - amount_in_order
            n_barrels = previous[1] + ((amount_in_order * 0.999)/prices[i+1]) # 0.999 is binance fee
        else: # we sell
            amount_in_order = previous[1]*(-prediction)
            balance = previous[0] + amount_in_order*0.999*prices[i+1] # 0.999 is binance fee
            n_barrels = previous[1] - amount_in_order
        estimated_total_balance = balance + n_barrels*prices[i+1]
        hist.append([balance, n_barrels, estimated_total_balance, prediction, prices[i+1] ])

elem = hist[len(hist)-1]
print(elem[0]," | ", elem[1]," | ", elem[2]," | ", elem[3]," | ", elem[4])
# The random trading for bitcoin is ~1800

import matplotlib.pyplot as plt
capital = []
for i in range(1, len(hist)):
    capital.append(hist[i][2][0])
plt.plot(capital)
plt.ylabel('Evolution du capital')
plt.show()