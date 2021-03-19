import csv

f = open('C:\\Users\\chass\\Downloads\\bat-usd-max.csv', 'rb').readlines()[2:]
data = []
for i in range(len(f)):
    line_str = f[i].decode("utf-8")
    data.append(line_str)
data.reverse()

with open('bat-usd-max(2).csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    for element in data:
        spamwriter.writerow(element)