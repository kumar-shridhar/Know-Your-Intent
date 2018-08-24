import csv
import json

with open("/home/dash/projects/imli/data/datasets/AskUbuntuCorpus.json", "r") as f:
    dataset = json.load(f)
    all = [[sample["text"], sample["intent"]] for sample in dataset["sentences"]]

with open('test.txt') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    all_rows = list(readCSV)

test, train = [], []
for t in all:
    found = False
    for tes in all_rows:
        if tes[0] == t[0]:
            test.append(t)
            found = True
            break
    if not found:
        train.append(t)

print(len(train))
print(len(test))

with open("train.csv", "w") as f:
    f.write("\n".join(["\t".join(sample) for sample in train]))

with open("test.csv", "w") as f:
    f.write("\n".join(["\t".join(sample) for sample in test]))