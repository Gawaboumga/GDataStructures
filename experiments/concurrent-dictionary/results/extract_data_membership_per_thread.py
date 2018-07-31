# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
from operator import itemgetter, attrgetter, methodcaller
import re


def extract_data_from_title(file_name, tree_size):
    regex = str(tree_size) +  r"-(\d+).+"
    matches = re.finditer(regex, file_name)
    for match in matches:
        return [int(match.group(1)), int(tree_size)]


def extract_data_from_file(data):
    regex = r"\[ITERATIONS\]        Average time: (\d+\.\d+) us \(~(\d+\.\d+)"
    matches = re.finditer(regex, data)
    results = []
    for match in matches:
        results.append([float(match.group(1)), float(match.group(2))])
    return results


def extract_experience_from_file(data):
    regex = r"\[ RUN      \] (\w+)(Insertion|GetWorstThread|GetThread|PredecessorWarp|PredecessorWorstWarp|Predecessor|Successor|GetWorstWarp|GetWarp)(Thread)?Fixture\.\w+ \(\d+ runs, \d+ iteration(s)? per run\)"
    matches = re.finditer(regex, data)
    results = []
    for match in matches:
        if match.group(2) == "PredecessorWarp":
            results.append([match.group(1), "Predecessor"])
        else:
            results.append([match.group(1), match.group(2)])
    return results


def sort_row(data_row):
    return sorted(data_row, key=itemgetter(0))


def extract(elems, method, strategy):
    tmp = list(filter(lambda e: e[4] == strategy and e[5] == method, elems))
    return sort_row(tmp)


def get_ratio(ones, *others):
    result = []
    for i in range(len(ones)):
        row = []
        x = ones[i][0]
        for j in range(len(others)):
            y = others[j][i][0]
            row.append(x / y)
        result.append(row)
    return result


def get_files(directory_name):
    return [f for f in listdir(directory_name) if isfile(join(directory_name, f))]


all_files = get_files(".")

HEIGHT = 32
data_structures = ["LSM", "BTree", "Warp", "vanEB"]
showing_names = {"LSM": "LSM",
                 "BTree": "BTree",
                 "Warp": "Warp",
                 "vanEB": "HAMT"
                 }
METHOD = "GetThread"
methods = ["Insertion", "GetThread", "GetWorstThread", "Predecessor", "GetWarp", "GetWorstWarp", "PredecessorWorstWarp"]

all_data = []
for file_name in all_files:
    if file_name.endswith(".txt") and file_name.startswith(str(HEIGHT)):
        data_name = extract_data_from_title(file_name, HEIGHT)
        with open(file_name) as f:
            scores = extract_data_from_file(f.read())
            f.seek(0)
            experiments = extract_experience_from_file(f.read())
            for score, experiment in zip(scores, experiments):
                row_data = []
                row_data.append(data_name[0])
                row_data.append(data_name[1])
                row_data.extend(score)
                row_data.extend(experiment)
                all_data.append(row_data)


print(all_data)

# INSERTIONS, HEIGHT, MEAN, SD, STRATEGY, METHOD

total = {}
for data_structure in data_structures:
    for method in methods:
        values = extract(all_data, method, data_structure)
        if method in total:
            total[method].update({data_structure: values})
        else:
            total[method] = {data_structure: values}

#print(total)
for key, value in total[METHOD].items():
    print(key, value)

import math

def get(component, l):
    res = []
    for elem in l:
        if component == "x":
            res.append(elem[0])
        elif component == "y":
            res.append(math.log(elem[2], 2))
            #res.append(elem[0])
        else:
            res.append(elem[3])# ** 0.5)# ** 0.6)
    return res

def lerp(ys, original_value, destination_value):
    for i in range(len(ys)):
        alpha = 1 - i / (len(ys) - 1)
        new_value = alpha * original_value + (1 - alpha) * destination_value
        ys[i] = new_value * ys[i]
    return ys

def draw(ax, liste, name):
    #ax.errorbar(get("x", liste), get("y", liste), get("sd", liste), linestyle='None', label=name)
    xs = get("x", liste)
    ys = list(map(lambda k: k[2], liste))
    print(name, ys)
    """if name == "BTree":
        ys = list(map(lambda k: k * 4/3, ys))
    if name == "LSM":
        lerp(ys, 7/10, 1/10)
    if name == "Warp":
            ys = list(map(lambda k: k * 2.4, ys))
    print(ys)"""
    ys = list(map(lambda k: math.log(k, 2), ys))
    ax.plot(xs, ys, label=name)
    ax.scatter(xs, ys, s=get("sd", liste))

print(METHOD + "," + ",".join(data_structures))
for i in range(len(data_structures)):
    score_row = []
    for key, value in total[METHOD].items():
        if i < len(value):
            scores = value[i]
            score_row.append((scores[2], scores[3], scores[0]))

    print(score_row[0][2], end=",")
    for score in score_row:
        print("{} ({})".format(int(score[0]), int(score[1])), end=",")
    print()
    print(",", end=",")
    btree_score = score_row[0]
    for score in score_row[1:]:
        print("{}".format(round(score[0] / btree_score[0], 2)), end=",")
    print()

import numpy as np
import matplotlib.pyplot as plt

for key, value in total.items():
    print(key, value)
    print("===")

ax = plt.subplot()
for key, value in total[METHOD].items():
    if key == "vanEB":
        continue
    print(key, value)
    draw(ax, value, showing_names[key])
print(total.keys())
for key, value in total["GetWorstThread"].items():
    if key == "LSM":
        draw(ax, value, "LSM worst")
ax.legend()
plt.xlabel("Number of insertions (2^i)")
plt.ylabel("Time in µs (2^j)")
plt.title("{} - Comparison".format("Membership per thread"))
plt.show()

"""

ratios = get_ratio(BTree_times, Binary_times, Group1_times, Group2_times, Group3_times, Group4_times, Warp_times)
#print(ratios)

print("Insertions," + ",".join(data_structures), end=",\n")
for i in range(len(grouped_data[0])):
    print(grouped_data[0][i][3], end=",")
    for d in grouped_data:
        if float('Inf') != d[i][0]:
            print("{} ({})".format(int(d[i][0]), int(d[i][1])), end=',')
        else:
            print("", end=",")
    print()
    print(",", end=",")
    for ratio in ratios[i]:
        print(round(ratio, 2), end=",")
    print()
"""

"""#draw(ax, AOS_4_bytes, "AOS 4")
#draw(ax, SOA_4_bytes, "SOA 4")"""
""""""
"""
draw(ax, AOS_8_bytes, "AOS 8")
draw(ax, SOA_8_bytes, "SOA 8")
"""

#draw(ax, AOS_16_bytes, "AOS 16")
#draw(ax, SOA_16_bytes, "SOA 16")
""""""
#ax.legend()
#plt.xlabel("Number of blocks (2^i)")
#plt.ylabel("Number of warps (2^j)")
#plt.show()
