from collections import *

def train_char_lm(fname, order=4):
    with open (fname, 'r', encoding='utf-8') as file:
        data = file.read()

    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm

from random import random

def generate_letter(lm, history, order):
        history = history[-order:]
        dist = lm[history]
        x = random()
        for c,v in dist:
            x = x - v
            if x <= 0: return c

def generate_text(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)


def gen_text (lm, seed, nletters=1000):
    """Same as generate_text, except now handles keys its not seen before"""
    for k in lm.keys():
        order = len(k)

    if len(seed) < order:
        seed = ' ' * order + seed

    history = seed[-order:]
    out = []
    for i in range(nletters):
        if history not in lm:
            if history.lower() in lm:
                history = history.lower()
                break
            def find_suitable_replacement():
                for removed_letters in range (1, order):
                    for k, v in lm.items():
                        if k[-order+removed_letters:] == history[-order+removed_letters:]:
                            return k
            history = find_suitable_replacement()
        c = generate_letter(lm, history, order)
        history = history[-order+1:] + c
        out.append(c);
    return "".join(out)

import json

def train (dataset, name='model',lookback=8):

    lm = train_char_lm('../datasets/' + dataset, order=8)

    with open (name + '_model.json', 'w') as f:
        f.write(json.dumps(lm))

train ('drseuss.txt', 'drSeuss')
train ('wizard-of-oz.txt', 'WoOz')
train ('nancy-drew.txt', 'nancy')
train ('alice-in-wonderland.txt', 'AiW')
train ('shakespeare_input.txt', 'shakespeare')
train ('hamlet.txt', 'hamlet', lookback=20)


with open ('hamlet_model.json', 'r') as f:
    datastore = json.loads(f.read())

gen_text(lm, 8, 'running', nletters=40)
gen_text(datastore, 'Big oof', nletters=400)
