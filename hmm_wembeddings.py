from __future__ import division
from collections import Counter
import string
import operator
import random
import sys
import argparse
import math
import numpy as np
import pandas


def alpha_prob(wordlist, states):
    global prob_sum
    psum = 0
    time = 0
    nstates = len(states)
    alpha = np.zeros([len(wordlist)+1,nstates])
    for i in states:
        alpha[time][i] = Pi[i]
    time = 1
    for word in wordlist:
        for tstate in states:
            for fstate in states:
                alpha[time][tstate] += alpha[time-1][fstate]*transitions.item(fstate,tstate)*em_prob[fstate][word]
        time += 1
    for state in states:
        psum += alpha[time-1][state]
    prob_sum += psum
#f.write("String probability from Alphas: %8.10f\n" % psum)
    return alpha


def beta_prob(wordlist, states):
    last = len(wordlist) + 1
    nstates = len(states)
    beta = [[0 for x in range(nstates)] for y in range(last)]
    psum = 0
    for state in states:
        beta[last-1][state] = 1
    for i in range((last-2),-1,-1):
        for fstate in states:
            for tstate in states:
                beta[i][fstate] += beta[i+1][tstate]*transitions.item(fstate,tstate)*em_prob[fstate][wordlist[i]]
    for state in states:
        psum += Pi[state]*beta[0][state]
#f.write("String Probability from Betas: %8.10f\n\n" % psum)
    return beta

def soft_counts(alpha, beta, wordlist, states):
    global soft_counts_t, init_soft_counts_t
    time = 0
    strlen = len(wordlist)
    psum = 0
    for state in states:
        psum += alpha[strlen][state]
    for word in wordlist:
        lstates = []
        for fstate in states:
            for tstate in states:
                s = (alpha[time][fstate]*transitions.item(fstate,tstate)*em_prob[fstate][word]*beta[time+1][tstate])/(psum)
                lstates.append(s)
        time += 1
        if word in soft_counts_t:
            soft_counts_t[word] = [a + b for a, b in zip(soft_counts_t[word], lstates)]
        else:
            soft_counts_t[word] = lstates
        if (time == 1):
            if word in init_soft_counts_t:
                init_soft_counts_t[word] = [a + b for a, b in zip(init_soft_counts_t[word], lstates)]
            else:
                init_soft_counts_t[word] = lstates
    return init_soft_counts_t, soft_counts_t

def calc_a(soft_counts_t, states): # Maximization
    counts = soft_counts_t.values()
    res = [sum(i) for i in zip(*counts)]
    sums = []
    p = len(res)/len(states)
    j = 0
    while j <= (len(res)-1):
        s = 0
        count = 1
        while p >= count:
            s += res[j]
            j += 1
            count += 1
        sums.append(s)
    Alist = []
    j = 0
    s = 0
    while j <= (len(res)-1):
        count = 1
        while p >= count:
            Alist.append(res[count-1]/sums[s])
            j += 1
            count += 1
        s += 1
        sums.append(s)
    return Alist

def calc_b(soft_counts_t, b_prob, states): # Maximization
    counts=soft_counts_t.values()
    res = [sum(i) for i in zip(*counts)]
    fsums = []
    tsums = []
    p = len(res)/len(states)
    j = 0
    while j <= (len(res)-1):
        s = 0
        count = 1
        while p >= count:
            s += res[j]
            j += 1
            count += 1
        fsums.append(s) # Sum from each State
    for key in soft_counts_t:
        j = 0
        while j <= (len(res)-1):
            s = 0
            count = 1
            while p >= count:
                s += soft_counts_t[key][j]
                j += 1
                count += 1
            tsums.append(s)
        b_prob[key] = [x/y for x,y in zip(tsums, fsums)]
        tsums = []
    return b_prob

def calc_pi(init_soft_counts_t, z, states): # Maximization
    counts = init_soft_counts_t.values()
    res = [sum(i) for i in zip(*counts)]
    sums = []
    p = len(res)/len(states)
    j = 0
    while j <= (len(res)-1):
        s = 0
        count = 1
        while p >= count:
            s += res[j]
            j += 1
            count += 1
        sums.append(s)
    pi = [x/z for x in sums]
    return pi

def state_filter(b_prob, states):
    s = dict()
    for state in states:
        s[state] = dict()
        for key in b_prob:
            if (b_prob[key][state] > 0.001):
                s[state].update({key:b_prob[key][state]})
        s[state] = dict((x, y) for x, y in sorted(s[state].items(), key=operator.itemgetter(1), reverse=True))
    return s

def print_em(state_em):
    f.write("\n-------- Emission Probabilities ------\n")
    for state in state_em:
        f.write("State %d:\n" % state)
        sorted_emdict = sorted(state_em[state].items(), key=operator.itemgetter(1), reverse=True)
        for key in sorted_emdict:
            f.write("%s\t%8.3f\n" % (key[0], key[1]))

def calc_purity(state_em, tags):
    tags_to_prob = dict()
    for state in state_em:
        tags_to_prob[state] = dict()
        for key in state_em[state]:
            if key in tags:
                if (tags[key]) in tags_to_prob[state]:
                    tags_to_prob[state].update({tags[key]:state_em[state][key]+tags_to_prob[state][tags[key]]})
                else:
                    tags_to_prob[state].update({tags[key]:state_em[state][key]})
    avg_purity = 0
    for state in tags_to_prob:
        maximum = max(tags_to_prob[state].iteritems(), key=operator.itemgetter(1))[1]
        ssum = sum(tags_to_prob[state].values())
        avg_purity += maximum/ssum
    return avg_purity

def print_transitions(transitions, states):
    f.write("\n-------- Transition Probabilities ------\n")
    row_labels = []
    for state in states:
        row_labels.append('State %d' % state)
    df = pandas.DataFrame(transitions, columns=row_labels, index=row_labels)
    np.savetxt(f, df, fmt='%8.3f')

def print_purity(avg_purity):
    f.write("\n# states\taverage purity\n")
    for key in avg_purity:
        f.write("%s \t %8.3f\n" % (key[0], (key[1]/key[0])))



avg_purity = []
f=open('output.txt', 'w')
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbosity:
    f=sys.stdout
for j in range(3,11): # States 3 - 10
    state_em = dict()
    b_prob = dict((k,[]) for k in range(1))
    b_prob.pop(0)
    states = []
    for k in range(0,j):
        states.append(k)
    Pi = np.zeros(j)
    transitions = np.zeros([j,j])
    Pi.fill(1/j)
    transitions.fill(1/j)
    for data_iter in range(1,4): # Read data 3 times for each state
        with open('toy.txt', 'r') as file:
            words = file.read().lower()
            sentences = words.split('\n')
        file.close()
        data = words.replace('\n', ' ').split(' ')
        z = len(sentences)
        l_counter = Counter(data)
        prob_dict = dict((k,round(val/sum(l_counter.values()),5)) for k, val in l_counter.items())
        emission_prob = sorted(prob_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_pdict = dict((x, y) for x, y in emission_prob)
        em_prob = []
        em_prob.append(sorted_pdict)
        l1_counter = Counter(data[10:344])
        prob1_dict = dict((k,round(val/sum(l_counter.values()),5)) for k, val in l1_counter.items())
        emission1_prob = sorted(prob1_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted1_pdict = dict((x, y) for x, y in emission1_prob)
        em_prob.append(sorted1_pdict)
        for k in range(2,j):
            if (k % 2 == 0):
                em_prob.append(sorted_pdict)
            else:
                em_prob.append(sorted1_pdict)
        prob_sum = 0
        for i in range(1,400): # 400 iterations for each E-M loop
            prob_sum = 0
            soft_counts_t = dict((k, []) for k in range(1))
            init_soft_counts_t = dict((k, []) for k in range(1))
            soft_counts_t.pop(0)
            init_soft_counts_t.pop(0)
            for sentence in sentences:
                wordlist = sentence.split(' ')
                alpha = alpha_prob(wordlist, states)
                beta = beta_prob(wordlist, states)
                init_soft_counts_t, soft_counts_t  = soft_counts(alpha, beta, wordlist, states)
            a_prob = calc_a(soft_counts_t, states) # Re-computes A
            b_prob = calc_b(soft_counts_t, b_prob, states) # Re-computes B
            pi_prob = calc_pi(init_soft_counts_t, z, states) # Re-computes Pi
            transitions = np.asmatrix(np.reshape(a_prob, (j, j)))
            for key in b_prob:
                for state in states:
                    em_prob[state][key] = b_prob[key][state]
                    em_prob[state][key] = b_prob[key][state]
            for state in states:
                Pi[state] = pi_prob[state]
        tags = {'?': 'punc', '.': 'punc', 'a':'art', 'big':'adj', 'bit':'verb', 'bite':'verb', 'boy':'noun', 'can':'aux','cat':'noun','did':'aux', 'dog':'noun', 'heard':'verb', 'I': 'pronoun', 'is':'verb', 'little':'adj', 'may':'aux', 'not':'neg', 'saw':'verb', 'see':'verb', 'slept':'verb', 'talk':'verb', 'talked':'verb', 'the':'art', 'to':'prep', 'we':'pronoun', 'what':'wh-word', 'when':'wh-word', 'who':'wh-word', 'you':'pronoun'}
        state_em = state_filter(b_prob, states) # returns dict with tags-to-emissions
        avg_purity.append((len(states), calc_purity(state_em, tags)))
    print_em(state_em)
    print_transitions(transitions, states)
print_purity(avg_purity)
f.close()
