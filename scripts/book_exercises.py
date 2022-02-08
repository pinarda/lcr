"""
Solutions to select problems from Introduction to Data Compression, 5th Edition by Khalid Sayood
"""
import math
from itertools import permutations


# 2.3

def entropy(alphabet, n=1):
    entropy = 0
    for letter, prob in alphabet.items():
        if prob==0:
            continue
        entropy = entropy - prob * math.log(prob, 2)
    return entropy/n

# 2.6
def subseq(s, length, depth=1, cur=[]):
    if depth != length+1:
        tmp = [i * len(s) ** (length-depth) for i in s] * (len(s) ** (depth-1))
        tmp = list(''.join(tmp))
        if not cur:
            cur = [""] * len(tmp)
        for i in range(0,len(tmp)):
            cur[i] = cur[i] + tmp[i]
        return subseq(s, length, depth+1, cur)
    else:
        return cur


if __name__ == "__main__":
    # 2.3
    # part a
    a = {"a1": 0.25, "a2": 0.25, "a3": 0.25, "a4": 0.25}
    print(entropy(a))

    # part b
    b = {"a1": 0.5, "a2": 0.25, "a3": 0.125, "a4": 0.125}
    print(entropy(b))

    # part c
    c = {"a1": 0.505, "a2": 0.25, "a3": 0.125, "a4": 0.12}
    print(entropy(c))

    # 2.6
    # part a
    sequence = "ATGCTTAAGCTGCTTAACCTGAAGCTTCCGCTGAAGAACCTG" \
               "CTGAACCCGCTTAAGCTGAACCTTCTGAAGCTTAACCTGCTT"

    prob_A = sequence.count("A") / len(sequence)
    prob_T = sequence.count("T") / len(sequence)
    prob_G = sequence.count("G") / len(sequence)
    prob_C = sequence.count("C") / len(sequence)

    char_list = set(sequence)
    for i in range(1, 5):
        # actually very inefficient, could just get the set of subsequences in sequence
        # of length n instead of creating a dict with all possible subsequences...
        seqs = subseq(char_list, i)
        probs = {}
        for j in range(0, len(seqs)):
            probs[seqs[j]] = sequence.count(seqs[j]) / len(sequence)
        print(entropy(probs, i))

    # part b
    # Based just on the entropies, we can infer that there is strong correlation
    # between the letters that make the sequence more compressible, as the
    # entropy decreases from nearly 2 bits/letter (which would make sense if the
    # sequence was iid with 4 symbols) to only about 1 bit/letter for 4-th order
    # entropy.