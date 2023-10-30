# This code implements the DNA clustering algorithm described in the paper "Clustering Billions of Reads for DNA Data Storage" by Cyrus Rashtchian, et al.
# Code by Lleyton Emery, TUM PREP program 2023

# Modules
import random
import numpy
from multiprocessing import Pool
import itertools
import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tracemalloc
import time


# Constants
ALPHABET = ('A', 'C', 'G', 'T') # tuple of available letters 
ONE_GRAMS = ('A', 'C', 'G', 'T') # all possible one-grams from our alphabet
TWO_GRAMS = ('AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT')
THREE_GRAMS = ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 
               'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 
               'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 
               'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 
               'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT')
FOUR_GRAMS = ('AAAA', 'AAAC', 'AAAG', 'AAAT', 'AACA', 'AACC', 'AACG', 'AACT', 'AAGA', 'AAGC', 'AAGG', 'AAGT',
              'AATA', 'AATC', 'AATG', 'AATT', 'ACAA', 'ACAC', 'ACAG', 'ACAT', 'ACCA', 'ACCC', 'ACCG', 'ACCT',
              'ACGA', 'ACGC', 'ACGG', 'ACGT', 'ACTA', 'ACTC', 'ACTG', 'ACTT', 'AGAA', 'AGAC', 'AGAG', 'AGAT',
              'AGCA', 'AGCC', 'AGCG', 'AGCT', 'AGGA', 'AGGC', 'AGGG', 'AGGT', 'AGTA', 'AGTC', 'AGTG', 'AGTT',
              'ATAA', 'ATAC', 'ATAG', 'ATAT', 'ATCA', 'ATCC', 'ATCG', 'ATCT', 'ATGA', 'ATGC', 'ATGG', 'ATGT',
              'ATTA', 'ATTC', 'ATTG', 'ATTT', 'CAAA', 'CAAC', 'CAAG', 'CAAT', 'CACA', 'CACC', 'CACG', 'CACT',
              'CAGA', 'CAGC', 'CAGG', 'CAGT', 'CATA', 'CATC', 'CATG', 'CATT', 'CCAA', 'CCAC', 'CCAG', 'CCAT',
              'CCCA', 'CCCC', 'CCCG', 'CCCT', 'CCGA', 'CCGC', 'CCGG', 'CCGT', 'CCTA', 'CCTC', 'CCTG', 'CCTT',
              'CGAA', 'CGAC', 'CGAG', 'CGAT', 'CGCA', 'CGCC', 'CGCG', 'CGCT', 'CGGA', 'CGGC', 'CGGG', 'CGGT',
              'CGTA', 'CGTC', 'CGTG', 'CGTT', 'CTAA', 'CTAC', 'CTAG', 'CTAT', 'CTCA', 'CTCC', 'CTCG', 'CTCT',
              'CTGA', 'CTGC', 'CTGG', 'CTGT', 'CTTA', 'CTTC', 'CTTG', 'CTTT', 'GAAA', 'GAAC', 'GAAG', 'GAAT',
              'GACA', 'GACC', 'GACG', 'GACT', 'GAGA', 'GAGC', 'GAGG', 'GAGT', 'GATA', 'GATC', 'GATG', 'GATT',
              'GCAA', 'GCAC', 'GCAG', 'GCAT', 'GCCA', 'GCCC', 'GCCG', 'GCCT', 'GCGA', 'GCGC', 'GCGG', 'GCGT',
              'GCTA', 'GCTC', 'GCTG', 'GCTT', 'GGAA', 'GGAC', 'GGAG', 'GGAT', 'GGCA', 'GGCC', 'GGCG', 'GGCT',
              'GGGA', 'GGGC', 'GGGG', 'GGGT', 'GGTA', 'GGTC', 'GGTG', 'GGTT', 'GTAA', 'GTAC', 'GTAG', 'GTAT',
              'GTCA', 'GTCC', 'GTCG', 'GTCT', 'GTGA', 'GTGC', 'GTGG', 'GTGT', 'GTTA', 'GTTC', 'GTTG', 'GTTT',
              'TAAA', 'TAAC', 'TAAG', 'TAAT', 'TACA', 'TACC', 'TACG', 'TACT', 'TAGA', 'TAGC', 'TAGG', 'TAGT',
              'TATA', 'TATC', 'TATG', 'TATT', 'TCAA', 'TCAC', 'TCAG', 'TCAT', 'TCCA', 'TCCC', 'TCCG', 'TCCT',
              'TCGA', 'TCGC', 'TCGG', 'TCGT', 'TCTA', 'TCTC', 'TCTG', 'TCTT', 'TGAA', 'TGAC', 'TGAG', 'TGAT',
              'TGCA', 'TGCC', 'TGCG', 'TGCT', 'TGGA', 'TGGC', 'TGGG', 'TGGT', 'TGTA', 'TGTC', 'TGTG', 'TGTT',
              'TTAA', 'TTAC', 'TTAG', 'TTAT', 'TTCA', 'TTCC', 'TTCG', 'TTCT', 'TTGA', 'TTGC', 'TTGG', 'TTGT',
              'TTTA', 'TTTC', 'TTTG', 'TTTT')
BLOCK_SIZE = 22 # binary signatures for each sequence calculated in blocks of this size

# More constants fused in the data generation process
NUMBER_OF_REFERENCES = 500 # Number of original sequences of DNA that noisy copies will be made of
REFERENCE_LENGTH = 150 # the length of each original reference
# For each reference, the number of copies to be generated will come from N(mean, var)
MEAN_COPIES = 10.0
STAND_DEV_COPIES = 2.0 # note this is the standard deviation, not the variance
# The following are the probabilities of each kind of error occurring when making a copy of a reference string
INSERTION_PROB = 0.017
DELETION_PROB = 0.02
SUB_PROB = 0.022
MATCH_PROB = 1 - (INSERTION_PROB + DELETION_PROB + SUB_PROB) # probability of no error occurring for a given letter
# If testing outlier robustness, outlier strands are generated using the following error rate
OUTLIER_ERROR = 0.2 # this is epsilon prime in the billions paper

# This function is used if you want to read sequences from a file instead of generating them
# Reads from a file of clusters, where each cluster is separated by a line of '='
# Returns a list of lists, where each list in the returned list is a cluster of sequences. This list of list contains the underlying clusters
def read_file():
    file = open("microsoft_sequences.txt", "r")
    number_clusters = 0
    current_cluster = []
    underlying_clusters = []
    for line in file:
        if number_clusters == 500: # can change 500 depending on how many clusters you want
            break
        if '=' in line:
            number_clusters += 1
            underlying_clusters.append(current_cluster + [])
            current_cluster.clear()
        else:
            current_cluster.append(line.strip())
    file.close()
    return underlying_clusters

# This function is used if you want to generate sequences instead of read them from a file
# Generates all of our random reference strings, returns as a list
def construct_references():
    references = [] # list of all reference sequences
    current_ref = [] # the current reference
    for _ in range(NUMBER_OF_REFERENCES):
        for _ in range(REFERENCE_LENGTH):
            current_ref.append(ALPHABET[random.randint(0, 3)]) # choose a random letter to add
        references.append("".join(current_ref)) # join all letters of our reference into a string, and add to list of references
        current_ref.clear() # clear current reference so we can have an empty string for next iteration
    return references

# This function is used if you want to generate sequences instead of read them from a file
# For each reference, noisy copies are constructed
# Returns a list of lists, where each list in the returned list is a cluster of sequences. This list of list contains the underlying clusters
def construct_copies(references):
    all_copies = [] # list of lists of copies
    reference_copies = [] # list of all copies of a given reference
    for ref in references:
        number_of_copies = int(numpy.random.normal(loc=MEAN_COPIES, scale=STAND_DEV_COPIES, size=None))
        if number_of_copies > 0:
            for _ in range(number_of_copies):
                current_copy = []
                for letter in ref:
                    roll_die = True # our boolean to see if we should calculate the prob again
                    while roll_die:
                        # get random value to determine which type of error (or no error) occurs
                        rand = random.randint(1, 1000) / 1000
                        # insertions
                        if rand <= INSERTION_PROB:
                            current_copy.append(ALPHABET[random.randint(0, 3)]) # choose a random letter to add
                        # substitutions
                        elif rand <= INSERTION_PROB + SUB_PROB:
                            new_letter = ALPHABET[random.randint(0, 3)] # choose a random new letter to add
                            while new_letter == letter:
                                new_letter = ALPHABET[random.randint(0, 3)] # again, choose a random new letter to add
                            current_copy.append(new_letter) # add the new letter
                            roll_die = False
                        # matches
                        elif rand <= INSERTION_PROB + SUB_PROB + MATCH_PROB:
                            current_copy.append(letter) # add the letter we are on
                            roll_die = False
                        # deletions, we don't append the letter
                        else:
                            roll_die = False
                reference_copies.append("".join(current_copy)) # add our copy to our list of copies
                current_copy.clear() # clear so we can make a new copy
        all_copies.append(reference_copies + []) # add the list of copies for a specific reference
        reference_copies.clear()
    return all_copies

# This function can be used to turn our list of lists (i.e. the underlying clustering) into a list of all of our sequences that we can perform
# the clustering algorithm on
# Takes in a list of lists and flattens it into a list
def flatten(lst):
    return [item for sublist in lst for item in sublist]

# If testing outlier robustness, generated outliers which are completely random
# I am making outliers that are the same length as the references, and every single letter is random
# Returns list of outlier sequences
def generate_outliers():
    outliers = [] # list of all outliers sequences
    current_outlier = [] # the current outlier
    # the size of our outlier set is espilon prime times k, or outlier_error times number_refs (same as # of clusters)
    for _ in range(int(OUTLIER_ERROR * NUMBER_OF_REFERENCES)):
        for _ in range(REFERENCE_LENGTH):
            current_outlier.append(ALPHABET[random.randint(0, 3)]) # choose a random letter to add
        outliers.append("".join(current_outlier)) # join all letters of our outlier into a string, and add to list of outliers
        current_outlier.clear() # clear current outlier so we can have an empty string for next iteration
    return outliers

# This is our main function which performs the clustering
# S is our list of input sequences
# r is the threshold when checking Levenshtein distance
# q is the length of our q-grams
# w is the first parameter for our hash function
# l is the second parameter for our hash function
# theta_low and theta_high are the low and high thresholds when comparing q-gram distance
# comm_steps is the number of global communication steps performed, where the clusters are redistributed among the cores
# local_steps is the number of local computation/comparison steps performed within each core for a given comm_step
def clusters(S, r, q, w, l, theta_low, theta_high, comm_steps, local_steps):
    C = list(map(lambda el:[el], S)) + [] # initialize clustering as singletons, where each sequence in S is its own cluster
    signatures = get_binary_signatures_blocked(S, q) # make binary signatures for all of the sequences, where signatures is a dictionary
    for i in range(comm_steps):
        print(f'global {i}') # checks what global step we are on
        global_anchor = get_hash_function(w) # get the hash function
        global_cluster_groups = group_clusters(C, global_anchor, w, l) # "distribute" the sequences among the cores. I didn't actually distribute them
                                                                       # to different cores, I just put them into different groups using a list of lists
        for j in range(local_steps):
            print(f'local {j}') # checks what local step we are on
            for group in global_cluster_groups:
                # seeing which sequences need to be put into buckets to potentially be compared can actually be done multiple times, in this case 2
                local_anchor_1 = get_hash_function(w) # get the first hash function
                local_anchor_2 = get_hash_function(w) # get the second has function
                # local_cluster_buckets is a list of lists of lists, where we are putting clusters (lists) into the same bucket (the same list)
                # local_rep_buckets is a list of the representative sequences from the bucketed clusters, which are in the same order as the clusters
                # in local_cluster_buckets
                local_cluster_buckets, local_rep_buckets = bucket_clusters(group, local_anchor_1, local_anchor_2, w, l)
                # compare the representatives in a bucket and merge their corresponding clusters if needed to make a new clustering C
                C = do_comparisons(C, local_rep_buckets, local_cluster_buckets, signatures, q, theta_low, theta_high, r)
    return C

# Create binary signatures for each sequence, without using the blocking method
# Returns a dictionary with the key as the sequence and the value as its binary signature
def get_binary_signatures_unblocked(S, q):
    binary_sigs = {}
    current_sig = []
    if q == 1:
        q_grams = ONE_GRAMS
    elif q == 2:
        q_grams = TWO_GRAMS
    elif q == 3:
        q_grams = THREE_GRAMS
    else:
        q_grams = FOUR_GRAMS
    for x in S:
        for gram in q_grams:
            if gram in x:
                current_sig.append('1')
            else:
                current_sig.append('0')
        binary_sigs[x] = "".join(current_sig + [])
        current_sig.clear()
    return binary_sigs

# Create binary signatures for each sequence, using the blocking method
# Returns a dictionary with the key as the sequence and the value as its binary signature
def get_binary_signatures_blocked(S, q):
    binary_sigs = {}
    current_sig = []
    if q == 1:
        q_grams = ONE_GRAMS
    elif q == 2:
        q_grams = TWO_GRAMS
    elif q == 3:
        q_grams = THREE_GRAMS
    else:
        q_grams = FOUR_GRAMS
    for x in S:
        blocks = [x[i:i+BLOCK_SIZE] for i in range(0, len(x), BLOCK_SIZE)]
        for block in blocks:
            for gram in q_grams:
                if gram in block:
                    current_sig.append('1')
                else:
                    current_sig.append('0')
        binary_sigs[x] = "".join(current_sig + [])
        current_sig.clear()
    return binary_sigs

# A function to generate a random permutation of the alphabet of size w
# Our returned "anchor" is our hash function
def get_hash_function(w):
    # Choose an anchor
    anchor = []
    for _ in range(w):
        # Append a random letter
        anchor.append(ALPHABET[random.randint(0, 3)])
    return "".join(anchor)

# Try to find a hash value in our sequence using our "anchor"
# rep is the representative sequence from a cluster
def get_hash_value(rep, anchor, w, l):
    start_index = rep.find(anchor) # returns the index where the anchor first occurs in the representative (-1 if not found)
    end_index = min(len(rep), start_index + w + l) # finds the last index of our substring we want to find
    if start_index != -1:
        return rep[start_index:end_index] # end_index is not inclusive here
    else:
        return '' # return empty substring if none was found in the representative

# Pick a random representative sequence from a cluster
def get_representative(cluster):
    return cluster[random.randint(0, len(cluster) - 1)]

# Redistribute the clusters into groups, or "cores", after each comm_step is done 
def group_clusters(clustering, anchor, w, l):
    hash_dict = {}
    empty_key = 0 # this will be used to add clusters that had an empty hash value to the dictionary
    for cluster in clustering:
            rep = get_representative(cluster)
            hash_val = get_hash_value(rep, anchor, w, l)[:2]
            # if hash value already in dictionary, add this cluster to the list with other clusters
            if hash_val in hash_dict.keys():
                hash_dict[hash_val].append(cluster)
            # if not, and hash value is empty, then create new entry
            elif hash_val:
                hash_dict[hash_val] = [cluster]
            # if hash value is empty, then add the cluster with a new key
            else:
                hash_dict[empty_key] = [cluster]
                empty_key += 1 # add 1 so the next cluster with an empty hash value is added with a different key
    return list(hash_dict.values())

# For each local computation step, we want to bucket the clusters and their corresponding representatives to see which need to be compared
# We want to group the clusters together that will be compared and also the representatives from those clusters
def bucket_clusters(clusters, anchor1, anchor2, w, l):
    hash_dict = {}
    rep_dict = {}
    empty_key = 0 # this will be used to add clusters that had an empty hash value to the dictionary
    for cluster in clusters:
        rep = get_representative(cluster)
        hash_val1 = get_hash_value(rep, anchor1, w, l)
        hash_val2 = get_hash_value(rep, anchor2, w, l)
        # if hash value already in dictionary, add this cluster to the list with other clusters
        if hash_val1 + hash_val2 in hash_dict.keys():
            hash_dict[hash_val1 + hash_val2].append(cluster)
            rep_dict[hash_val1 + hash_val2].append(rep)
        # if not, and hash value is empty, then create new entry
        elif hash_val1 + hash_val2:
            hash_dict[hash_val1 + hash_val2] = [cluster]
            rep_dict[hash_val1 + hash_val2] = [rep]
        # if hash value is empty, then add the cluster with a new key
        else:
            hash_dict[empty_key] = [cluster]
            rep_dict[empty_key] = [rep]
            empty_key += 1 # add 1 so the next cluster with an empty hash value is added with a different key
    return list(hash_dict.values()), list(rep_dict.values())

# buckets is a list of representative sequences grouped locally to be compared (they produced the same hash value)
# clusters is a list of clusters that correspond to the grouped representatives (this list is in the same order as buckets)
def do_comparisons(C, buckets, clusters, binary_sigs, q, theta_low, theta_high, r):
    # We need to keep track of which cluster we are on so we can merge properly
    # c_index_1 and c_index_2 incerase by 1 each time we count a representative. They ensure we get the right cluster when we are dealing with its
    # corresponding representative
    already_compared = [] # List of sets of representatives that have already been compared
    # go through each bucket in our list of buckets
    for z in range(len(buckets)):
        c_index_1 = -1
        # for this bucket, go through each representative in the bucket and compare to each other representative in the bucket
        for i in range(len(buckets[z])):
            c_index_1 += 1
            c_index_2 = -1 # initialize here so it resets every time we want to go through the same bucket again
            # go through bucket and compare each representative to each other representative
            for j in range(len(buckets[z])):
                c_index_2 += 1
                q_gram_dist = 0
                # we don't want to check the same exact representatives against each other, and we don't want to compare ones we already have
                if i != j and {buckets[z][i], buckets[z][j]} not in already_compared:
                    already_compared.append({buckets[z][i], buckets[z][j]})
                    # loop over the indices of the binary signatures to compare them
                    # 4 ^ q is the length of the binary signatures
                    for k in range(min(len(binary_sigs.get(buckets[z][i])), len(binary_sigs.get(buckets[z][j])))):
                        # if a given number in the signatures is not the same, increase q-gram distance by 1
                        if int(binary_sigs.get(buckets[z][i])[k]) != int(binary_sigs.get(buckets[z][j])[k]):
                            q_gram_dist += 1
                    # first comparison step using theta_low
                    if q_gram_dist <= theta_low:
                        # make sure the clusters are in C before we merge them
                        if clusters[z][i] in C and clusters[z][j] in C:
                            C.remove(clusters[z][i])
                            if clusters[z][j] in C:
                                C.remove(clusters[z][j])
                            C.append(clusters[z][i] + clusters[z][j])
                            # add the new representatives and clusters to our lists of representatives and clusters to potentially compare them 
                            buckets[z].append(buckets[z][i])
                            clusters[z].append(clusters[z][i] + clusters[z][j])
                    # second comparison step, check q-gram distance first using theta_low, then check Levenshtein distance if needed
                    elif q_gram_dist <= theta_high:
                        lev_dist = calculate_lev_distance(buckets[z][i], buckets[z][j])
                        if lev_dist <= r:
                            if clusters[z][i] in C and clusters[z][j] in C:
                                C.remove(clusters[z][i])
                                if clusters[z][j] in C:
                                    C.remove(clusters[z][j])
                                C.append(clusters[z][i] + clusters[z][j])
                                # add the new representatives and clusters to our lists of representatives and clusters to potentially compare them 
                                buckets[z].append(buckets[z][i])
                                clusters[z].append(clusters[z][i] + clusters[z][j])
        already_compared.clear()
    return C

# In the Billions paper, the authors mention using a different way to compare sequences by comparing adjacent parameters
# I never implemented that, but this function was the start of that
def do_comparisons_adjacent(C, buckets, clusters, binary_sigs, q, theta_low, theta_high, r):
    pass

# Calculated the Levenshtein distance between two strings
# Note: we only need to check if the Levenshtein distance is below or above a threshold, so potentially some overhead could be reduced if we didn't
# calculated the Levenshtein distance each time and instead constantly checked if it had already exceeded the threshold, at which point we wouldn't need
# to continue calculating the Levenshtein distance. This function calculates it in full every time, though.
def calculate_lev_distance(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    # printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]

# This accuracy measurement looks at how many sequences we put in the wrong cluster, so a higher classification error is worse
# This accuracy measurement is specifically designed to work only if the number of clusters in our algorithm's clustering is the same as the number of
# clusters in the underlying clustering
def get_classification_error(our_clustering, underlying, inputs):
    # we want to find the minimum number of wrong labels we can get, so we will replace this with our new values
    # every time we find a permutation that is better
    min_wrong = len(inputs)

    # go through all permutations
    for perm in list(itertools.permutations(range(10))):
        # number of wrong labels should be reset to 0 for every element of the permutation
        wrong_labels = 0
        # we will compare the first index of the perm to index 0 of the underlying cluster, and so on
        for i in range(10):
            # go through each sequence in our cluster and see if it is in the underlying cluster,
            # because we want to count how many sequences in our clusters are not in the given underlying cluster
            for sequence in our_clustering[perm[i]]:
                if sequence not in underlying[i]:
                    wrong_labels += 1
            if wrong_labels < min_wrong:
                min_wrong = wrong_labels
    
    return min_wrong / len(inputs)

# This is the implementation of the accuracy measurement explained in the Billions paper
# find all subsets of UNDERLYING of size len(our_clustering)
# permute each subset to find all perms of all subsets
# go thru each of these perms
# for a given perm, go thru i in range(len(UNDERLYING))
# if i < len(our_clustering), compare UNDERLYING[i] to our_clustering[perm[i]]
def get_accuracy(our_clustering, gamma, underlying):
    best_accuracy = 0 # this will be replaced by better and better accuracies until we find the best permutation
    max_value = max(len(underlying), len(our_clustering))
    min_value = min(len(underlying), len(our_clustering))
    if len(our_clustering) > len(underlying):
        large_clustering = our_clustering + []
        small_clustering = underlying + []
    else:
        large_clustering = underlying + []
        small_clustering = our_clustering + []
    # get all permutations of range indexes
    perms = find_injective_maps(range(max_value), min_value)
    for perm in perms:
        sum_of_indicators = 0
        # we will compare the given underlying cluster to the given cluster of our clustering
        for i in range(min_value):
            # first check if all of the sequences in the given cluster from our algo are in the 
            # given underlying cluster
            # if the underlying cluster is largest, it uses the permutations for indexing
            if max_value == len(underlying):
                if all(sequence in underlying[perm[i]] for sequence in our_clustering[i]):
                    # check if the intersection is big enough
                    intersection = [value for value in underlying[perm[i]] if value in our_clustering[i]]
                    if len(intersection) >= gamma * len(underlying[perm[i]]):
                        sum_of_indicators += 1
            # if our clustering is larger, it uses the permutations for indexing
            else:
                if all(sequence in underlying[i] for sequence in our_clustering[perm[i]]):
                    # check if the intersection is big enough
                    intersection = [value for value in underlying[i] if value in our_clustering[perm[i]]]
                    if len(intersection) >= gamma * len(underlying[i]):
                        sum_of_indicators += 1
        # after you have gone through one perm, calculate the accuracy
        current_accuracy = sum_of_indicators / len(underlying)
        # replace our best accuracy if our current accuracy is better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
    return best_accuracy

# This seems like the best way to test accuracy
# It is very similar to the accuracy measurement described in the Billions paper, but with some differences
# If we use gamma = 1.0, we can't have one cluster our underlying clustering match with more than one cluster in our algorithm's clustering
# Therefore, we can just compare every cluster in our algorithm's clustering to every algorithm in the underlying clustering to see if any match
# This avoids the overhead of calculating every possible permutation clusters in our clustering to clusters in the underlying clustering
def get_accuracy_simple(our_clustering, gamma, underlying):
    new_our = our_clustering + []
    new_under = underlying + []
    sum_of_indicators = 0
    # we will compare the given underlying cluster to the given cluster of our clustering
    for our_cluster in new_our:
        for underlying_cluster in new_under:
            if all(sequence in underlying_cluster for sequence in our_cluster):
                # check if the intersection is big enough
                # intersection = [value for value in underlying_cluster if value in our_cluster]
                if len(our_cluster) >= gamma * len(underlying_cluster):
                    # new_our.remove(our_cluster)
                    # new_under.remove(underlying_cluster)
                    sum_of_indicators += 1
    # after you have gone through every cluster in our clustering, calculate accuracy
    # print(f'length of underlying is {len(underlying)}')
    accuracy = sum_of_indicators / len(underlying)
    return accuracy

# Finds all permutations for the accuracy measurements that require permutations
def find_injective_maps(s, n):
    perms = []
    subsets = list(itertools.combinations(s, n))
    for subset in subsets:
        perms.append(list(itertools.permutations(subset)))
    # returned the list of perms flattened
    return [item for sublist in perms for item in sublist] 

# This main function initializes our data, clusters it, then prints results
def main():
    # If this boolean is true, we read our sequences from a file. If false, we generated data.
    READ = True
    if READ:
        underlying = read_file()
    else:
        references = construct_references()
        underlying = construct_copies(references)
    inputs = flatten(underlying)
    # Helpful things to print to see what our data looks like
    print('Underlying length', len(underlying))
    print('Inputs length', len(inputs))
    print(underlying[0])
    print('-----')
    print(inputs[0])
    # Call our main clustering function, start timer
    start = time.time()
    # These parameters work well for clustering the generated data
    our_clustering = clusters(inputs, 26, 3, 4, 4, 110, 130, 26, 30)
    # End timer
    end = time.time()
    simple_accuracy = get_accuracy_simple(our_clustering, 1.0, underlying)
    print('Our simple accuracy is', simple_accuracy)
    print(f'Our runtime is {end - start} seconds')

# Call the main function
if __name__ == '__main__':
    main()
