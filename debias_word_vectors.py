import numpy as np

def read_glove_vecs(glove_file):
    """
    Reads GloVe word embeddings from a file
    """

    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)
    """

    distance = 0.0

    # Compute the dot product between u and v
    dot = np.dot(u.T, v)
    # Compute the L2 norm of u
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    # Compute the L2 norm of v
    norm_v = np.sqrt(np.sum(np.power(v, 2)))
    # Compute the cosine similarity
    cosine_similarity = np.divide(dot, norm_u * norm_v)

    return cosine_similarity

# Neutralize bias for non-gender specific words
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    # Select word vector representation of "word"
    e = word_to_vec_map[word]

    # Compute e_biascomponent
    e_biascomponent = np.divide(np.dot(e, g), np.linalg.norm(g) ** 2) * g

    # Neutralize e by substracting e_biascomponent from it
    e_debiased = e - e_biascomponent

    return e_debiased

# Equalize bias for gender specific words
def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    # Select word vector representation of "word"
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # Compute the mean of e_w1 and e_w2
    mu = (e_w1 + e_w2) / 2.0

    # Compute the projections of mu over the bias axis and the orthogonal axis
    mu_B = np.divide(np.dot(mu, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
    mu_orth = mu - mu_B

    # Apply the formula
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * np.divide(e_w1B - mu_B, np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * np.divide(e_w2B - mu_B, np.abs(e_w2 - mu_orth - mu_B))

    # Debias by equalizing e1 and e2 to the sum of their corrected projections
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2

# Read word embeddings
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# g vector roughly encodes the concept of "gender"
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print("Gender concept from GloVe vectors:\n")
print(g)

print("\nCosine similarity between a given word and the gender concept\n")
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))

e = "receptionist"
print("\ncosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("\ncosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

print("\ncosine similarities before equalizing:")
print("\ncosine_similarity(word_to_vec_map[\"actor\"], gender) = ", cosine_similarity(word_to_vec_map["actor"], g))
print("cosine_similarity(word_to_vec_map[\"actress\"], gender) = ", cosine_similarity(word_to_vec_map["actress"], g))
print()
e1, e2 = equalize(("actor", "actress"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("\ncosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
print("\n")