import os
import numpy as np
import re
from scipy import stats

def cos_similarity(W_norm, ind1, ind2):
    return np.dot(W_norm[ind1], W_norm[ind2])

def cos_similarity_words(W_norm, vocab, w1, w2):
    return cos_similarity(W_norm, vocab[w1], vocab[w2])

def most_similar_k(W, words, vocab, ws, k):
    ind1 = vocab[ws[0]]
    ind2 = vocab[ws[1]]
    ind3 = vocab[ws[2]]
    pred_vec = W[ind2, :] - W[ind1, :] + W[ind3, :]
    dist = np.dot(W, pred_vec.T)
    dist[ind1] = -np.Inf
    dist[ind2] = -np.Inf
    dist[ind3] = -np.Inf
    best_k = (-dist).argsort(axis=0).flatten()[0:k]
    return [words[d] for d in best_k]

def most_similar(W, words, vocab, ws):
    return most_similar_k(W, words, vocab, ws, 1)[0]

def user_try_analogy(W, words, vocab):
    regExp = r'\s*(\w*)\s*\-?\s*(\w*)\s*\+?\s*(\w*).*'
    ws = re.match(regExp, input("analogy expression: ")).groups()
    pred = most_similar(W, words, vocab, ws)
    print("analogy expression: {} - {} + {} = {}".format(ws[0], ws[1], ws[2], pred))

def user_eval_similarity(W, vocab):
    ws = input("two words to compare for similarity seperated by space: ").split()
    print("cosine similarity = {}".format(cos_similarity_words(W, vocab, ws[0], ws[1])))


def test_single_similarity(W, vocab, dataset):
    scores = []
    calculated = []
    for data in dataset:
        word_1 = data[0]
        word_2 = data[1]
        score = data[2]
        scores.append(score)
        calculated.append(cos_similarity_words(W, vocab, word_1, word_2))
    return stats.spearmanr(scores, calculated).correlation

def test_similarity(W, vocab, datasets):
    dataset_results = {}
    for dataset_name, dataset in datasets.items():
        print("testing similarity on dataset '{}'".format(dataset_name))
        data = [x for x in dataset if (x[0] in vocab and x[1] in vocab)]
        dataset_results[dataset_name] = test_single_similarity(W, vocab, data)
    return dataset_results

def test_single_analogy(W, words, vocab, dataset):
    indices = np.array([[vocab[word] for word in data] for data in dataset])
    ind1, ind2, ind3, ind4 = indices.T

    predictions = np.zeros((len(indices),))
    split_size = 1000
    num_iter = int(np.ceil(len(indices) / float(split_size)))
    for j in range(num_iter):
        subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

        pred_vec = (W[ind2[subset], :] - W[ind1[subset], :] +  W[ind3[subset], :])
        dist = np.dot(W, pred_vec.T)
        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf

        predictions[subset] = np.argmax(dist, 0).flatten()

    val = (ind4 == predictions)
    return sum(val)

def test_analogy(W, words, vocab, datasets):
    correct_count = 0
    total_count = 0
    dataset_results = {}

    for dataset_name, dataset in datasets.items():
        print("testing analogy on dataset '{}'".format(dataset_name))
        data = [x for x in dataset if all(word in vocab for word in x)]
        count = test_single_analogy(W, words, vocab, data)
        correct_count += count
        total_count += len(data)
        dataset_results[dataset_name] = count / len(data)

    return (correct_count / total_count, dataset_results)

def load_datasets(base_dir, dataset_names):
    datasets = {}
    for name in dataset_names:
        with open(base_dir + name + '.txt') as f:
            datasets[name] = [line.rstrip().split() for line in f]
    return datasets

def load_similarity_datasets():
    base_dir = 'datasets/similarity/'
    dataset_names = ['SimVerb-3500', 'wordsim_relatedness', 'wordsim_similarity', 'MEN']
    datasets = load_datasets(base_dir, dataset_names)
    for name, dataset in datasets.items():
        datasets[name] = [[x[0], x[1], float(x[2])] for x in dataset]
    return datasets

def load_analogy_datasets():
    base_dir = 'datasets/analogy/'
    dataset_names = [
        'capital-common-countries', 'capital-world', 'currency',
        'city-in-state', 'family', 'gram1-adjective-to-adverb',
        'gram2-opposite', 'gram3-comparative', 'gram4-superlative',
        'gram5-present-participle', 'gram6-nationality-adjective',
        'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs',
    ]
    return load_datasets(base_dir, dataset_names)

def run_all_tests(W, words, vocab):
    similarity_datasets = load_similarity_datasets()
    analogy_datasets = load_analogy_datasets()

    similarity_results = test_similarity(W, vocab, similarity_datasets)
    analogy_results = test_analogy(W, words, vocab, analogy_datasets)

    print()
    print("Similarity results:")
    print("\t{:<30} {:<10}".format('Dataset','Spearman correlation'))
    print("\t-------------------------------------------------")
    for dataset_name, result in similarity_results.items():
        print("\t{:<30} {:<10}".format(dataset_name, '%.5f' % result))
    print()
    print("Analogy results:")
    print("\t{:<30} {:<10}".format('Dataset','guess accuracy'))
    print("\t-------------------------------------------------")
    analogy_total, analogy_per_dataset = analogy_results
    for dataset_name, result in analogy_per_dataset.items():
        print("\t{:<30} {:<10}".format(dataset_name, ('%.2f' % (result * 100))+'%'))
    print('total = '+('%.2f' % (analogy_total * 100))+'%')
    print()

def main():
    print('loading vector...')
    vectors_file = 'glove_text8_vectors.txt'

    # read words vector
    words = []
    vectors = {}
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0] != '<unk>':
                words.append(vals[0])
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((len(words), vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    while True:
        print("1. run analogy and similarity tests")
        print("2. interactive similarity testing")
        print("3. interactive analogy testing")
        print("4. exit")
        option = int(input("choose option (1-4): "))
        if option == 1:
            run_all_tests(W_norm, words, vocab)
        elif option == 2:
            user_eval_similarity(W_norm, vocab)
        elif option == 3:
            print("input analogy in the form like for example 'king - queen + men'")
            user_try_analogy(W_norm, words, vocab)
        else:
            break
        print()

if __name__ == "__main__":
    main()
