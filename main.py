import os
import numpy as np
import re
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import fasttext
from gensim.models.word2vec import Word2Vec

def user_try_analogy(model: KeyedVectors):
    regExp = r'\s*(\w*)\s*\-?\s*(\w*)\s*\+?\s*(\w*).*'
    ws = re.match(regExp, input("analogy expression: ")).groups()

    result = model.most_similar(positive=[ws[0], ws[2]], negative=[ws[1]])
    preds = []
    for i in range(3):
        pred, _ = result[i]
        preds.append(pred)

    print("analogy expression: {} - {} + {} = [{}, {}, {}]".format(ws[0], ws[1], ws[2], preds[0], preds[1], preds[2]))

def user_eval_similarity(model: KeyedVectors):
    ws = input("two words to compare for similarity seperated by space: ").split()
    similarity = model.similarity(ws[0], ws[1])
    print("cosine similarity = {}".format(similarity))

def test_similarity(model: KeyedVectors):
    dataset_names = ['wordsim_relatedness', 'wordsim_similarity', 'MEN', 'SimVerb-3500']

    dataset_results = {}
    for name in dataset_names:
        print("testing similarity on dataset '{}'".format(name))
        pearson, spearman, oov = model.evaluate_word_pairs('datasets/similarity/{}.txt'.format(name))
        dataset_results[name] = spearman[0]

    return dataset_results

def test_analogy(model: KeyedVectors, dir_name, dataset_names):
    correct_count = 0
    total_count = 0
    dataset_results = {}
    for name in dataset_names:
        print("testing analogy on dataset '{}'".format(name))
        score, sections = model.evaluate_word_analogies('datasets/analogy/{}/{}.txt'.format(dir_name, name))
        correct_count += len(sections[0]['correct'])
        total_count += len(sections[0]['correct']) + len(sections[0]['incorrect'])
        dataset_results[name] = score

    return (correct_count / total_count, dataset_results)

def test_analogy_google(model: KeyedVectors):
    dataset_names = [
        'capital-common-countries', 'capital-world', 'currency',
        'city-in-state', 'family', 'gram1-adjective-to-adverb',
        'gram2-opposite', 'gram3-comparative', 'gram4-superlative',
        'gram5-present-participle', 'gram6-nationality-adjective',
        'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs',
    ]
    return test_analogy(model, 'google', dataset_names)

def test_analogy_msr(model: KeyedVectors):
    return test_analogy(model, 'MSR', ['MSR'])

def run_all_tests(model: KeyedVectors):
    similarity_results = test_similarity(model)
    google_analogy_results = test_analogy_google(model)
    msr_analogy_results = test_analogy_msr(model)

    print()
    print("Similarity results:")
    print("\t{:<40} {:<10}".format('Dataset','Spearman correlation'))
    print("\t-----------------------------------------------------------")
    for dataset_name, result in similarity_results.items():
        print("\t{:<40} {:<10}".format(dataset_name, '%.5f' % result))
    print()
    print("Analogy results:")
    print("\t{:<40} {:<10}".format('Dataset','guess accuracy'))
    print("\t-----------------------------------------------------------")
    analogy_total, analogy_per_dataset = google_analogy_results
    for dataset_name, result in analogy_per_dataset.items():
        print("\t{:<40} {:<10}".format('google-'+dataset_name, ('%.2f' % (result * 100))+'%'))
    print("\t{:<40} {:<10}".format('google total', ('%.2f' % (analogy_total * 100))+'%'))
    analogy_total, _ = msr_analogy_results
    print("\t{:<40} {:<10}".format('MSR', ('%.2f' % (analogy_total * 100))+'%'))
    print()

def get_new_model():
    model_names = ['word2vec-skipgram', 'word2vec-cbow', 'glove', 'fasttext-skipgram', 'fasttext-cbow', 'en-glove', 'en-fasttext-skipgram', 'en-fasttext-cbow']
    model = None
    while model == None:
        chosen_model = input('choose one of ' + str(model_names) + ': ')
        if chosen_model in model_names:
            print("loading model...")
        if chosen_model == 'word2vec-skipgram':
            model = Word2Vec.load('models/word2vec-skipgram/word2vec.model', mmap='r').wv
        elif chosen_model == 'word2vec-cbow':
            model = Word2Vec.load('models/word2vec-cbow/word2vec.model', mmap='r').wv
        elif chosen_model == 'glove':
            vectors_file = 'models/simplewiki_glove.txt'
            tmp_file = "models/gensim_glove_vectors.txt"
            glove2word2vec(glove_input_file=vectors_file, word2vec_output_file=tmp_file)
            model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
        elif chosen_model == 'en-glove':
            vectors_file = 'models/enwiki_glove.txt'
            tmp_file = "models/gensim_glove_vectors.txt"
            glove2word2vec(glove_input_file=vectors_file, word2vec_output_file=tmp_file)
            model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
        elif chosen_model == 'fasttext-skipgram':
            model = fasttext.load_facebook_model('models/simplewiki_fasttext_skipgram.bin').wv
        elif chosen_model == 'fasttext-cbow':
            model = fasttext.load_facebook_model('models/simplewiki_fasttext_cbow.bin').wv
        elif chosen_model == 'en-fasttext-skipgram':
            model = fasttext.load_facebook_model('models/enwiki_fasttext_skipgram.bin').wv
        elif chosen_model == 'en-fasttext-cbow':
            model = fasttext.load_facebook_model('models/enwiki_fasttext_cbow.bin').wv
        else:
            print('unrecognized model name. choose again')
    return model

def main():
    model = get_new_model()
    while True:
        print("0. load model")
        print("1. run analogy and similarity tests")
        print("2. interactive similarity testing")
        print("3. interactive analogy testing")
        print("4. exit")
        option = input("choose option (0-4): ")
        if option == '0':
            model = get_new_model()
        elif option == '1':
            run_all_tests(model)
        elif option == '2':
            user_eval_similarity(model)
        elif option == '3':
            print("input analogy in the form like for example 'king - man + woman'")
            user_try_analogy(model)
        else:
            break
        print()

if __name__ == "__main__":
    main()
