import metapy
import numpy as np
import math
import argparse


def normalize_row(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    # print("input:", input_matrix)
    # print("row sum:", row_sums)

    row_sums = np.nan_to_num(input_matrix).sum(axis=1, keepdims=True)
    # print("row sum:", row_sums)

    #new_matrix = input_matrix / row_sums if np.isscalar(row_sums) else input_matrix / row_sums[:, np.newaxis]
    new_matrix = np.divide(input_matrix, row_sums, out=np.zeros_like(input_matrix), where=row_sums != 0)
    return np.nan_to_num(new_matrix)

def normalize_col(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    col_sums = np.nan_to_num(input_matrix).sum(axis=0, keepdims=True)

    #new_matrix =  input_matrix / col_sums if np.isscalar(col_sums) else input_matrix / col_sums[np.newaxis, :]
    new_matrix = np.divide(input_matrix, col_sums, out=np.zeros_like(input_matrix), where=col_sums != 0)
    return np.nan_to_num(new_matrix)


class Corpus(object):
    """
    A collection of documents.
    """

    def __init__(self, documents_path, lambda_b, lambda_c):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None
        self.document_topic_prob = []  # P(z | d), pi
        self.topic_word_prob = None  # P(w | z)
        self.topic_word_prob_background = None  # P(w | z)
        self.topic_word_prob_collection_specific = []
        self.topic_prob_j = None  # P(z | d, w)
        self.topic_prob_B = None  # P(z | d, w)
        self.topic_prob_C = None  # P(z | d, w)
        self.lambda_B = lambda_b
        self.lambda_C = lambda_c

        self.number_of_collections = 0
        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]

        Update self.number_of_documents
        """
        # #############################

        doc = metapy.index.Document()
        tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
        tok = metapy.analyzers.LowercaseFilter(tok)
        tok = metapy.analyzers.LengthFilter(tok, min=3, max=1000)
        tok = metapy.analyzers.ListFilter(tok, "lemur-stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
        tok = metapy.analyzers.Porter2Filter(tok)
        collection = -1

        with open(self.documents_path) as file:
            for num, line in enumerate(file):
                l = line.strip()
                c = int(l[0])
                l = l[2:]
                doc.content(l)
                tok.set_content(doc.content())
                if c != collection:
                    self.documents.append([])
                    collection = c
                self.documents[c].append([token for token in tok])
        self.number_of_collections = len(self.documents)
        self.number_of_documents = len(self.documents[0])
        #print(self.number_of_collections)
        #print(self.number_of_documents)
        #print(self.documents[0])

    def build_vocabulary(self):

        voc = set([])
        for documents in self.documents:
            voc = voc.union(set(i for j in documents for i in j))
        self.vocabulary = list(voc)
        self.vocabulary_size = len(self.vocabulary)
        #print(self.vocabulary_size)
        #print(self.vocabulary)

    def build_term_doc_matrix(self):

        self.term_doc_matrix = np.zeros((self.number_of_collections, self.number_of_documents, self.vocabulary_size, 1))
        self.topic_word_prob_background = np.zeros(self.vocabulary_size)

        for k in range(self.number_of_collections):
            for i in range(self.number_of_documents):
                for j in range(self.vocabulary_size):
                    self.term_doc_matrix[k][i][j] = self.documents[k][i].count(self.vocabulary[j])
                    self.topic_word_prob_background[j] += self.term_doc_matrix[k][i][j]
                #print(self.term_doc_matrix[k][i])
        # print(self.term_doc_matrix[0][0])
        #self.topic_word_prob_background = np.array(self.topic_word_prob_background)
        self.topic_word_prob_background = normalize_col(np.transpose(self.topic_word_prob_background))
        # print(self.topic_word_prob_background)



    def initialize_randomly(self, number_of_topics):

        for i in range(self.number_of_collections):
            prob = np.random.random_sample((self.number_of_documents, number_of_topics))
            prob = normalize_row(prob)
            self.document_topic_prob.append(prob)

        self.topic_word_prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize_row(self.topic_word_prob)

        for i in range(self.number_of_collections):
            prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))
            prob = normalize_row(prob)
            self.topic_word_prob_collection_specific.append(prob)
        # print(len(self.topic_word_prob_collection_specific))


    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        self.initialize_randomly(number_of_topics)

        #print("pi", self.document_topic_prob)
        #print("p(w|theta)", self.topic_word_prob)

    def expectation_step(self, number_of_topics, verbose):
        """ The E-step updates P(z | w, d)
        """
        if verbose:
            print("E step:")

        # ############################

        for k in range(self.number_of_collections):
            for i in range(self.number_of_documents):
                tp_j = np.multiply(np.transpose([self.document_topic_prob[k][i]]), np.multiply(self.lambda_C, self.topic_word_prob) + np.multiply(1 - self.lambda_C, self.topic_word_prob_collection_specific[k]))
                a = np.asmatrix(np.multiply(self.lambda_B, self.topic_word_prob_background))
                b = np.multiply(self.lambda_B, self.topic_word_prob_background) + np.multiply(1 - self.lambda_B, np.nan_to_num(tp_j).sum(axis=0, keepdims=True))
                tp_b = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                tp_j = normalize_col(tp_j)
                self.topic_prob_j[k][i] = np.transpose(tp_j)
                self.topic_prob_B[k][i] = np.transpose(tp_b)
                a = np.multiply(self.lambda_C, self.topic_word_prob)
                b = np.multiply(self.lambda_C, self.topic_word_prob) + np.multiply(1 - self.lambda_C, self.topic_word_prob_collection_specific[k])
                tp_c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                self.topic_prob_C[k][i] = np.transpose(tp_c)
        #print("p(z):")
        #print(self.topic_prob_j[k][i])

    def maximization_step(self, number_of_topics, verbose):
        """ The M-step updates P(w | z)
        """
        if verbose:
            print("M step:")

        self.topic_word_prob = np.zeros((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob_collection_specific = []

        for k in range(self.number_of_collections):
            topic_word_prob_collection_specific = np.zeros((number_of_topics, len(self.vocabulary)))
            for i in range(self.number_of_documents):
                # update P(w | z)

                # ############################

                self.topic_word_prob = np.add(self.topic_word_prob,
                                              np.transpose(np.multiply(np.multiply(np.multiply(self.term_doc_matrix[k][i], 1 - self.topic_prob_B[k][i]), self.topic_prob_j[k][i]), self.topic_prob_C[k][i])))

                topic_word_prob_collection_specific = np.add(self.topic_word_prob,
                                              np.transpose(np.multiply(np.multiply(np.multiply(self.term_doc_matrix[k][i], 1 - self.topic_prob_B[k][i]), self.topic_prob_j[k][i]), 1 - self.topic_prob_C[k][i])))

                # update P(z | d)

                # ############################

                matrix = np.dot(np.transpose(self.term_doc_matrix[k][i]), self.topic_prob_j[k][i])
                self.document_topic_prob[k][i] = normalize_row(matrix)
            topic_word_prob_collection_specific = normalize_row(topic_word_prob_collection_specific)
            self.topic_word_prob_collection_specific.append(topic_word_prob_collection_specific)

        self.topic_word_prob = normalize_row(self.topic_word_prob)

        #print("pi:")
        #print(self.document_topic_prob)
        #print("p(w|theta):")
        #print(self.topic_word_prob)

    def calculate_likelihood(self):

        likelihood = 0
        for k in range(self.number_of_collections):
            likelihood += np.multiply(self.term_doc_matrix[k, :, :, 0], np.log(np.multiply(self.lambda_B, self.topic_word_prob_background) + np.multiply(1 - self.lambda_B, np.dot(self.document_topic_prob[k], np.multiply(self.lambda_C, self.topic_word_prob) + np.multiply(1 - self.lambda_C, self.topic_word_prob_collection_specific[k]))))).sum()
        self.likelihoods.append(likelihood)
        return self.likelihoods[-1]

    def ccmix(self, number_of_topics, max_iter, epsilon, verbose=True):

        """
        Model topics.
        """
        if verbose:
            print("EM iteration begins...")

        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(z | d, w)
        self.topic_prob_j = np.zeros([self.number_of_collections, self.number_of_documents, self.vocabulary_size, number_of_topics], dtype=np.float)
        self.topic_prob_B = np.zeros([self.number_of_collections, self.number_of_documents, self.vocabulary_size, 1], dtype=np.float)
        self.topic_prob_C = np.zeros([self.number_of_collections, self.number_of_documents, self.vocabulary_size, number_of_topics], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            if verbose:
                print("Iteration #" + str(iteration + 1) + "...")

            # ############################

            self.expectation_step(number_of_topics, verbose)
            self.maximization_step(number_of_topics, verbose)
            next_likelihood = self.calculate_likelihood()
            if verbose:
                print("Likelihood:{}".format(next_likelihood))
            if abs(next_likelihood - current_likelihood) < epsilon:
                print(abs(next_likelihood - current_likelihood))
                break
            current_likelihood = next_likelihood
            #input("Press Enter to continue...")

        #print(self.document_topic_prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./data/laptop_reviews.txt')
    parser.add_argument("--output_common_path", type=str, default='./result/common_laptop.txt')
    parser.add_argument("--output_specific_path", type=str, default='./result/specific_laptop.txt')
    parser.add_argument("--lambda_b", type=float, default=0.95)
    parser.add_argument("--lambda_c", type=float, default=0.25)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--number_topics", type=int, default=5)
    parser.add_argument("--number_top_words", type=int, default=8)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()
    documents_path = args.input_path

    corpus = Corpus(documents_path, args.lambda_b, args.lambda_c)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    # print(corpus.vocabulary)
    if args.verbose:
        print("Vocabulary size:" + str(len(corpus.vocabulary)))
    number_of_topics = args.number_topics
    max_iterations = args.max_iterations
    epsilon = 0.00001
    corpus.ccmix(number_of_topics, max_iterations, epsilon, args.verbose)
    # print("topic word prob")
    #print(corpus.topic_word_prob[0])

    with open(args.output_common_path, 'w') as file:
        for cluster in corpus.topic_word_prob:
            ind = np.argsort(-cluster)[:args.number_top_words]
            for i in ind:
                file.write(str(corpus.vocabulary[i]) + ' ' + str(cluster[i]) + '\n')
            file.write('\n')
        file.close()
    with open(args.output_specific_path, 'w') as file:
        for collection in range(corpus.number_of_collections):
            file.write('collection ' + str(collection) + '\n')
            for cluster in corpus.topic_word_prob_collection_specific[collection]:
                ind = np.argsort(-cluster)[:args.number_top_words]
                for i in ind:
                    file.write(str(corpus.vocabulary[i]) + ' ' + str(cluster[i]) + '\n')
                file.write('\n')
        file.close()




if __name__ == '__main__':
    main()
