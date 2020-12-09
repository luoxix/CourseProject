import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """
    if len(input_matrix.shape) == 1:
        total_sum = input_matrix.sum()
        new_matrix = input_matrix / total_sum
        return new_matrix

    row_sums = input_matrix.sum(axis=1)
    assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path, lambda_b=0.95):
        """
        Initialize empty document list.
        """
        self.lambda_b = lambda_b
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.background_word_prob = None # p(w | B)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################

        f = open(self.documents_path, "r").readlines()
        for line in f:
            items = line.strip().split()
            words = []
            for e in items:
                if e in ["0", "1"]:
                    continue
                words.append(e)
            self.documents.append(words)
        self.n = len(self.documents)
        self.number_of_documents = self.n
        
    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        for d in self.documents:
            for w in d:
                self.vocabulary.append(w)
        self.vocabulary = list(set(self.vocabulary))
        self.m = len(self.vocabulary)
        self.vocabulary_size = self.m

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        self.term_doc_matrix = np.zeros([self.n, self.m])
        for i, d in enumerate(self.documents):
            for w in d:
                pos = self.vocabulary.index(w)
                self.term_doc_matrix[i, pos] += 1
                
    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize!
        HINT: you will find numpy’s random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        self.k = number_of_topics
        self.document_topic_prob = np.random.random_sample((self.n, self.k))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random_sample((self.n, self.m))
        self.topic_word_prob = normalize(self.topic_word_prob)

        self.background_word_prob = np.random.random_sample((self.m))
        self.background_word_prob = normalize(self.background_word_prob)

        
    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.k = number_of_topics
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

        self.background_word_prob = np.ones((self.m))
        self.background_word_prob = normalize(self.background_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        for i in range(self.n):
            for j in range(self.m):
                fm = 0.0
                for l in range(self.k):
                    self.topic_prob[i, j, l] = self.topic_word_prob[l, j] * self.document_topic_prob[i, l]
                    fm += self.topic_prob[i, j, l]
                if fm == 0:
                    for l in range(self.k):
                        self.topic_prob[i, j, l] = 0
                    self.background_prob[i, j] = 1
                else:
                    for l in range(self.k):
                        self.topic_prob[i, j, l] /= fm
                    back_prob = self.background_word_prob[j]
                    back_fm = self.lambda_b * back_prob + (1 - self.lambda_b) * fm
                    self.background_prob[i, j] = self.lambda_b * back_prob / back_fm
   

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################
        for l in range(self.k):
            for j in range(self.m):
                self.topic_word_prob[l, j] = 0
                for i in range(self.n):
                    self.topic_word_prob[l, j] += self.term_doc_matrix[i, j] * self.topic_prob[i, j, l] * (1 - self.background_prob[i, j])
             
        self.topic_word_prob = normalize(self.topic_word_prob)  
        
        # update P(z | d)

        # ############################
        # your code here
        # ############################
        for i in range(self.n):
            for l in range(self.k):
                self.document_topic_prob[i, l] = 0.0
                for j in range(self.m):
                    self.document_topic_prob[i, l] += self.term_doc_matrix[i, j] * self.topic_prob[i, j, l]
        
        self.document_topic_prob = normalize(self.document_topic_prob)
                    
    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        loglikelihood = 0
        for i in range(self.n):
            for j in range(self.m):
                topic_prob = 0.0
                for l in range(self.k):
                    topic_prob += self.topic_word_prob[l, j] * self.document_topic_prob[i, l]
                prob = self.background_word_prob[j] * self.lambda_b + (1 - self.lambda_b) * topic_prob 
                if prob > 0:
                    loglikelihood += self.term_doc_matrix[i, j] * math.log(prob)
        self.likelihoods.append(loglikelihood)
        

    def smm(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.n, self.m, number_of_topics], dtype=np.float)

        self.background_prob = np.zeros([self.n, self.m], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            print("Iter:{}\tLikelihood:{:.2f}".format(iteration, self.likelihoods[iteration]))
            



def main():
    documents_path = './data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))


    
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.smm(number_of_topics, max_iterations, epsilon)
    '''
    topics = np.argmax(corpus.document_topic_prob, axis=1)
    for i in range(corpus.n):
        if i >= 20:
            break
        print(topics[i])
    '''


if __name__ == '__main__':
    main()
