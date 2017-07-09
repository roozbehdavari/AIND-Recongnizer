import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # List of average logL for each choice of n_component
        BIC_n_components = []

        # iterate over all the range of n_components for finding the best model
        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                model = GaussianHMM(n_components=n, n_iter=1000).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                BIC = -2*logL + n * np.log(len(self.lengths))
                BIC_n_components.append([BIC, n])

            except:
                continue

        try:
            best_n = min(BIC_n_components)[1]
            model = GaussianHMM(n_components=best_n, n_iter=1000).fit(self.X, self.lengths)
            return model
        except:
            if self.verbose:
                print("failure on {}".format(self.this_word))
            return None
        

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Getting the sequence of each occurrence of the word
        word_sequences = self.sequences

        # Setting the number of splits to 2 to 4
        split_method = KFold(n_splits= min(len(self.lengths), 4))

        # List of average logL for each choice of n_component
        logL_n_components = []


        # iterate over all the range of n_components for finding the best model
        for i in range(self.min_n_components, self.max_n_components+1):

            # List of average logL for CV
            logL_cv = []

            # Splitting word sequences into train and test for cv
            for cv_train_idx, cv_test_idx in split_method.split(word_sequences):

                # Converting the sequences into the format needed for hmmlearn
                # Alternatively, I could do the split on X and lengths directly.
                X_train, lengths_train, X_test, lengths_test = [], [], [], []
                for idx in cv_train_idx:
                    X_train += word_sequences[idx]
                    lengths_train.append(len(word_sequences[idx]))
                for idx in cv_test_idx:
                    X_test += word_sequences[idx]
                    lengths_test.append(len(word_sequences[idx]))

                try:
                    model = GaussianHMM(n_components=i, n_iter=1000).fit(X_train, lengths_train)
                    logL = model.score(X_test, lengths_test)
                    logL_cv.append(logL)

                except:
                    continue

            if logL_cv:
                logL_n_components.append([np.mean(logL_cv),i])


        try:
            best_n = max(logL_n_components)[1]
            model = GaussianHMM(n_components=best_n, n_iter=1000).fit(self.X, self.lengths)
            return model
        except:
            if self.verbose:
                print("failure on {}".format(self.this_word))
            return None


