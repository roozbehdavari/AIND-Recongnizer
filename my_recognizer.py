import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # List of all words in the test set
    test_set_dict = test_set.get_all_Xlengths()

    for index in test_set_dict:

        # dict of {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... } for probabilities list
        max_logL = -10 ** (10)
        word_dict = {}
        best_guess = ''

        # Getting X, and lengths for scoring
        X, lengths = test_set_dict[index]

        for word, model in models.items():

            try:
                LogLvalue = model.score(X, lengths)

                # Adding it to the dictionary
                word_dict[word] = LogLvalue

                # Logic for finding the best guess -- finding the max logL
                if max_logL < LogLvalue:
                    max_logL = LogLvalue
                    best_guess = word

            except:
                word_dict[word] = None

        # Adding the best guess to the guesses list
        guesses.append(best_guess)
        probabilities.append(word_dict)

    return probabilities, guesses
