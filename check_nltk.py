import nltk

def check(resource):
    try:
        nltk.data.find(resource)
        print(f"FOUND: {resource}")
    except LookupError:
        print(f"MISSING: {resource}")

check('taggers/averaged_perceptron_tagger')
check('corpora/wordnet')
check('tokenizers/punkt')
