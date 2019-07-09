import nltk as nk
#nk.download_gui()
# for tokenizing words and sentence
from nltk.tokenize import sent_tokenize, word_tokenize
# for removing stop words 
from nltk.corpus import stopwords
# steam tagging 
from nltk.stem import PorterStemmer
# tagging 
from nltk.tokenize import PunktSentenceTokenizer
from asyncio.log import logger
# for gender prediction
from nltk.corpus import names
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
data = "All work and no play makes jack a dull boy, all work and no play"
data2 = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
# string intot array
word = word_tokenize(data)
sentence = sent_tokenize(data2, language='english')
# printing array
#print(word)
#print(sentence)
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
print("------Filtered words-----")
print(wordsFiltered)

print("NLTP streaming")

words = ["game","gaming","gamed","games"]
ps = PorterStemmer()
for word in words:
    print(word)
sentence = "gaming, the gamers play games"
words2 = word_tokenize(sentence)
for w in words2:
     print(ps.stem(w))
print("Tagging the words")
# tagging 
document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
sentences = nk.sent_tokenize(document)   
for sent in sentences:
    print(nk.pos_tag(nk.word_tokenize(sent)))

logger.info("Filtering the words based on type of words i.e, Noun, pronoun, verbs etc")

document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'
sentences = nk.sent_tokenize(document)

data = []
for sent in sentences:
    data = data + nk.pos_tag(nk.word_tokenize(sent))

# list the words with second argument contains NNP
for word in data: 
    if 'NNP' in word[1]: 
        print(word)

#"""nlp prediction"""

print("gender predictions")
 
def gender_features(word): 
    return {'last_letter': word[-1]} 
# Load data and training 
allnames = ([(name, 'male') for name in names.words('male.txt')] + 
     [(name, 'female') for name in names.words('female.txt')])


featuresets = [(gender_features(n), g) for (n,g) in allnames] 
train_set = featuresets
classifier = nk.NaiveBayesClassifier.train(train_set) 
 
# Predict
#name = input("Name: ")
print(classifier.classify(gender_features("test")))

# Sentiment analysis
def word_feats(words):
    return dict([(word, True) for word in words])

positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = negative_features + positive_features + neutral_features
classifier = NaiveBayesClassifier.train(train_set) 

#prediction
neg = 0
pos = 0
sentence = "Awesome movie, I liked it"
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
   for word in words:
    classResult = classifier.classify( word_feats(word))
    print("words:%s, classResult %s", words,classResult)
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))

