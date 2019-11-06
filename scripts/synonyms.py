from nltk.corpus import wordnet
from nltk.corpus import stopwords
class Syn():
    def __init__(self):
        #TODO
        self.synonyms_list = []
        self.stop_words_list = list(set(stopwords.words('english')))
        return
    
    def GetSyn(self,word):
        #TODO
        wordnet_list = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                wordnet_list.append(l.name())
        wordnet_list = list(set(wordnet_list))
        for w in wordnet_list:
            if w not in self.stop_words_list:
                self.synonyms_list.append(w)
        return self.synonyms_list

if __name__ == "__main__":
    syn = Syn()
    print(syn.GetSyn('capital'))
