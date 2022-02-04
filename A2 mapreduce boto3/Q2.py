from mrjob.job import MRJob
from mrjob.step import MRStep
from nltk.corpus import stopwords
import re

WORD_RE = re.compile(r"[\w']+")
stop = set(stopwords.words('english'))

class MRMostUsedWord(MRJob):

    def mapper_get_words(self, _, txt):
        for word in WORD_RE.findall(txt):
            w = word.lower()
            if w not in stop:
                yield (word.lower(), 1)

    def combiner_count_words(self, word, counts):
        yield (word, sum(counts))

    def reducer_count_words(self, word, counts):
        yield None, (sum(counts), word)

    # discard the key; it is just None
    def reducer_find_max_word(self, _, word_count_pairs):
        lst=list(word_count_pairs)
        lst=sorted(lst, key=lambda x: x[0],reverse=True)
        
        lst = lst[:10]
        for pair in lst:
            yield pair

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(reducer=self.reducer_find_max_word)
        ]

if __name__ == '__main__':
    MRMostUsedWord.run()
