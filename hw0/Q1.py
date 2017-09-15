import sys
from collections import OrderedDict
file = open(sys.argv[1], "r")
words = file.read()
file.close()
words_split = words.split()
word_count = OrderedDict()
for word in words_split:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1
out = open('Q1.txt', 'w')
for index, word in enumerate(word_count):
#    print(f'{word} {index} {word_count[word]}', file=out)
	print('%s %d %d'%(word, index, word_count[word]) ,file=out)
out.close()
