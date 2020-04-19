import sys
import stopwatch

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

files = []
if len(sys.argv) > 1:
    for file in sys.argv[1:]:
        files.append(str(file))

words_min_size = 2 
 
text = ""

for file in files:
    f = open(file,"rU")
    time = stopwatch.Timer()
    for line in f:
        bigram_list = find_ngrams([w for w in line.lower().split() if len(w) >= 2], 3)
        bigram_line = ""
        for bigram in bigram_list:
            bigram_line += "_".join(bigram)
            bigram_line += " "
        # print bigram_line
    time.stop()

print 'Total Time:\t',     time.elapsed
