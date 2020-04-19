import re
import string
import unicodedata
import stopwatch
# Import DBConn to connect to database
from thesis_db import DBConn

# Create a connection to the thesis database
my_conn = DBConn()
conn = my_conn.create_conn()
cur = conn.cursor()

count_total = 0

# First of all handle the bullying cases
#cur.execute("select question from sample_questions where class = 'bullying'")
cur.execute("select distinct question from sample_questions where class = 'not_bullying'")

for row in cur:

    count_total += 1

    question = row['question']
    question_lower = question.lower()

    # Convert question to ascii
    question_ascii = unicodedata.normalize(
        'NFKD',
        question_lower).encode(
        'ascii',
        'ignore')

    # Write to file
    filename = 'C:\\Tools\\nltk_data\\corpora\\cyberbullying_00\\not_bullying\\' + \
        str(count_total).zfill(4) + '.txt'
    file = open(filename, 'w')
    file.write(question_ascii)
    file.close()

