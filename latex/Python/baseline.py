import re
import string
import unicodedata
import stopwatch
from replacers import RegexpReplacer
from replacers import RegexpReplacerAge
from replacers import RepeatReplacer

# Import DBConn to connect to database
from thesis_db import DBConn

# Create a connection to the thesis database
my_conn = DBConn()
conn = my_conn.create_conn()
cur = conn.cursor()

# First of all handle the bullying cases
cur.execute("select distinct question from sample_data_02 \
             where  class = 'not_bullying'")

for row in cur:

    # Convert question to lower case
    question = row['question']
    question_lower = question.lower()

    # Convert question to ascii
    question_ascii = unicodedata.normalize(
        'NFKD',
        question_lower).encode(
        'ascii',
        'ignore')

    # Remove URL
    question_url = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
        '',
        question_ascii)

    # Remove any remaining punctuation
    question_punct = question_url.translate(
        string.maketrans(
            "",
            ""),
        string.punctuation)

    # Replace ages 11 - 25 with text values
    age_replacer = RegexpReplacerAge()
    question_age = age_replacer.replace(question_punct)

    # Remove any remaining digits
    question_no_digit = question_age.translate(
        None, string.digits)

    # Remove repeating characters
    rep_replacer = RepeatReplacer()
    question_rep_list = []

    for word in question_no_digit.split():
        question_rep_list.append(
            rep_replacer.replace(word))

    question_no_repeat = ' '.join(question_rep_list)

    # Fix abbreviations to display whole words
    replacer = RegexpReplacer()
    question_replace = replacer.replace(question_no_repeat)

    # Write to file
    filename = 'R:/Data/Corpora/01/not_bullying' + str(
        count_total).zfill(5) + '.txt'
    file = open(filename, 'w')
    file.write(question_replace)
    file.close()

conn.close()
