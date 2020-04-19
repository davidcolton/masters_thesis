import re
import csv
import yaml
import enchant
from nltk.corpus import wordnet
from nltk.metrics import edit_distance

##################################################
## Replacing Words Matching Regular Expressions ##
##################################################

replacement_patterns = [
    (r'(\su\s)', ' you '),
    (r'(\su$)', ' you'),
    (r'(^u\s)', 'you '),
    (r'(\sur\s)', ' your '),
    (r'(\sur$)', ' your'),
    (r'(^ur\s)', 'your '),
    (r'(\sr\s)', ' are '),
    (r'(\sr$)', ' are'),
    (r'(^r\s)', 'are '),
    (r'(\sn\s)', ' and '),
    (r'(\sn$)', ' and'),
    (r'(^n\s)', 'and '),
    (r'(\sk\s)', ' ok '),
    (r'(\sk$)', ' ok'),
    (r'(^k\s)', 'ok '),
    (r'(\sy\s)', ' why '),
    (r'(\sy$)', ' why'),
    (r'(^y\s)', 'why '),
    (r'(\sw\s)', ' with '),
    (r'(\sw$)', ' with'),
    (r'(^w\s)', 'with '),
]

digit_replacement_patterns = [
    (r'(\s11\s)', ' eleven '),
    (r'(\s11$)', ' eleven'),
    (r'(^11\s)', 'eleven '),
    (r'(\s12\s)', ' twelve '),
    (r'(\s12$)', ' twelve'),
    (r'(^12\s)', 'twelve '),
    (r'(\s13\s)', ' thirteen '),
    (r'(\s13$)', ' thirteen'),
    (r'(^13\s)', 'thirteen '),
    (r'(\s14\s)', ' fourteen '),
    (r'(\s14$)', ' fourteen'),
    (r'(^14\s)', 'fourteen '),
    (r'(\s15\s)', ' fifteen '),
    (r'(\s15$)', ' fifteen'),
    (r'(^15\s)', 'fifteen '),
    (r'(\s16\s)', ' sixteen '),
    (r'(\s16$)', ' sixteen'),
    (r'(^16\s)', 'sixteen '),
    (r'(\s17\s)', ' seventeen '),
    (r'(\s17$)', ' seventeen'),
    (r'(^17\s)', 'seventeen '),
    (r'(\s18\s)', ' eighteen '),
    (r'(\s18$)', ' eighteen'),
    (r'(^18\s)', 'eighteen '),
    (r'(\s19\s)', ' nineteen '),
    (r'(\s19$)', ' nineteen'),
    (r'(^19\s)', 'nineteen '),
    (r'(\s20\s)', ' twenty '),
    (r'(\s20$)', ' twenty'),
    (r'(^20\s)', 'twenty '),
    (r'(\s21\s)', ' twenty one '),
    (r'(\s21$)', ' twenty one'),
    (r'(^21\s)', 'twenty one '),
    (r'(\s22\s)', ' twenty two '),
    (r'(\s22$)', ' twenty two'),
    (r'(^22\s)', 'twenty two '),
    (r'(\s23\s)', ' twenty three '),
    (r'(\s23$)', ' twenty three'),
    (r'(^23\s)', 'twenty three '),
    (r'(\s24\s)', ' twenty four '),
    (r'(\s24$)', ' twenty four'),
    (r'(^24\s)', 'twenty four '),
    (r'(\s25\s)', ' twenty five '),
    (r'(\s25$)', ' twenty five'),
    (r'(^25\s)', 'twenty five '),
]


class RegexpReplacer(object):

    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl)
                         for (regex, repl) in patterns]

    def replace(self, text):
        s = text

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)

        return s


class RegexpReplacerAge(object):

    def __init__(self, patterns=digit_replacement_patterns):
        self.patterns = [(re.compile(regex), repl)
                         for (regex, repl) in patterns]

    def replace(self, text):
        s = text

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)

        return s


class RepeatReplacer(object):

    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word

        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word
