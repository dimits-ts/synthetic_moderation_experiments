"""
Code from
https://github.com/kj2013/deliberative-politics/blob/main/notebooks/01%20-%20Deliberation%20-%20Feature%20extraction.ipynb
happierfuntokenizer_v3

This code implements a basic, Twitter-aware tokenizer.

A tokenizer is a function that splits a string of text into words. In
Python terms, we map string and unicode objects into lists of unicode
objects.

There is not a single right way to do tokenizing. The best method
depends on the application.  This tokenizer is designed to be flexible
and this easy to adapt to new domains and tasks.  The basic logic is
this:

1. The tuple regex_strings defines a list of regular expression
   strings.

2. The regex_strings strings are put, in order, into a compiled
   regular expression object called word_re.

3. The tokenization is done by word_re.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   Tokenizer.

4. When instantiating Tokenizer objects, there is a single option:
   preserve_case.  By default, it is set to True. If it is set to
   False, then the tokenizer will downcase everything except for
   emoticons.

The __main__ method illustrates by tokenizing a few examples.

I've also included a Tokenizer method tokenize_random_tweet(). If the
twitter library is installed (http://code.google.com/p/python-twitter/)
and Twitter is cooperating, then it should tokenize a random
English-language tweet.
"""

import re
import html.entities

######################################################################


######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most imporatantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# This particular element is used in a couple ways, so we define it
# with a name:
EMOTICON_STRING = r"""
    (?:
      [<>]?
      [:;=8>]                    # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpPxX/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpPxX/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8<]                    # eyes
      [<>]?
      |
      <[/\\]?3                         # heart(added: has)
      |
      \(?\(?\#?                   #left cheeck
      [>\-\^\*\+o\~]              #left eye
      [\_\.\|oO\,]                #nose
      [<\-\^\*\+o\~]              #right eye
      [\#\;]?\)?\)?               #right cheek
    )"""

# The components of the tokenizer:
REGEX_STRINGS = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?
      \d{3}          # exchange
      [\-\s.]*
      \d{4}          # base
    )""",
    # Emoticons:
    EMOTICON_STRING,
    # http:
    # Web Address:
    r"""(?:(?:http[s]?\:\/\/)?(?:[\w\_\-]+\.)+(?:com|net|gov|edu|info|org|ly
    |be|gl|co|gs|pr|me|cc|us|gd|nl|ws|am|im|fm|kr|to|jp|sg)(?:\/[\s\b$])?)""",
    r"""(?:http[s]?\:\/\/)""",  # need to capture it alone sometimes
    # command in parens:
    r"""(?:\[[\w_]+\])""",  # need to capture it alone sometimes
    # HTTP GET Info
    r"""(?:\/\w+\?(?:\;?\w+\=\w+)+)""",
    # HTML tags:
    r"""(?:<[^>]+\w=[^>]+>|<[^>]+\s\/>|<[^>\s]+>?|<?[^<\s]+>)""",
    # r"""(?:<[^>]+\w+[^>]+>|<[^>\s]+>?|<?[^<\s]+>)"""
    # Twitter username:
    r"""(?:@[\w_]+)""",
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # Remaining word types:
    r"""
    (?:[\w][\w'\-_]+[\w])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)

######################################################################
# This is the core tokenizing regex:

WORD_REGEX = re.compile(
    r"""(%s)""" % "|".join(REGEX_STRINGS), re.VERBOSE | re.I | re.UNICODE
)

# The emoticon string gets its own regex so that we can preserve case
# for them as needed:
EMOTICON_REGEX = re.compile(REGEX_STRINGS[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
HTML_DIGIT_UNICODE_REGEX = re.compile(r"&#\d+;")
HTML_ENTITY_UNICODE_REGEX = re.compile(r"&\w+;")
AMP_REGEX = "&amp;"

HEX_REGEX = re.compile(r"\\x[0-9a-z]{1,4}")

######################################################################


class Tokenizer:
    def __init__(self, preserve_case=False, use_unicode=True):
        self.preserve_case = preserve_case
        self.use_unicode = use_unicode

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the
        original string if preserve_case=False
        """
        # Try to ensure unicode:
        if self.use_unicode:
            try:
                s = str(s)
            except UnicodeDecodeError:
                s = str(s).encode("string_escape")
                s = str(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        s = self.__removeHex(s)
        # Tokenize:
        words = WORD_REGEX.findall(s)
        # print words #debug
        # Possible alter the case, but avoid changing emoticons
        # like :D into :d:
        if not self.preserve_case:
            words = map(
                (lambda x: x if EMOTICON_REGEX.search(x) else x.lower()), words
            )

        return words

    def __html2unicode(self, s):
        """
        Internal method that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(HTML_DIGIT_UNICODE_REGEX.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, chr(entnum))
                except Exception as e:
                    print("Error while decoding HTML: ")
                    print(e)

        # Now the alpha versions:
        ents = set(HTML_ENTITY_UNICODE_REGEX.findall(s))
        ents = filter((lambda x: x != AMP_REGEX), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                # Use html.entities.name2codepoint instead of html.entitydefs
                s = s.replace(ent, chr(html.entities.name2codepoint[entname]))
            except Exception as e:
                print("Error while decoding HTML: ")
                print(e)

        # Replace ampersand entities
        s = s.replace(AMP_REGEX, " and ")
        return s

    def __removeHex(self, s):
        return HEX_REGEX.sub(" ", s)
