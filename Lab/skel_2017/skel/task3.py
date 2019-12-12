import io
import re # for multiple delimiter split
"""
Objectives:
- Lists
- Dict
- constructs: loops, conditionals
- more work with files
- functions
"""

#TODO implement the following functions

# function #1
"""
    Returns the number of lines in a file given as parameter.
    @param filename: the file's name
"""
def number_of_lines(filename):
    f = open(filename, "r")
    counter  = 0
    for line in f:
        counter += 1
    f.close()
    return counter


# function #2
"""
    Reads the content of a file and fills the given list with the sentences
    found in the file
    @param filename: the file's name
    @param sentences: the list that will be contain the sentences
"""
def fill_sentences(filename, sentences):
    f = open(filename, "r")
    inner_sentences = []
    for line in f:
        inner_sentences.append(line)
    sentences = inner_sentences
    f.close()
# function #3
"""
    Return a list of the top N most used words in a given file
    @param filename: the file's name
    @param n: the number of words in the top, default is 5
"""
#bonus 1p: implement your own sort method instead of using an existing one
def top_used_words(filename, n = 5):
    f  = open(filename, "r")
    words = []
    pattern = " ,.;?!"
    pattern = '|'.join(map(re.escape, pattern))
    for line in f:
       line_words = re.split(pattern, line)
       for entity in line_words:
           words.append(entity)
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    max_count = 0
    for entity in word_freq:
        if word_freq[entity] > max_count:
            max_count = word_freq[entity]
    
    number_of_elements = n
    word_freq = sorted(word_freq, reverse=True)
    for entity in word_freq:
        if number_of_elements > 0:
            print entity                
            number_of_elements -= 1
        else:
            break

if __name__ == "__main__":

    filename = "fisier_input"

    # TODO test the functions
    print number_of_lines(filename)
    # TODO print all the sentences with less than 15 words
    sentences = []
    fill_sentences(filename, sentences)
    for sentence in sentences:
        print "Idf as a new sentence: %s" % sentence
    # TODO write the most used words in a file, one per line
    top_used_words(filename)