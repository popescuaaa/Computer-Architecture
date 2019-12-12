"""

Objectives:
- Lists
- Dict
- constructs: loops, conditionals
- more work with files
- functions
"""

import io
def num_lines(filename):
    """
        Returns the number of lines in a file given as parameter.
        @param filename: the file's name
    """
    count = 0
    with open(filename, "r") as f:
        for line in f:
            count += 1
    return count


def get_sentences(filename, sentences):
    """
        Reads the content of a file and fills the given list with the sentences
        found in the file
        @param filename: the file's name
        @param sentences: the list that will be contain the sentences
    """

    with open(filename, "r") as f:
        s = f.read()
    sentences += s.split(". ") # !!! you have to modify the given list object,
                               # if you use just sentences = ..., the list will
                               # be the same outside the function


def get_top_words(filename, num = 5):
    """
        Return a list of the top N most used words in a given file
        @param filename: the file's name
        @param n: the number of words in the top, default is 5
    """

    with open(filename, "r") as f:
        s = f.read()

    words = s.split()

    print words

    stats = {}

    for w in words:
        if w not in stats:
            stats[w] = 1
        else: stats[w] += 1

    print stats

    return sorted(stats, key=stats.get, reverse=True)[0:5]

    #bonus 1p: implement your own sort method instead of using an existing one

if __name__ == "__main__":

    # test the methods
    filename = "fisier_input"

    print num_lines(filename)

    sentences = []
    get_sentences(filename, sentences)

    print sentences

    # print all the sentences with less than 15 words

    for sentence in sentences:
        if len(sentence.split()) < 15:
            print sentence

    print "-----------------------"
    words = get_top_words(filename)


    # write the most used words in a file, one per line
    with open("top_words", 'w') as f:
        f.writelines("\n".join(words))
        f.write("\n")
