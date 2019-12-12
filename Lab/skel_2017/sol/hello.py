"""
Task 1 - Let's print the current time!

Objectives:
- use documentation!!!
- print
- print with %
- print without new line
- format

Print the current date and time. Also print it using % and using format().
---> see examples from 'Stringuri' section.
"""


# Hints for date and time: datetime module for objects representing time,
# strftime function for formatting
# https://docs.python.org/2/library/datetime.html#datetime-objects
# https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior

# Use only the official Python documentation to solve this exercise.

import datetime
print "Cerinta: afisati data si ora curenta in acest format: 19 February 2017 19:57:51"

i = datetime.datetime.now()

print "Today is",
print i.strftime("%d %B %Y %H:%M:%S")

print "Today is {}, and I'm attending {} lab".format(
                                    i.strftime("%d %B %Y %H:%M:%S"), "ASC")

print "Today is %s, and I'm attending %s lab"%(i.strftime("%d %B %Y %H:%M:%S"),
                                                "ASC")
