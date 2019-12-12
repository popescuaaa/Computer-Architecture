"""
    Create and run more than 1000 threads that work together. They share a list,
    input_list, and an integer variable.
    When they run, they each select one random item from the input list, that
    was not previously selected, and add it to that variable.
    When they finish, all elements in the list should have been selected.
    Make the critical section(s) as short as possible.

    Requirements:
        * the length of the input list is equal to the number of threads
        * the threads have a random delay when running (see time.sleep), before
            accessing the input_list, in the order of 10-100ms
        * before starting the threads compute the sum of the input list
        * after the threads stopped, the shared variable should be identical to
            the sum of the input list

    Hint:
        * In CPython some operations on data structures are atomic, some are
            not. Use locks when the operations are not thread-safe.
        * Useful links:
        https://docs.python.org/2/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe
"""

import random

shared_var = 0


if __name__ == "__main__":

    #TODO provide the number of threads from the command line
    num_threads = 10

    input_list = [random.randint(0, 500) for i in range(num_threads)]
    initial_sum = reduce(lambda x, y: x + y, input_list)
    print " ".join([str(x) for x in input_list])

    print initial_sum
    print shared_var
