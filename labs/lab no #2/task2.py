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
import sys
from threading import Thread, Lock
from  functools import reduce
import time 
shared_var = 0

class CustomThread(Thread):
    def __init__(self, thread_id, input_list, lock, shared_var):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.input_list = input_list
        self.lock = lock
        self.shared_var = shared_var
    
    def run(self): 
        index = random.randint(0, len(self.input_list)-1)
        time.sleep(random.randint(0, 1))
        with lock:
            curr_elem = self.input_list[index][0]
            while self.input_list[index][1]:
                index = random.randint(0, len(self.input_list)-1)
                curr_elem = self.input_list[index][0]
            self.input_list[index][1] = True # marked as selected
            self.shared_var += curr_elem

        print("Current thread: %s has a value for the shared variable equal to: %s" % (self.thread_id, self.shared_var))

if __name__ == "__main__":
    num_threads = int(sys.argv[1])

    input_list = [random.randint(0, 500) for i in range(num_threads)]
    initial_sum = reduce(lambda x, y: x + y, input_list)
    
    input_dict = []
    for entry in input_list:
        input_dict.append([entry, False])


    print(" ".join([str(x) for x in input_list]))

    another_shared_var = 0
    lock = Lock()
    thread_array = []

    for i in range(num_threads):
        thread_array.append(CustomThread(i, input_dict, lock, another_shared_var))
    for entry in thread_array:
        entry.start()
    for entry in thread_array:
        entry.join()


    print(another_shared_var)