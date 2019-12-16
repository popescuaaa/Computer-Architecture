"""
    Basic thread handling exercise:

    Use the Thread class to create and run more than 10 threads which print their name and a random
    number they receive as argument. The number of threads must be received from the command line.

    e.g. Hello, I'm Thread-96 and I received the number 42

"""
import sys
import threading
import random
from threading import Thread


class CustomThread(Thread):
    def __init__(self, message, thread_id):
        Thread.__init__(self)
        self.message = message
        self.thread_id = thread_id

    def run(self):
        print "Thread %s: %s\n" % (self.thread_id, self.message)
        
        
def main():
    number_of_threads = int(sys.argv[1])
    threads_array = []
    for i in range(number_of_threads):
        threads_array.append(CustomThread("Hi <3", i))
    for thread_entry in threads_array:
        thread_entry.start()
    for thread_entry in threads_array:
        thread_entry.join()

if __name__ == "__main__":
    main()
