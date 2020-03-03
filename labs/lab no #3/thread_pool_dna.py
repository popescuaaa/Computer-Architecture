'''
 Random DNA sequence generator.

'''

import numpy as np
from time import sleep
from concurrent.futures  import ThreadPoolExecutor
from concurrent.futures import as_completed
from threading import current_thread

OUT_FILE = 'dna'
DNA_COMPONENTS = ['A', 'C', 'T', 'G']
FIND_SQ = 'TTTGGGGGCCCTCCTTGGATAGGGCGCACAGGCTTAAAGCCGT'
samples = []

class DNAGenerator():
    def __init__(self, samples_number, samples_length):
        self.samples_length = samples_length
        self.samples_number = samples_number
    
    def generate(self):
        samples = []
        print('Begin execution for: {} seqesnces of DNA with length: {}'.format(self.samples_number, self.samples_length))
        for _ in range(self.samples_number):
            sample = [np.random.choice(['A', 'C', 'T', 'G']) for _ in range(self.samples_length)]
            samples.append(''.join(sample))
        print("Finished execution\n")    
        return samples    

    def write_data(self):
        samples = self.generate()
        f = open(OUT_FILE, 'w+')
        for sq in samples:
            f.write('{}\n'.format(sq))
        f.close()


def external_find_function(idx):
    sleep(0.1)
    if FIND_SQ in samples[idx]:
        return ('Found sq in sample ix: {}\n'.format(idx))

if __name__ == "__main__":
    generator  = DNAGenerator(100, 10000)
    samples = generator.generate()
    results = []
    curr_idx = 0
    FIND_SQ = samples[0][0:100]
    with ThreadPoolExecutor(max_workers = len(samples)) as executor:
        future = executor.submit(external_find_function, curr_idx)
        curr_idx += 1
        results.append(future.result())
    for result in results:
        print(result)
