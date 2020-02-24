"""
Coffee Factory: A multiple producer - multiple consumer approach

Generate a base class Coffee which knows only the coffee name
Create the Espresso, Americano and Cappuccino classes which inherit the base class knowing that
each coffee type has a predetermined size.
Each of these classes have a get message method

Create 3 additional classes as following:
    * Distributor - A shared space where the producers puts coffees and the consumers takes them
    * CoffeeFactory - An infinite loop, which always sends coffees to the distributor
    * User - Another infinite loop, which always takes coffees from the distributor

The scope of this exercise is to correctly use threads, classes and synchronization objects.
The size of the coffee (ex. small, medium, large) is chosen randomly everytime.
The coffee type is chosen randomly everytime.

Example of output:

Consumer 65 consumed espresso
Factory 7 produced a nice small espresso
Consumer 87 consumed cappuccino
Factory 9 produced an italian medium cappuccino
Consumer 90 consumed americano
Consumer 84 consumed espresso
Factory 8 produced a strong medium americano
Consumer 135 consumed cappuccino
Consumer 94 consumed americano
"""

from threading import Thread
from threading import Semaphore
import time 
import random 
import sys


COFFEE_SIZES = ['SMALL', 'MEDIUM', 'LARGE']
COFFEE_TYPES = ['americano', 'esspresso', 'cappuccino']
COFFEE_INTENSITY = ['strong', 'nice', 'italian']


# main product class

class Coffee:
    def __init__(self, name, size, intensity):
        self.name = name
        self.size = size
        self.intensity = intensity
        
    def get_name(self):
        raise self.name

    def get_size(self):
        return self.size
    
    def get_style(self):
        return self.intensity

    def __str__(self):
        return 'Coffee: %s size: %s intensity/style: %s' % (self.name, self.size, self.intensity)

# main channel for communication

class Distribuitor():
    def __init__(self):
        self.coffee_list = list()

    def add_coffee(self, coffee):
        self.coffee_list.append(coffee)

    def get_coffee(self):
        l = len(self.coffee_list)
        if l is not 0:
            return self.coffee_list.pop()
        else:
            return  None

# coffee factory 

class CoffeeFactory(Thread):
    def __init__(self, can_produce, can_consume, distributor, producer_id):
        Thread.__init__(self)
        self.producer_id = producer_id
        self.can_produce = can_produce
        self.can_consume = can_consume
        self.distributor = distributor
    
    def wait(self):
        time.sleep(random.uniform(0, 3))

    def produce_coffee(self):
        coffee_type = COFFEE_TYPES[random.randint(0, len(COFFEE_INTENSITY) - 1)]
        coffee_size = COFFEE_SIZES[random.randint(0, len(COFFEE_SIZES) - 1)]
        coffee_intensity = COFFEE_INTENSITY[random.randint(0, len(COFFEE_INTENSITY) - 1)]
        coffee = Coffee(coffee_type, coffee_size, coffee_intensity)
        self.distributor.add_coffee(coffee)
        print('Producer {} produced: {}'.format(self.producer_id, coffee))


    def run(self):
        while 1:
            self.wait()
            self.can_produce.acquire()
            self.produce_coffee()
            self.can_consume.release()

class User(Thread):
    def __init__(self, can_consume, can_produce, consumer_id, distributor):
        Thread.__init__(self)
        self.can_produce = can_produce
        self.can_consume = can_consume
        self.consumer_id = consumer_id
        self.distributor = distributor

    def wait(self):
        time.sleep(random.uniform(0, 3))

    def consume_coffee(self):
        coffee = self.distributor.get_coffee()
        print('Consumer: {} consumed: {}'.format(self.consumer_id, coffee))

    def run(self):
        while 1:
            self.wait()
            self.can_consume.acquire()
            self.consume_coffee()
            self.can_produce.release()


if __name__ == '__main__':
    count_providers = int(sys.argv[1])
    count_consumers = int(sys.argv[2])
    BUFFER_LEN = int(sys.argv[3])
    
    can_produce = Semaphore(BUFFER_LEN)
    can_consume = Semaphore()
    distributor = Distribuitor()

    providers = []
    consumers = []

    for i in range(count_providers):
        providers.append(CoffeeFactory(can_produce, can_consume, distributor, i))
        providers[-1].start()
    
    for i in range(count_consumers):
        consumers.append(User(can_consume, can_produce, i, distributor))
        consumers[-1].start()
    
    for p in providers:
        p.join()
    
    for u in consumers:
        u.join()
