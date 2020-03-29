"""
This module represents the Producer.
Computer Systems Architecture Course
Assignment 1
March 2020
"""

from threading import Thread
from time import sleep

class Producer(Thread):
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer must
        wait until the marketplace becomes available

        @type kwargs:
        @param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.current_product_index = 0
       
        self.producer_id = self.marketplace.register_producer()
    

    def run(self):
        while True:
            product = self.products[self.current_product_index % len(self.products)]
            self.current_product_index += 1

            product_spec = product[0]
            product_quantity = product[1]
            product_specific_sleep = product[2]

            while product_quantity > 0:
                result = self.marketplace.publish(self.producer_id, product_spec)
                if (result == True):
                    product_quantity -= 1
                    sleep(product_specific_sleep)
                else:
                    sleep(self.republish_wait_time)
                    continue
            
