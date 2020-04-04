"""
This module represents the Producer.
Computer Systems Architecture Course
Assignment 1
March 2020
"""

from threading import Thread
from time import sleep

PRODUCT_SPECS = 0
PRODUCT_QUANTITY = 1
PRODUCT_SLEEP_TIME = 2

class Producer(Thread):
    """
    Class that represents a consumer.
    """

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
        # get the specific if for the producer
        with self.marketplace.producer_id_lock:
            self.producer_id = self.marketplace.register_producer()

    def run(self):
        limit = len(self.products)
        # the producer has a continuous process of production
        while True:
            product = self.products[self.current_product_index % limit]
            self.current_product_index += 1
            # decompose current << product >>  specs
            product_specs = product[PRODUCT_SPECS]
            product_quantity = product[PRODUCT_QUANTITY]
            product_sleep_time = product[PRODUCT_SLEEP_TIME]

            while product_quantity > 0:
                result = self.marketplace.publish(self.producer_id, product_specs)
                # of the result is true -> append to marketplace queue
                if result:
                    product_quantity -= 1
                    sleep(product_sleep_time)
                # otherwise the producer has to wait for consummers to
                # get the products form the marketplace
                else:
                    sleep(self.republish_wait_time)
                    continue
