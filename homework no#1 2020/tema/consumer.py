"""
This module represents the Consumer.

Computer Systems Architecture Course
Assignment 1
March 2020
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Class that represents a consumer.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor.

        :type carts: List
        :param carts: a list of add and remove operations

        :type marketplace: Marketplace
        :param marketplace: a reference to the marketplace

        :type retry_wait_time: Time
        :param retry_wait_time: the number of seconds that a producer must wait
        until the Marketplace becomes available

        :type kwargs:
        :param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_id = self.marketplace.new_cart()
        self.kwargs = kwargs
        self.current_index_op_set = 0
        self.limit = len(self.carts)
    
    def run(self):
        while self.current_index_op_set < self.limit:
            current_index_op = 0
            while current_index_op < len(self.carts[self.current_index_op_set]):
                operation_dict = self.carts[self.current_index_op_set][current_index_op]
                op_type = operation_dict['type']
                op_product = operation_dict['product']
                op_quantity = operation_dict['quantity']
                
                if op_type == 'add':
                    i = 0
                    while i < op_quantity:
                        result = self.marketplace.add_to_cart(self.consumer_id, op_product)
                        if result == False:
                            sleep(self.retry_wait_time)
                            # print('{}'.format("One time"))
                            continue
                        else:
                            i += 1
                else:
                    for i in range(0, op_quantity):
                        self.marketplace.remove_from_cart(self.consumer_id, op_product)
                current_index_op += 1

            self.current_index_op_set += 1
        
        cart_products = self.marketplace.place_order(self.consumer_id)
        for product in cart_products:   
            print('{} bought {}'.format(Thread.getName(self), product))
        