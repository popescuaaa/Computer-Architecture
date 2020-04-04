"""
This module represents the Consumer.

Computer Systems Architecture Course
Assignment 1
March 2020
"""

from threading import Thread
from time import sleep

OP_TYPE = 'type'
OP_PRODUCT = 'product'
OP_QUANTITY = 'quantity'

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

        # get the specific id for every consumer
        with self.marketplace.consumer_id_lock:
            self.consumer_id = self.marketplace.new_cart()
        # current set of opperations initialized at 0
        self.current_index_op_set = 0
        self.limit = len(self.carts)

    def run(self):
        # iterate trough all sets of operations
        while self.current_index_op_set < self.limit:
            # current operation index initialized at 0
            current_index_op = 0
            current_index_set = self.current_index_op_set
            while current_index_op < len(self.carts[current_index_set]):

                operation_dict = self.carts[current_index_set][current_index_op]
                # decompose the current operation and extract data
                op_type = operation_dict[OP_TYPE]
                op_product = operation_dict[OP_PRODUCT]
                op_quantity = operation_dict[OP_QUANTITY]

                if op_type == 'add':
                    # for the add operation call the add_to_cart 'quantity' times
                    q_i = 0
                    while q_i < op_quantity:
                        # get the result in separate variable
                        result = self.marketplace.add_to_cart(self.consumer_id, op_product)
                        if not result:
                            # if the product is not on the market then wait
                            sleep(self.retry_wait_time)
                            continue
                        q_i += 1
                else:
                    # otherwise remove the specified quantity from cart
                    q_i = 0
                    while q_i < op_quantity:

                        self.marketplace.remove_from_cart(self.consumer_id, op_product)
                        q_i += 1

                current_index_op += 1
            self.current_index_op_set += 1
        # retrive data from the marketplace
        cart_products = self.marketplace.place_order(self.consumer_id)
        # print with specified format
        for product in cart_products:
            print('{} bought {}'.format(Thread.getName(self), product))
        