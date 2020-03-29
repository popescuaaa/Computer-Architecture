"""
This module represents the Marketplace.

Computer Systems Architecture Course
Assignment 1
March 2020
"""
from threading import Lock
from copy import deepcopy

class Marketplace:
    """
    Class that represents the Marketplace. It's the central part of the implementation.
    The producers and consumers use its methods concurrently.
    """
    def __init__(self, queue_size_per_producer):
        """
        Constructor

        :type queue_size_per_producer: Int
        :param queue_size_per_producer: the maximum size of a queue associated with each producer
        """
        self.limit = queue_size_per_producer
        self.global_id = 0
        self.marketplace_producers_db = {}  # < producer id, list (max_size)>
        self.marketplace_consumers_db = {}  # < cart_id, shopping_list>
        self.carts = list()
        self.current_index_cart = 0

    def get_db_status(self):
        return len(self.marketplace_db)

    def register_producer(self):
        """
        Returns an id for the producer that calls this.
        """
        self.global_id += 1
        return self.global_id

    def publish(self, producer_id, product):
        """
        Adds the product provided by the producer to the marketplace

        :type producer_id: String
        :param producer_id: producer id

        :type product: Product
        :param product: the Product that will be published in the Marketplace

        :returns True or False. If the caller receives False, it should wait and then try again.
        """
        if producer_id in self.marketplace_producers_db:
            producer_list = self.marketplace_producers_db[producer_id]
            if len(producer_list) < self.limit:
                self.marketplace_producers_db[producer_id].append(product)
            else:
                return False
            return True
        else:
            self.marketplace_producers_db[producer_id] = list()
            self.marketplace_producers_db[producer_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """
        self.current_index_cart += 1
        cart_id = self.current_index_cart
        self.marketplace_consumers_db[cart_id] = list()

        return self.current_index_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to the given cart. The method returns

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to add to cart

        :returns True or False. If the caller receives False, it should wait and then try again
        """
        for producer_id in self.marketplace_producers_db:
            current_list = self.marketplace_producers_db[producer_id]
           
            if product in current_list:
                self.marketplace_consumers_db[cart_id].append(product)
                self.marketplace_producers_db[producer_id].remove(product)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        """
        if product in self.marketplace_consumers_db[cart_id]:
            self.marketplace_consumers_db[cart_id].remove(product)
           
            for producer_id in self.marketplace_producers_db:
                current_list = self.marketplace_producers_db[producer_id]
                if len(current_list) < self.limit // 2:
                    self.marketplace_producers_db[producer_id].append(product)

    def place_order(self, cart_id):
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """
        return self.marketplace_consumers_db[cart_id]