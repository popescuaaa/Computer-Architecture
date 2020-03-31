"""
This module represents the Marketplace.

Computer Systems Architecture Course
Assignment 1
March 2020
"""
from threading import Lock

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
        self.global_producer_id = 0
        self.global_cart_id = 0

        self.marketplace_producers_db = {}
        self.marketplace_consumers_db = {}

        self.producer_id_lock = Lock()
        self.consumer_id_lock = Lock()

    def register_producer(self):
        """
        Returns an id for the producer that calls this.
        """

        self.global_producer_id += 1
        return self.global_producer_id

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

        self.marketplace_producers_db[producer_id] = list()
        self.marketplace_producers_db[producer_id].append(product)

        return True

    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """

        self.global_cart_id += 1
        cart_id = self.global_cart_id
        self.marketplace_consumers_db[cart_id] = list()

        return self.global_cart_id

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

                self.marketplace_consumers_db[cart_id].append((producer_id, product))
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

        for prod_tuple in self.marketplace_consumers_db[cart_id]:

            producer_id = prod_tuple[0]
            product_spec = prod_tuple[1]

            if product == product_spec:

                self.marketplace_consumers_db[cart_id].remove(prod_tuple)
                self.marketplace_producers_db[producer_id].append(product)
                break


    def place_order(self, cart_id):
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """
        consumer_list = self.marketplace_consumers_db[cart_id]
        return [prod_tuple[1] for prod_tuple in consumer_list]
