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
COFFEE_SIZES = ['SMALL', 'MEDIUM', 'LARGE']

class Coffee:
    def __init__(self, name, size):
        self.name = name 
        self.size = size

    def get_name(self):
        raise self.name

    def get_size(self):
        return self.size
    
    def __str__(self):
        return 'Coffee: %s size: %s' % (self.name, self.size)

class Americano(Coffee):
    def __init__(self, name = 'americano', size = 'LARGE'):
        Coffee.__init__(self, name, size)
    
class Espresso(Coffee):
    def __init__(self, name = 'esspresso', size = 'SMALL'):
        Coffee.__init__(self, name, size)

class Cappuccino(Coffee):
    def __init__(self, name = 'cappuccino', size = 'MEDIUM'):
        Coffee.__init__(self, name, size)

class ExampleCoffee:
    """ Espresso implementation """
    def __init__(self, size):
        pass

    def get_message(self):
        """ Output message """
        raise NotImplementedError


if __name__ == '__main__':
    coffee = Americano()
    print(coffee)
