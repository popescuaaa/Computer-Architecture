from threading import Thread, Lock
from random import randint, uniform, seed
from time import sleep


FULL = 'full'
HUNGRY = 'hungry'

class Philosopher(Thread):
    
    # base statisfiability constratint per instance
    __status = HUNGRY

    def get_status(self):
        return self.__status
    
    def set_status(self, new_status):
        self.__status = new_status

    def __init__(self, name, forkOnLeft, forkOnRight):
        Thread.__init__(self)
        self.name = name
        self.forkOnLeft = forkOnLeft
        self.forkOnRight = forkOnRight
 
    def run(self):
        while self.__status == HUNGRY:
            print('{} is hungry.'.format(self.name))
            sleep(10)
            self.dine()
 
    def dine(self):
        fork1, fork2 = self.forkOnLeft, self.forkOnRight
        while self.__status == HUNGRY:
            fork1.acquire(True)
            locked = fork2.acquire(False)

            if locked: 
                break
            fork1.release()

            print('{} swaps forks'.format(self.name))
            fork1, fork2 = fork2, fork1
        else:
            return

        self.dining()
        fork2.release()
        fork1.release()
 
    def dining(self):			
        print('{} starts eating '.format(self.name))
        for i in range(10): print('{} YUM YUM YUM YUM YUM YUM'.format(self.name))
        print('{} finishes eating '.format(self.name))
 
def main():
    forks = [Lock() for n in range(5)]
    philosopher_names = ('Aristotle','Kant','Buddha','Marx', 'Russel')
    philosophers= [Philosopher(philosopher_names[i], forks[i%5], forks[(i+1)%5]) \
            for i in range(5)]
    seed(42)
    for p in philosophers: p.start()
    sleep(1000)
    Philosopher.set_status(FULL)
    print ("Now we're finishing.")
 
if __name__ == "__main__":
     main()