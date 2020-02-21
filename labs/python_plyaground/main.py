'''
    Data Structures manipulation

    Shallo copy and deepcopy
'''
from random import randint
from copy import deepcopy
from copy import copy
from functools import reduce

if __name__ == "__main__":
    l = list()
    for i in range(0, 1000):
        l.append(randint(0, 1000))
    
    l2 = list(map(lambda x: x/2, l))
   
    l3 = list(filter(lambda x: x//2 == 1, l))
    
    # this creates biding between target and object 
    l4 = l3
    # l4[0] = 1000

    # this creates a deep copy of the object
    l5 = deepcopy(l3)

    #this creates a shallow copy of the objcet
    l6 = copy(l3)

    # The difference between shallow copy and deep copy is only
    # relevant for compound objects 
    l5[0] = 10000
    #l3_str = ' '.join(list(map(str, l3))
    #l5_str = ' '.join(list(map(str, l5))
    print('{}\n'.format(list(map(str, l3))))
    print('{}\n'.format(list(map(str, l5))))

    # Reduce function: [x, y, z] => reduced to a single varibale
    b = [1,2,3,4,5,6,7,8,8,9,0,0,-1]
    r = reduce((lambda x,y: x + y), b)
    print(r)