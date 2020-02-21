'''
    List comprehensions 

'''
if __name__ == "__main__":

    # calculates without any side effects 
    squares = [x**2 for x in range(10)]
    print(squares)
    
    # calculates with map and lambda 
    cubes = list(map(lambda x: x**3, range(10)))
    print(cubes)    

    # generate combinations
    main_list = [1,2,3,4]
    combs = [(x, y) for x in main_list for y in main_list]

    print(combs)

    combs2 = [(x, y) for x in main_list for y in main_list if x != y]
    print(combs2)

    # can use list comprehensions instead of map
    vec  = [1,2,3,4,5,5,6,6,7,7,8,8,9,9,0,0]
    odd = [x**2 for x in vec]
    print(odd)


    # duplicates removal
    l = [1,2,2,2,2,2,1,2,34,5,66,7]
    s = list(set(l))
    print(s)
