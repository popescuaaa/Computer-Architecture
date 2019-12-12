# More python stuff!!!

""" Mutability """

l1 = [0,1,2,3,4,5]
l2 = l1

# TODO modify l2, print l1 and l2

l2[3] = 'a'
print l1
print l2

t1 = ('a', 'b')
t2 = t1

print

# TODO modify t1, print t1 and t2
#t2[1] = 0 -- invalid operation

s1 = "abc"
s2 = s1

print id(s1)
print id(s2)

s1 = s1.upper()

print

# TODO print s1 and s2 and their id()

print id(s1)
print id(s2)

print s1
print s2

print

""" List comprehensions """

my_list = [5]*5 + [3] * 2 + [4] * 2 + [10] * 3
print [elem/10.0 for elem in my_list if elem%2]
print
# TODO: print a list of all the odd elements divided by 10, using list comprehensions
    
""" Generators """


# what happens if we use tuple comprehensions? 
# TODO replace the [ with ( in the expression used for the previous todo
# try to print it


print type((elem/10.0 for elem in my_list if elem%2))
tc = (elem/10.0 for elem in my_list if elem%2)

for i in tc: print i 

print

# TODO use xrange instead of range in the following code ... what is the difference?

for i in range(100000):
    pass

print

""" Identity vs equality """

size = 10;
a = 'a' * size
b = 'a' * size

print a == b
print a is b

print id(a)
print id(b)

print
# TODO check the size from which a no longer has the same references as b

n = 10
m = 10

print n == m
print n is m

print id(n)
print id(m)

# TODO Give n and m a value so that n is m is False

# read more: http://blog.lerner.co.il/why-you-should-almost-never-use-is-in-python/


