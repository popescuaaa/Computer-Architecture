# More python stuff!!!

""" Mutability """

l1 = [0, 1, 2, 3, 4, 5]
l2 = l1

# TODO modify l2, print l1 and l2

######
for i in range(0, 100):
    l2.append(i)

print l1 == l2
print l1
print l2


t1 = ('a', 'b')
t2 = t1

t1 = ('x', 'y')

# TODO modify t1, print t1 and t2

######

print t1 == t2
print t1
print t2

s1 = "abc"
s2 = s1

print id(s1)
print id(s2)

s1 = s1.upper()

print s1

# TODO print s1 and s2 and their id()

######

print s1
print s2
print id(s1)
print id(s2)
""" List comprehensions """

my_list = [5] * 5 + [3] * 2 + [4] * 2 + [10] * 3
# TODO: print a list of all the odd elements divided by 10, using list comprehensions
new_range  = [float(i) / 10  for i in range(10)  if i % 2 == 0] 
print new_range

""" Generators """

# what happens if we use tuple comprehensions? 
# TODO replace the [ with ( in the expression used for the previous todo
# try to print it


# TODO use xrange instead of range in the following code ... what is the difference?

for i in range(100000):
   pass

print

""" Identity vs equality """

size = 10
a = 'a' * size
b = 'a' * size

print a == b
print a is b

print id(a)
print id(b)

print
# TODO check the size from which a no longer has the same references as b



n = 10^(1)
m = 10

print n == m
print n is m

print id(n)
print id(m)

# TODO Give n and m a value so that n is m is False

