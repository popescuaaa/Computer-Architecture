import task3

import sys, os.path

"""
Task 4 - import your module

The filename is given as command-line argument 
    ---> see how we obtain it and check its existence

"""

def usage(argv): 
    print "Usage: " + argv[0] + " input_file" 

if __name__ == "__main__":

    # command line arguments parsing 
    if __name__ == "__main__": 
        if len(sys.argv) < 2: 
            usage(sys.argv)     
            exit(0) 
 
    # check 
    if not os.path.isfile(sys.argv[1]): 
        print "Not a valid file: " + sys.argv[1] 
        exit(0)  
    
    # test one of the functions from task3
    
    print task3.num_lines(sys.argv[1])
