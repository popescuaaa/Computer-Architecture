import sys
from  load_recipes import load_recipes
import time
print "I'm a simple coffee maker\n"



# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"  #!!! when making coffee you must first check that you have enough resources!
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

"""
Example result/interactions:

I'm a smart coffee maker
Enter command:	
list
americano, cappuccino, espresso
Enter command:	
status
water: 100%
coffee: 100%
milk: 100%
Enter command:	
make
Which coffee?
espresso
Here's your espresso!
Enter command:	
refill
Which resource? Type 'all' for refilling everything
water
water: 100%
coffee: 90%
milk: 100%
Enter command:	
exit
"""

# Coffee examples
ESPRESSO = "espresso"
AMERICANO = "americano"
CAPPUCCINO = "cappuccino"

# Resources examples
WATER = "water"
COFFEE = "coffee"
MILK = "milk"

# Coffee maker's resources - the values represent the fill percents
resources = {WATER: 100, COFFEE: 100, MILK: 100}
recipes = [ESPRESSO, AMERICANO, CAPPUCCINO]
resources_entries = [WATER, COFFEE, MILK]

#prepare the recipe

recipes  = load_recipes()

def perform_action(user_answer):
    if user_answer == EXIT:
        print "All the best! Bye!\n"
        sys.exit()
    if user_answer == LIST_COFFEES:
        for recipe in recipes:
            print recipe, "\n"
    if user_answer == RESOURCE_STATUS:
        for i in range(0, 3):
           print resources_entries[i], ": ", resources[resources_entries[i]], " %" 
    if user_answer == REFILL:
        user_resource = sys.stdin.readline().rstrip('\n')
        
        if user_resource == 'all':
            for resource in resources:
                resources[resource] = 100
        elif user_resource in resources:
            resources[user_resource] = 100
        else:
           print "No such resource!\n"    
    if user_answer == MAKE_COFFEE:
        user_choice = sys.stdin.readline().rstrip('\n')
        print recipes
        if user_choice in recipes:
            print "Wait a couple of seconds..I am not a racket...\n"
            time.sleep(5)
            
            water_needed = recipes[user_choice][0]
            coffee_nedded = recipes[user_choice][1]
            milk_nedded = recipes[user_choice][2]
            if resources[WATER] - water_needed > 0:
                resources[WATER] -= water_needed
            else:
                print "No resources\n"
            if resources[COFFEE] - coffee_nedded > 0:
                resources[COFFEE] -= coffee_nedded
            else:
                print "No resources\n"
            if resources[MILK] - milk_nedded > 0:
                resources[MILK] -= milk_nedded
            else:
                print "No resources\n"

            print "Your coffee is served! \n"
        else:
            print "I don't know such a recipe!\n"
            print "Plase choose from: \n\t americano, cappucciono, espresso \n"

def interactions():
    while True:
        user_answer = sys.stdin.readline().rstrip('\n')
        if user_answer in commands:
            perform_action(user_answer)
        else:
            print "The command is not good! I don't have such commands in memory!\n"
            print "Try the following commands: \n"
            for command in commands:
                print command, " \n"



if __name__ == "__main__":
    
    interactions()
