"""
	Bonus task: load all the available coffee recipes from the folder 'recipes/'
	File format:
		first line: coffee name
		next lines: resource=percentage

	info and examples for handling files:
		http://cs.curs.pub.ro/wiki/asc/asc:lab1:index#operatii_cu_fisiere
		https://docs.python.org/2/library/io.html
		https://docs.python.org/2/library/os.path.html
"""

RECIPES_FOLDER = "recipes"
recipes = ["./recipes/americano.txt", "./recipes/espresso.txt", "./recipes/cappuccino.txt"]

def load_recipes():
	result = {}
	for recipe in recipes:
		f  = open(recipe)
		current_recipe = {}
		
		name = f.readline().rstrip('\n')
		water = f.readline()
		coffee = f.readline()
		milk = f.readline()

		result[name] = [ int(water.split("=")[1]), int(coffee.split("=")[1]), int(milk.split("=")[1]) ]				
		f.close()

	return result