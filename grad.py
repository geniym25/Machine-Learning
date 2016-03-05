import math
import random

#constants
lambdaConst = 0.00004
deltaLambda = 0.99999
EPS = 1e-9
minX = -10
maxX = 10
MAXIterations = 1e5

#lambda for method 3
def dichotomyLambda(x, grad, function):
	l = 0
	r = 1
	while(r - l > EPS):
		x1 = l + (r - l) / 3
		x2 = l + (r - l) / 3 * 2
		f1 = function(diff(x, multVectorByConstant(grad, x1)))
		f2 = function(diff(x, multVectorByConstant(grad, x2)))
		if(f1 > f2):
			l = x1
		else:
			r = x2
	f1 = function(diff(x, multVectorByConstant(grad, l)))
	f2 = function(diff(x, multVectorByConstant(grad, r)))
	if(f1 < f2):
		return l
	else:
		return r
	return l

#functions
def twoDimensionalRosenbrock(x):
	return ((1 - x[0]) ** 2) + 100 * ((x[1] - x[0] ** 2) ** 2)

def multidimensionalRosenbrock(n):
	def f(x):
		res = 0
		for i in range(int(n/2)):
			res = res + (1 - x[2 * i]) ** 2 + 100 * ((x[2 * i + 1] - x[2 * i] ** 2) ** 2)
		return res
	return f
	
#Gradients
def gradTwo(x):
	res = []
	res.append(2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1))
	res.append(200 * (x[1] - x[0] ** 2))
	return res

def gradMultidimensional(n):
	def grad(x):
		res = []
		for i in range(int(n/2)):
			res.append(2 * (200 * x[2 * i] ** 3 - 200 * x[2 * i] * x[2 * i + 1] + x[2 * i] - 1))
			res.append(200 * (x[2 * i + 1] - x[2 * i] ** 2))
		return res
	return grad
		

#vector difference
def diff(x1, x2):
	x = []
	for i in range(len(x1)):
		x.append(x1[i] - x2[i])
	return x

#vector length
def length(x):
	res = 0
	for i in x:
		res = res + i ** 2
	return math.sqrt(res)
	

def multVectorByConstant(x, k):
	res = []
	for i in x:
			res.append(i * k)
	return res


def GradientDescent(x, function, GradientF, method):
	Lambda = lambdaConst
	if(method == 3):
			Lambda = dichotomyLambda(x, GradientF(x), function)
	nextX = diff(x, multVectorByConstant(GradientF(x), Lambda))
	Iterations = 0
	while(length(diff(x, nextX)) > EPS 
          and math.fabs(function(x) - function(nextX)) > EPS 
          and length(GradientF(x)) > EPS
          and Iterations < MAXIterations):
		x = nextX
		if(method == 2):
			Lambda = Lambda * deltaLambda
		if(method == 3):
			Lambda = dichotomyLambda(x, GradientF(x), function)
		nextX = diff(x, multVectorByConstant(GradientF(x), Lambda))
		if(Iterations > 100000):
			print(Iterations)
		Iterations += 1
	x = nextX
	return x
	
def MonteCarlo(n, vLength, function, GradientF, method):
	minF = math.inf
	for i in range(n):
		x = []
		for j in range(vLength):
			x.append(random.uniform(minX, maxX))
		print("x0 == ", x)
		x = GradientDescent(x, function, GradientF, method)
		print("x == ", x, "\tf(x) == ", function(x), "\n------------------------")
		if function(x) < minF:
			minF = function(x)
			minArg = x
	return minArg

print("Двумерная функция Розенброка")
minArg = MonteCarlo(5, 2, twoDimensionalRosenbrock, gradTwo, 3)
print("x = ", minArg)
print("f(x) = ", twoDimensionalRosenbrock(minArg))

print("Многомерная функция Розенброка")
minArg = MonteCarlo(5, 4, multidimensionalRosenbrock(4), gradMultidimensional(4), 3)
print("x = ", minArg)
print("f(x) = ", multidimensionalRosenbrock(4)(minArg))
