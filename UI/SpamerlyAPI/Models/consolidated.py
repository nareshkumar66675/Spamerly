from svm_predict import *


def callFunction(text,name):
	if name == "SVM":
		return svm_model(text)
	elif name == "NN":
		return nn_model(text)
	elif name == "Bayesian":
		return  bayesian_model(text)
	else:
		return Null	


def main():
	print(callFunction("I am Sandip Dey", "SVM"))


if __name__ == '__main__':
    main()
