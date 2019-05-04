from svm_predict import *
from bayesian import *


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
	print(callFunction("07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow", "Bayesian"))


if __name__ == '__main__':
    main()
