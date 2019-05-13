# Spamerly

Yeah!!!! It is a Rip off of Grammerly. But it is different.


# Overview

  - This application helps the user to identify the spam while composing an article/email.
  - So, the users need not worry that the mails will end up in the spam folder of the reciever.
  - This helps the genuine/advertiser to evade spam filtration done by most services.

# Classification Models:
##### Bayesian
##### Support Vector Machines
##### Neural Network



# Prediction Rate
![PredictionRate](Viz/PredictionRate.png)

# Application Demo
- It is a web application.
- User can enter the data and check for spam or not.
#### Demo 1: Entering the text and Checking for Spam
1) Enter the text in the editor.
2) For every senetence(identified by a *"period(.)"*) the spam is highlighted in the below block.
3) **Spam Levels:**

| Highlighted Color | Spam Severity |
|-------------------|---------------|
| None              | Not a Spam    |
| Grey              | Low           |
| Yellow            | Medium        |
| Red               | high          |


![Demo1](Viz/BayesianDemo.gif)

#### Demo 2: Changing the model
1) User can change the model by selecting the model from drop down.
2) All the text entered will be updated using the new selected model.
3) Additionally, if user hover over the highlighted text, the actual probability of the text being spam is shown.

![enter image description here](Viz/SVMDemo.gif)

# Project Struture
##### Spamerly
- **/Model**  - All Files related to Models
	- NN 
	- SVM
	- Bayesian
- **/UI** - All files related to creating the web Application
	- SpamerlyAPI - API to get the spam probability from the models
	- Specsy - UI for the app
- **/Viz** - All Visualizations and Demo
- **/Data Acquire** -  Retrieve spam data from Gmail 


# Presentation

[Link to PDF](Viz/Spamerly.pdf)

[Link to PPT](Viz/Spamerly.pptx)


  



