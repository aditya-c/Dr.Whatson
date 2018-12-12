# Dr.Whatson
Med 277 Project
A google assistant bot that aims to predict diseases based on symptoms.

DialogFlow has been used to converse with a patient and a combination of rake, naive bayes classifier and a kaggle dataset, we suggest probable symtoms and diseases to the patient.

## Data
The data from the kaggle challenge available [here](https://www.kaggle.com/plarmuseau/primer/data) has been used for the predictive tasks. We use `dia_3.csv`, `sym_3.csv` and `sym_dis_matrix.csv`.

## Heroku App
You can deploy the server code on heroku from here:

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

## GoogleAssistant App
The zip of Dr.Whatson's Dialogflow agents and intents is available [here](https://drive.google.com/file/d/1As4hW-UfV1pP5v0B6PrcQypy7qw2ujHd/view?usp=sharing).


To recreate the app:
- Fork/make a copy of the repository. Upload datafiles from the link to a new `data` folder 
- Create and launch the Heroku server (Click the button above)
- Create a new DialogFlow agent [here](https://dialogflow.com)
- Import all contents of the zip downloaded above into this agent
- Set fulfilment webhook url as `<your heroku-app>/webhook`

You will be able to access the app through the google assitant on your devices.
