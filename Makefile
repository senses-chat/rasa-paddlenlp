.EXPORT_ALL_VARIABLES:

RASA_X_PASSWORD = Qwe123123
ACTION_ENDPOINT = http://localhost:5055/webhook
RECOGNIZERS_SERVICE_URL = http://localhost:7000/recognize/number

install:
	pipenv install --skip-lock

train:
	pipenv run rasa train $(args)

shell:
	pipenv run rasa shell $(args)
