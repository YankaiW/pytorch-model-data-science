coding_standards:
	poetry run isort .
	poetry run black .
	poetry run flake8 .