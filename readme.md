# Autocorrect project
Sample query autocorrection project. Frames autocorrect as a ranking problem, assuming a set of candidate corrections is given.
The highest ranked candidate can be treated as the autocorrected query.

We won't bother to split the data into training/validation/test sets in this project.

## setup
Use Python 3. You may want to use a virtual environment as well

`pip install -r requirements.txt` ()  

## prepare query data
`python prepare_queries.py`

Some prelabelled queries are provided, including both correctly spelled queries, and typo-corrected query pairs.
In a real scenario, we could generate candidates with Elasticsearch and treat them as incorrect candidates.
In this project, we generate some random alterations of the input query.

## train model
`python model.py train`

Using XGBoost ranker.
Main features used in this project are edit distance related counts.
Eg. for the pair `tshot -> tshirt`, the feature `replace_o_to_i` is 1, `add_r` is 1
The idea is some edits are more common/natural than others (eg. characters close together on the keyboard)

## evaluate
`python model.py evaluate`

## interactive predict
`python model.py predict`

## edit distance unit test
`pytest test_edit_distance.py`
