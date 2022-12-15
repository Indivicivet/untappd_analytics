Various scripts to make graphs and/or dump out data about a user's untappd beer checkins. These are written for personal interest only so to-do list may never get ticked off. You will need an untappd data file, which requires being an untappd subscriber (or finding a friend with one).

    pip install -e .

`untappd.py` contains core functionality to load check-in data from a json file. The other python files in scripts/ make graphs or print some data.

# todos

- ~~generate fake untappd data so can run examples without being a subscriber~~
-- you can do this by running generate_sample_data_source.py; it doesn't generate all fields, so may not work for some scripts that want things other than datetime and ratings
- cache beers/venues/breweries
- ~~support CSV files~~ this is done I think, although haven't tested properly
- ~~structure as actual package+examples rather than random collection of scripts~~ scripts/ :)
- make it easy to run all scripts in batch and have them all save out their results
