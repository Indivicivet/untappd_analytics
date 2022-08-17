Various scripts to make graphs and/or dump out data about a user's untappd beer checkins. These are written for personal interest only so to-do list may never get ticked off. You will need an untappd data file, which requires being an untappd subscriber (or finding a friend with one).

`untappd.py` contains core functionality to load check-in data from a json file. The other python files make graphs or prints some data.

# todos

- generate fake untappd data so can run examples without being a subscriber
-- this is wip, see todos in generate_sample_data_source.py
- cache beers/venues/breweries
- ~~support CSV files~~ this is done I think, although haven't tested properly
- structure as actual package+examples rather than random collection of scripts
