# Change Log
All notable changes to this project will be documented in this file.

## Latest

### Added

### Changed
 
### Fixed

## [0.1.2] - 2024-10-09

Migration from GitLab to GitHub.

## [0.1.1] - 2024-09-18

### Added
- Added Author list.
- Added option to plot pattern IDs of the embedding in visualization.
- Added option to pass time stamps to the visualization plots. 
- Method to convert timedelta values and datetime indices to a number of observations

### Changed
- Methods to plot the time series and embedding have been moved to `patsemb.pattern_based_embedding`
 
### Fixed
- Raise exception when the window size used for computing the pattern based 
  embedding is larger than the size of the time series
- Exceptions are changed to ValueError when seemed suitable (in `PatternBasedEmbedder`)


## [0.1.0] - 2024-08-27

First release of `PaTSEmb`! While our toolbox is still a work in progress, 
we believe it is already in a usable stage. Additionally, by publicly releasing 
`PaTSEmb`, we hope to receive feedback from the community! Be sure to check 
out the [documentation](https://patsemb-u0143709-3a07c9d27a51b62b1b2bad2f623ad154a9a19db833f1f7.pages.gitlab.kuleuven.be/)
for additional information!

### Added
A summary of the available modules within ``PaTSEmb`` is given below:
- `discretization`: a module converting a time series into a symbolic representation.
- `pattern_based_embedding`: a module for effectively transforming a time series into
  the pattern-based embedding.
- `pattern_mining`:  a module for mining patterns within the symbolic representations.
- `postprocess`: a module for postprocessing the constructed pattern-based embedding.
- `visualization`: a simple module to visualize the time series and embedding. 
 
### Changed
 
### Fixed
