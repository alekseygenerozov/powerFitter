# Power Law Fitting Tool

## Overview

The main application, `power_law_fit.py`, finds the best fit (truncated) 
power-law distribution for 1D data.

Users must provide the name of a data file, which should contain only
positive real numbers. If the data contains invalid values,
a ValueError will be raised.

Additionally, users can specify inner and outer truncation limits 
for the power law using the `--min_value` and `--max_value` options. 
If these options are not specified, the minimum and maximum values 
from the data will be used. Data outside this range will not be included 
in the fitting process.

These truncation settings can be useful for fitting data
described by a broken power law; however, setting these values beyond 
the actual data range may yield unexpected results.

Finally, users can include a flag to obtain an uncertainty
estimate for the fit using `--uncertainty`.

## Files:
- **power_law_fit.py**: Routines for power_law fitting
- **test_fit.py**: Unit tests 
- **test_data**: Example data file
- **README.md**

## Requirements
- Python 3.6 or higher
- Numpy 
- Scipy

### Basic Usage
To fit a power law to data in a file named test_data  

```bash 
python power_law_fit.py test_data
```

To include uncertainty estimate 

```bash 
python power_law_fit.py --uncertainty test_data
```

To specify the domain

```bash 
python power_law_fit.py --min_value 2 --max_value 10 test_data
```

