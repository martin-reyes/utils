# utils

The functions in these python files are general tasks commonly run throughout the DS pipeline. 

Also included in this directory are templates for a final report .ipynb, README.md, and work/test notebooks for each part of the DS pipeline,

To import a python module in this folder in any python file or notebook (if `utils` directory is in the home directory):

```
import sys
import os

home_directory_path = os.path.expanduser('~')
sys.path.append(home_directory_path +'/utils')
```

then import the desired python modules.