# Assignment 1 Instructions

To execute the provided Python script, simply execute `python assignment1.py`. The MSE mean and standard deviation will be printed once all k-fold splits have been trained and evaluated. You should receive a result similar to `MSE mean= 11.9941` and `MSE standard deviation = 3.9730`.

Some restrictions:
  * The script depends on the file `housing.csv` existing in the current working directory when the script is executed.
  * The script cleans up `housing.csv` prior to processing. Specifically, it converts space-based field separators to tabs. If you provide a different `housing.csv` than the one included in this submission, the behavior is undefined, and will likely result in script failure.
