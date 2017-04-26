# TL:DR
Improved implementation of [Tariq Rashid's "Make Your Own Neural Network"](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608). Cleaned up code. MNIST data is retrieved automatically for you. Program traverses over range of network parameters to search for model with highest accuracy.

# WHY?
Tariq Rashid provided an [excellent introduction to neural networks](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608). Yet, the provided code is not clean - a disgrace to the beauty of python. (This is due to the book written for people with zero coding experience.) I therefore could not resist the urge to tidy up the project.

# WHAT ELSE?
The current most accurate model will be saved to a database file. At this version the model is just stored. Feel free to find your own uses for it (like nominating the winner model at the end of all training runs). The debug mode can be activated by setting the DEBUG variable to True at the beginning of the code. In debug mode only a single training run with a smaller sample size is conducted. Additionally the finished model will be backqueried to present you the perception pattern for each digit.

# HOW?
1. Clone repo:

``` git clone git@github.com:bfelder/MYONN myonn/ ```

2. Go to folder:

```cd myonn/myonn/```

3. Install requirements:

```pip install -r requirements.txt``` 

4. Start program:

```python3 myonn.py```

5. On first run MNIST data is retrieved automatically for you. :-)

# STARS ✰✰✰✰✰  
If you liked this repo, please star it. Because other users will more easily find it in the search results then. Thank you!