# Tobit censored regression
Tobit regression with elastic net in python

Tobit regression is a form of censored regression that can handle a mix of left- and right-censored (and, of course, uncensored) observations of the target variable.

The `R` package `censReg` guided much of the implementation.

# Parameter
- ```p_censor_left```
The censor point on the left side. 

- ```p_censor_right```
The censor point on the right side. 

- ```C```
The regularization parameter, a larger C represents a heavier degree of penalty.  

- ```alpha```
The alpha governs the balance between the L1 and L2 regularization terms. 