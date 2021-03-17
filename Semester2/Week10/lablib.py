from IPython.display import Markdown
from typing import Iterable, Tuple, Union, Any
import numpy as np
import pandas as pd
# Define our rules...

def rule_1(c, error_val):
    return abs(c) * error_val

# res = c * val ^ power, where error in value is error_val....
def rule_2(c, val, error_val, power):
    return abs(c * power * val ** (power - 1)) * error_val

def rule_3(*err_vals: Iterable[float]) -> float:
    """
    Calculate rule 3 from the paper 'Treatment of Data'.
    
    @param err_vals: A list of parameters, being the errors in each value. It is assumed they were summed 
                     together to get the final value.
    
    @returns: The error of all of the sum of the values...
    """
    total = 0
    
    for err in err_vals:
        total += err ** 2
    
    return np.sqrt(total)


def rule_4(value: float, *error_list: Iterable[Tuple[float, float, float]]) -> float:
    """
    Calculate rule 4 from the paper 'Treatment of Data'.
    
    @param value: The value of the thing we are trying to calculate the error of.
    @param error_list: A list of length 3 tuples. 
                       Each tuple should contain:
                       - A float: A value in the error formula.
                       - A float: The measured error in the above value.
                       - A float: The power of the above value in the multiplicative formula.

    @returns: A float, being the error in 'value'.
    """
    total = 0
    
    for x, x_err, power in error_list:
        total += (power * (x_err / x)) ** 2
        
    return abs(value) * np.sqrt(total)

FloatVec = Union[float, np.ndarray] 
BoolVec = Union[bool, np.ndarray]
OpVec = Union[Any, Iterable]
StrVec = Union[str, Iterable[str]]

def values_agree(val_1: FloatVec, err_1: FloatVec, val_2: FloatVec, err_2: FloatVec) -> BoolVec:
    """
    Determines if the values in 2 vectors(arrays) agree with each other, given there uncertainty values. 
    
    @param val_1: The 1st array of values.
    @param err_1: The uncertainty values for the 1st vector.
    @param val_2: The 2nd array of values.
    @param err_2: The uncertainty values for the 2st vector. 
    
    @returns: A vector of booleans, being whether each value agrees with the other.
    """
    # Grab the ranges for each value...
    r11, r12 = val_1 - err_1, val_1 + err_1
    r21, r22 = val_2 - err_2, val_2 + err_2
    
    # The ranges are sorted (r21 <= r22 and r11 <= r12), so the simple 2 checks below are enough. 
    # Check 1: Does r21(lowest value of 2nd range) land above the 1st range? If so fail...
    # Check 2: Does r22(highest value of 2nd range) fall below the 1st range? If so fail...
    return ((r21 <= r12) & (r11 <= r22))

# Some extra stuff for pretty printing measurements....
from IPython.display import Markdown

def format_result(msgs: OpVec, value: OpVec, value_error: OpVec, units: OpVec, past_dec: OpVec = 2) -> StrVec:
    all_arrs = [msgs, value, value_error, units, past_dec]
    
    def to_vec(val, types): 
        return [val] if(isinstance(val, types)) else list(val)
    
    (msgs, value, value_error, units, past_dec) = all_arrs = [
        to_vec(v, (float, int, str)) for v in all_arrs
    ]
    max_len = max(len(arr) for arr in all_arrs)
    
    msgs, value, value_error, units, past_dec = all_arrs = [
        (v * max_len if(len(v) == 1) else v) for v in all_arrs
    ]
    
    # Nesting variable in the formating of variables, not confusing at all....
    return [
        fr"{msg} $ {v:.0{pd}f} \pm {v_err:.0{pd}f} \: {u} $"
        for msg, v, v_err, pd, u in zip(msgs, value, value_error, past_dec, units)
    ]

def display_result(*args, **kwargs):
    for res in format_result(*args, **kwargs):
        display(Markdown(res))

# Print function for markdown :)...
def mkdwn(string):
    display(Markdown(string))
    
def rmse(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2) ** 2) / len(arr1))

