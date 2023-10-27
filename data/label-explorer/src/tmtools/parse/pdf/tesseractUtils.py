
import sys
import os

# for data
import pandas as pd
import numpy as np

# for pdf parsing
import pytesseract



#**********************************************************
#*                      functions                         *
#**********************************************************

# ------------------------ Parsing ------------------------
# see https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract

def parse_image_tesseract(image, *args, **kwargs):
    d = pytesseract.image_to_data(image, *args, **kwargs)
    text = d['text']
    span_x0 = d['left']
    span_y0 = d['top']
    span_x1 = [x0 + w for x0, w in zip(d['left'], d['width'])]
    span_y1 = [y0 + h for y0, h in zip(d['top'], d['height'])]
    
    df_text = pd.DataFrame({
        'text' : text,
        'span_x0': span_x0,
        'span_y0': span_y0,
        'span_x1': span_x1,
        'span_y1': span_y1,
    })
    return df_text

