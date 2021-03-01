# OCR_project
 Three approaches to do OCR tasks.



## Overview

Introduce three approaches (Google tesseract, Baidu API, and TR package)  to do OCR tasks.



## Features

- Develop some algorithms to help split pictures and improve OCR accuracy.
- Develop some rules (eg. re expression) to help optimize the OCR results.



## About the Split algorithm

1. Do binarization.
2. Scan every pixel in the picture by one direction (by row here) first and calculate the frequency of change.
3. Based on the frequency of change, we can divide the picture into blocks by one direction (by row here).
4. Within each block, we can do step2 & 3 again (by columns here) and get the sub-blocks by columns within the blocks by rows.
5. Here we got the picture divided into blocks and each block contains a complete text.