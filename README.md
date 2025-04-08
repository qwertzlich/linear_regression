# Linear Regression Project
## Table of Contents

- [Introduction](#introduction)
- [`linreg_num()`](#linreg_num)
    - [Usage](#usage)
- [`linreg_grad`](#linreg_grad)
    - [Usage](#usage-1)
- [📄 License](#-license)
- [📧 Contact](#-contact)

## Introduction

Welcome to the **Linear Regression** project! This repository demonstrates the implementation of a simple linear regression model to analyze and predict data trends.

---

## `linreg_num()`

This method performs linear regression by solving the _normal equation_ with qr decomposition. This is a numerical approach and was implemented to compare runtime for increasing amounts of data.

### Usage

**Command Line**: Use `linreg_num -h` for an overview of conditional and positional arguments. From cli you need to give an input file containing the data. The results will be printed to console. As of now (March 2025) the results do not include error ranges.

**Imported as module**: When imported as a method the data will be given as an Array.

---

## `linreg_grad`

This method performs linear regression by utilizing gradient decent. This is an iterative approach which can be modified to save computation effort with BFGS or L-BFGS which are not included in this project as of now (March 2025).

### Usage

**Command Line**: Use `linreg_grad -h` for an overview of conditional and positional arguments. From cli you need to give an input file containg the data. The results will be printed to console. As of now (March 2025) the results do not include error ranges.

**Imported as module**: When imported as a method the data will be given as an Array. Additional arguments are described in the functions doc string.

---

## File format

The file format should include the data as XY-Format seperated with commas (csv). The first row will always be interpreted as column-names.

### File Example

Below is an example of the expected file format:

```
x,y
1,2
2,4
3,6
4,8
5,10
```

Ensure the file is saved with a `.csv` extension and follows the structure shown above.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback, reach out at: **laurenz.fraas@icloud.com**

Happy coding! 😊
