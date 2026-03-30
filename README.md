# Linear Regression from Scratch 📈

A Python implementation of linear regression using **gradient descent**, built without any ML libraries. This project demonstrates how a model learns the best-fit line through iterative parameter updates.

---

## How It Works

The model fits a line `y = mx + b` to a dataset by minimizing **Mean Squared Error (MSE)** using gradient descent. At each step, it computes the partial derivatives of the error with respect to `b` and `m`, then nudges both parameters in the direction that reduces error.

---

## Project Structure

```
├── Lin_Reg_practice.py   # Core implementation
├── data.csv              # Training data (x, y columns)
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install numpy
```

### Running the Model

1. Make sure `data.csv` is in the same directory — it should have two columns: `x` and `y`.
2. Run the script:

```bash
python Lin_Reg_practice.py
```

### Example Output

```
Starting gradient descent at b = 0, m = 0, error = 5565.11
Ending at 1000 iterations: b = 0.089, m = 1.478, error = 112.65
```

---

## Configuration

You can tune the following hyperparameters in the `run()` function:

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.0001` | Step size for each gradient update |
| `initial_b` | `0` | Starting intercept value |
| `initial_m` | `0` | Starting slope value |
| `num_iterations` | `1000` | Number of gradient descent steps |

---

## Key Functions

- **`compute_error_for_line_given_points(b, m, points)`** — Calculates MSE for a given line
- **`step_gradient(b, m, points, learningRate)`** — Computes gradients and returns updated `b` and `m`
- **`gradient_descent_runner(...)`** — Runs the full training loop
- **`run()`** — Entry point; loads data and kicks off training

---

## Built With

- [NumPy](https://numpy.org/) — Array operations and CSV loading only; no ML libraries used
