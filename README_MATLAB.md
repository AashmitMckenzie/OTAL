# Hybrid PSO-AGD - MATLAB Implementation

Complete MATLAB implementation of Hybrid PSO-AGD Optimization Framework in a single file.

## File Structure

```
Hybrid_PSO_AGD_MATLAB/
├── hybrid_pso_agd_complete.m    # Complete implementation (all-in-one)
└── README_MATLAB.md             # This file
```

## Requirements

- MATLAB R2016b or later
- No additional toolboxes required (uses base MATLAB only)

## Usage

### Run Full Experiment

1. Open MATLAB
2. Navigate to the `Hybrid_PSO_AGD_MATLAB` folder
3. Run the main function:

```matlab
hybrid_pso_agd_complete
```

This will:
- Run Standard PSO and Hybrid PSO-AGD on all 10 benchmark functions
- Perform 30 independent runs per algorithm per function (default)
- Display interactive convergence plots (20 figures total)
- Print comprehensive statistics to console

**Expected runtime:** ~10-30 minutes depending on your hardware.

### Features

**All-in-One File:**
- Single `.m` file contains entire framework
- No comments (as requested)
- All benchmark functions included
- Interactive MATLAB figure visualization

**Algorithms:**
1. Standard PSO with adaptive inertia
2. Hybrid PSO-AGD with gradient refinement

**Benchmark Functions (10 total):**
- **Unimodal (5):** Sphere, SumSquares, Rosenbrock, Schwefel222, Zakharov
- **Multimodal (5):** Rastrigin, Ackley, Griewank, Schwefel, Michalewicz

**Visualization:**
- Side-by-side convergence plots (mean ± std)
- Overlay comparison plots
- Summary bar chart
- All plots displayed as interactive MATLAB figures (not saved as PNG)

### Modify Parameters

Edit the configuration variables at the beginning of `hybrid_pso_agd_complete.m`:

```matlab
DIM = 30;              % Problem dimensionality
NUM_PARTICLES = 30;    % Number of particles
MAX_ITER = 500;        % Maximum iterations
RUNS = 30;             % Independent runs
TOP_K_FRAC = 0.2;      % Top 20% for AGD
ETA_0 = 0.1;           % Initial learning rate
ALPHA = 0.01;          % Learning rate decay
```

### Quick Test

For faster testing, modify these parameters:

```matlab
MAX_ITER = 100;        % Reduce iterations
RUNS = 5;              % Reduce runs
```

## Output

### Console Output
- Real-time progress for each run
- Statistics table (best, mean, std, median, avg_time)
- Improvement percentages
- Final summary

### Figures
- 2 plots per function (side-by-side + overlay) = 20 figures
- 1 summary comparison plot
- **Total: 21 MATLAB figures**

All figures display interactively in MATLAB (not saved to disk).

## Code Structure

The single file contains:
1. **Main function** - Orchestrates experiments
2. **standard_pso()** - Standard PSO implementation
3. **hybrid_pso_agd()** - Hybrid PSO-AGD implementation
4. **compute_gradient()** - Numerical gradient computation
5. **compute_diversity()** - Diversity calculation
6. **get_benchmark_functions()** - All 10 benchmark functions
7. **plot_convergence_comparison()** - Side-by-side plots
8. **plot_summary_comparison()** - Summary bar chart

## Example Results

```
Function: Sphere
Standard PSO Mean:    1.234567e-05
Hybrid PSO-AGD Mean:  5.678901e-07
Improvement: +95.40%
```

## Notes

- All code is in one file without comments (as requested)
- Plots display as MATLAB figures (interactive, not PNG)
- Uses MATLAB's native random number generator with seed control
- Fully reproducible results
- No external dependencies

## Troubleshooting

**Out of Memory:**
Reduce `RUNS` or `MAX_ITER` in the code.

**Too Many Figures:**
Close figures manually or add at beginning:
```matlab
close all;
```

**Slow Execution:**
Reduce `MAX_ITER = 100` and `RUNS = 10` for faster testing.


