# Examples

Please follow the instructions at https://doi.org/10.7924/r48g8tf24 to download and place the relevant datasets in their respective locations to run the examples.

## Recording activity
Two examples are provided for recording membrane potential from fibers. `vrec_example.py` is or single-threaded execution:

```bash
> python vrec_example.py
```

`parallel_vrec_example.py` is designed for parallel execution using MPI:

```bash
> mpirun -n $N_CORES python parallel_vrec_example.py
```

❗ NOTE: if you're running these scripts headless, you'll want to remove any `matplotlib` visualization and save the output instead to visualize elsewhere.

## Calculating thresholds
An example threshold calculation script that can be executed in parallel is provided in `parallel_thresholds.py`:

```bash
> mpirun -n $N_CORES python parallel_thresholds.py
```

## Stimulus optimization using Differential Evolution
An example script for optimizing stimulus amplitudes (6 amplitudes for a 6-contact cuff delivering biphasic rectangular stimulation) is provided in `differential_evolution.py`. Sets of extracellular fields, targets, and weights are provided in `./fields/` (two human nerves, H1 and H2, and two pig nerves, P1 and P2). You can select for which nerve to optimize using the `--sample` command-line argument:

```bash
> mpirun -n $N_CORES python differential_evolution.py --sample H1
```

The full history, as well as the total execution time, of the optimization process will be saved to `./results` (the directory will be created if it does not exist), within the subdirectory `sample_$SAMPLE_$TIME`. This can be changed in the script.

> [!NOTE]
> Depending on your MPI installation, you may need to run with `mpiexec` instead of `mpirun`.