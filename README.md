# Sandbox wave equation V1D

Interactive 1D damped acoustic wave equation sandbox for geophysical education.

This project implements the 1D damped acoustic wave equation using finite
differences, with explicit control of:

- Velocity layering
- Attenuation (Q)
- Source wavelet type and energy
- Receiver detectability and noise

The sandbox is designed as a conceptual analogue to a zero-offset sonic or
vertical seismic experiment, emphasizing physical intuition over realism.

## Governing equation

∂²u(x,t)/∂t² + 2γ ∂u(x,t)/∂t = c(x)² ∂²u(x,t)/∂x² + s(t) δ(x − xs)

where γ = π f₀ / Q.

## Files

- wave_sandbox_v1d.py — interactive Tk-based simulator

## License

MIT License  
Author: maxseg2021
