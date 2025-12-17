"""
Sandbox wave equation V1D

Interactive 1D damped acoustic wave equation solver using finite differences.
Designed for educational exploration of seismic and sonic wave propagation,
attenuation (Q), layering, and detectability.

Author: maxseg2021
License: MIT
"""


import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def ricker(t, f0, t0):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gaussian_pulse(t, sigma, t0):
    sigma = max(sigma, 1e-9)
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def vibroseis_chirp(t, f1, f2, T):
    k = (f2 - f1) / max(T, 1e-9)
    w = np.zeros_like(t)
    mask = (t >= 0.0) & (t <= T)
    tm = t[mask]
    phase = 2.0 * np.pi * (f1 * tm + 0.5 * k * tm * tm)
    w[mask] = np.sin(phase)
    return w


def make_source_wavelet(t, source_type, f0, sigma, f1, f2, sweep_T, t0):
    if source_type == "Ricker":
        return ricker(t, f0, t0)
    if source_type == "Gaussian":
        return gaussian_pulse(t, sigma, t0)
    if source_type == "Vibroseis":
        return vibroseis_chirp(t, f1, f2, sweep_T)
    return ricker(t, f0, t0)


def estimate_fdom(source_type, f0, sigma, f1, f2):
    if source_type == "Ricker":
        return max(0.5, float(f0))
    if source_type == "Gaussian":
        return max(0.5, float(1.0 / (2.0 * np.pi * max(sigma, 1e-9))))
    if source_type == "Vibroseis":
        return max(0.5, float(0.5 * (f1 + f2)))
    return max(0.5, float(f0))


def build_velocity_model(x, c0, two_layer, c2, interface_x):
    c = c0 * np.ones_like(x)
    if two_layer:
        c[x >= interface_x] = c2
    return c


def build_sponge(nx, width, strength):
    s = np.ones(nx, dtype=float)
    if strength <= 0.0 or width <= 0:
        return s
    width = int(min(width, nx // 2))
    i = np.arange(width, dtype=float)
    taper = np.exp(-(strength**2) * ((width - i) / width) ** 2)
    s[:width] *= taper
    s[-width:] *= taper[::-1]
    return s


def kg_to_relative_amp(mass_kg, ref_kg=1.0):
    mass_kg = max(float(mass_kg), 0.0)
    ref_kg = max(float(ref_kg), 1e-12)
    return np.sqrt(mass_kg / ref_kg)


def kg_to_db(mass_kg, ref_kg=1.0):
    mass_kg = max(float(mass_kg), 1e-12)
    ref_kg = max(float(ref_kg), 1e-12)
    return 10.0 * np.log10(mass_kg / ref_kg)


def spreading_factor(r, mode):
    r = float(abs(r))
    if mode == "1D":
        return 1.0
    if mode == "2D":
        return 1.0 / np.sqrt(r + 1.0)
    if mode == "3D":
        return 1.0 / (r + 1.0)
    return 1.0


def pick_hit_events(t, trace, thr, hold=6, min_separation_s=0.06):
    """
    Event picker: record a "hit" when |trace| crosses threshold upward
    and stays above for 'hold' samples. Returns arrays (thit, ahit),
    where ahit is the signed trace amplitude at the hit time.
    """
    thr = float(thr)
    if thr <= 0.0:
        return np.array([]), np.array([])

    a = np.abs(trace)
    above = a >= thr
    n = len(a)
    hits_t = []
    hits_a = []

    last_t = -1e99
    i = 1
    while i < n:
        if above[i] and (not above[i - 1]):
            ok = 0
            j = i
            while j < n and ok < hold:
                if above[j]:
                    ok += 1
                else:
                    break
                j += 1
            if ok >= hold:
                th = float(t[i])
                if th - last_t >= float(min_separation_s):
                    hits_t.append(th)
                    hits_a.append(float(trace[i]))
                    last_t = th
                i = j
                continue
        i += 1

    return np.array(hits_t, dtype=float), np.array(hits_a, dtype=float)


def simulate_damped_wave(
    nx, L, nt,
    c0, two_layer, c2, interface_x,
    source_type, f0, sigma, f1, f2, sweep_T,
    sponge_strength,
    src_x, rec_x,
    source_mass_kg,
    Q,
    spread_mode,
    noise_rms,
    record_every
):
    """
    Damped 1D wave equation:
        u_tt + 2*gamma*u_t = c(x)^2 u_xx + w(t)*delta(x-xs)

    gamma tied to Q:
        gamma = pi * fdom / Q
    """
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]

    c = build_velocity_model(x, c0, two_layer, c2, interface_x)
    dt = 0.9 * dx / np.max(c)

    src_i = int(np.clip(round(src_x / dx), 1, nx - 2))
    rec_i = int(np.clip(round(rec_x / dx), 1, nx - 2))

    sponge = build_sponge(nx, width=60, strength=sponge_strength)

    t = np.arange(nt) * dt
    t0 = 0.12
    w = make_source_wavelet(t, source_type, f0, sigma, f1, f2, sweep_T, t0)

    amp_scale = kg_to_relative_amp(source_mass_kg, ref_kg=1.0)
    w = amp_scale * w

    fdom = estimate_fdom(source_type, f0, sigma, f1, f2)
    Q = max(float(Q), 1e-6)
    gamma = np.pi * fdom / Q

    u_nm1 = np.zeros(nx, dtype=float)
    u_n = np.zeros(nx, dtype=float)
    u_np1 = np.zeros(nx, dtype=float)

    rec = np.zeros(nt, dtype=float)

    t_hist = []
    u_hist = []

    a = (2.0 - 2.0 * gamma * dt)
    b = (1.0 - 2.0 * gamma * dt)

    for n in range(nt):
        u_xx = (u_n[2:] - 2.0 * u_n[1:-1] + u_n[:-2]) / (dx * dx)

        u_np1[1:-1] = (
            a * u_n[1:-1]
            - b * u_nm1[1:-1]
            + (c[1:-1] ** 2) * (dt * dt) * u_xx
        )

        u_np1[src_i] += w[n]
        u_np1 *= sponge

        rec[n] = u_np1[rec_i]

        if (n % record_every) == 0 or n == nt - 1:
            t_hist.append(t[n])
            u_hist.append(u_np1.copy())

        u_nm1, u_n, u_np1 = u_n, u_np1, u_nm1
        u_np1.fill(0.0)

    # Apply geometric spreading and receiver noise once, after the wavefield is computed
    r = abs(float(rec_x - src_x))
    rec *= spreading_factor(r, spread_mode)

    if noise_rms > 0.0:
        rng = np.random.default_rng()
        rec += float(noise_rms) * rng.standard_normal(nt)

    return x, c, dt, t, w, rec, np.array(t_hist), np.array(u_hist), gamma, fdom


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wave equation sandbox: source, receiver, attenuation")
        self.geometry("1600x900")
        self.var_noise = tk.DoubleVar(value=0.002)

        self.nx = 900
        self.L = 6000.0
        self.nt = 2600
        self.record_every = 4

        self.var_two = tk.BooleanVar(value=True)
        self.var_c0 = tk.DoubleVar(value=2000.0)
        self.var_c2 = tk.DoubleVar(value=2800.0)
        self.var_int_x = tk.DoubleVar(value=3300.0)

        self.var_src_x = tk.DoubleVar(value=0.0)
        self.var_rec_x = tk.DoubleVar(value=1250.0)

        self.var_spg = tk.DoubleVar(value=0.02)

        self.source_type = tk.StringVar(value="Ricker")
        self.var_f0 = tk.DoubleVar(value=15.0)
        self.var_sigma = tk.DoubleVar(value=0.02)
        self.var_f1 = tk.DoubleVar(value=8.0)
        self.var_f2 = tk.DoubleVar(value=55.0)
        self.var_sweepT = tk.DoubleVar(value=1.5)

        self.var_mass = tk.DoubleVar(value=10.0)
        self.var_Q = tk.DoubleVar(value=120.0)
        self.spread_mode = tk.StringVar(value="2D")

        self.var_speed = tk.IntVar(value=5)

        self.x = None
        self.c = None
        self.dt = None
        self.t_full = None
        self.w_full = None
        self.rec_full = None
        self.t_hist = None
        self.u_hist = None

        self.wave_ylim = 1.0
        self.rec_ylim = 1.0
        self.src_ylim = 1.0

        self.hit_t = np.array([])
        self.hit_a = np.array([])
        self.hit_thr = 0.0

        self.frame_idx = 0
        self.playing = False
        self._job = None
        self._debounce_job = None

        self._build_ui()
        self._rerun()

    def _build_ui(self):
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self._add_slider(controls, "source loc x (m)", self.var_src_x, 0.0, self.L, self._rerun, width=230, fmt="{:.0f}")
        self._add_slider(controls, "receiver loc x (m)", self.var_rec_x, 0.0, self.L, self._rerun, width=230, fmt="{:.0f}")
        self._add_slider(controls, "sponge", self.var_spg, 0.0, 0.08, self._rerun, width=170)

        row2 = ttk.Frame(self)
        row2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        ttk.Checkbutton(row2, text="two layer", variable=self.var_two, command=self._rerun).pack(side=tk.LEFT, padx=8)
        self._add_slider(row2, "layer 1 velocity, c1 (m/s)", self.var_c0, 800.0, 6000.0, self._rerun, width=190)
        self._add_slider(row2, "layer 2 velocity, c2 (m/s)", self.var_c2, 800.0, 6500.0, self._rerun, width=190)
        self._add_slider(row2, "interface loc x (m)", self.var_int_x, 0.0, self.L, self._rerun, width=230)

        row3 = ttk.Frame(self)
        row3.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        ttk.Label(row3, text="source type").pack(side=tk.LEFT, padx=6)
        combo = ttk.Combobox(
            row3,
            textvariable=self.source_type,
            values=["Ricker", "Gaussian", "Vibroseis"],
            width=12,
            state="readonly",
        )
        combo.pack(side=tk.LEFT, padx=6)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._rerun())

        self._add_slider(row3, "f0 (freq. of source)", self.var_f0, 2.0, 40.0, self._rerun, width=150)
        self._add_slider(row3, "sigma (for Gausian Pulse)", self.var_sigma, 0.005, 0.08, self._rerun, width=150)
        self._add_slider(row3, "f1 (min freq. Vibroseis)", self.var_f1, 2.0, 30.0, self._rerun, width=150)
        self._add_slider(row3, "f2 (max. freq. Vibroseis)", self.var_f2, 20.0, 120.0, self._rerun, width=150)
        self._add_slider(row3, "T (Vibroseis chirp duration)", self.var_sweepT, 0.5, 4.0, self._rerun, width=150)

        row4 = ttk.Frame(self)
        row4.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        self._add_slider(row4, "source energy (kg TNT eq)", self.var_mass,
                         0.000001, 50.0, self._rerun, width=240)
        self._add_slider(row4, "Q (attenuation control)", self.var_Q, 10.0, 2000.0, self._rerun, width=180)

        # receiver noise floor
        self._add_slider(row4, "noise RMS", self.var_noise, 0.0, 0.02, self._rerun, width=180)

        ttk.Label(row4, text="spreading").pack(side=tk.LEFT, padx=6)
        spread_combo = ttk.Combobox(
            row4,
            textvariable=self.spread_mode,
            values=["1D", "2D", "3D"],
            width=5,
            state="readonly",
        )
        spread_combo.pack(side=tk.LEFT, padx=6)
        spread_combo.bind("<<ComboboxSelected>>", lambda _e: self._rerun())

        # (removed "pick frac" slider: detection threshold is tied automatically to noise)

        speed_frame = ttk.Frame(row4)
        speed_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(speed_frame, text="speed").pack(anchor="w")
        sp = ttk.Spinbox(speed_frame, from_=1, to=20, textvariable=self.var_speed, width=4)
        sp.pack(anchor="w")

        row_time = ttk.Frame(self)
        row_time.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(row_time, text="play", command=self.play).pack(side=tk.LEFT, padx=6)
        ttk.Button(row_time, text="pause", command=self.pause).pack(side=tk.LEFT, padx=6)
        ttk.Button(row_time, text="rerun", command=self._rerun).pack(side=tk.LEFT, padx=10)

        ttk.Label(row_time, text="time (ms)").pack(side=tk.LEFT, padx=10)
        self.time_ms = tk.DoubleVar(value=0.0)
        self.time_slider = ttk.Scale(
            row_time,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.time_ms,
            length=760,
            command=lambda _v: self._on_time_slider(),
        )
        self.time_slider.pack(side=tk.LEFT, padx=10)

        self.time_label = ttk.Label(row_time, text="0.0 ms")
        self.time_label.pack(side=tk.LEFT, padx=10)

        self.info_label = ttk.Label(row_time, text="")
        self.info_label.pack(side=tk.LEFT, padx=10)

        self.fig = plt.Figure(figsize=(15.4, 7.8), dpi=100)
        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[3.0, 1.5],
            height_ratios=[1.0, 1.0],
            wspace=0.28,
            hspace=0.32,
        )

        self.ax_wave = self.fig.add_subplot(gs[0, 0])
        self.ax_rec = self.fig.add_subplot(gs[1, 0])
        self.ax_model = self.fig.add_subplot(gs[0, 1])
        self.ax_src = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _add_slider(self, parent, label, var, vmin, vmax, callback, width=160, fmt="{:.1f}"):
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=8, pady=2)

        ttk.Label(frame, text=label).pack(anchor="w")

        s = ttk.Scale(
            frame,
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            variable=var,
            length=width,
            command=lambda _v: self._debounced(callback),
        )
        s.pack()

        val_label = ttk.Label(frame, width=12)
        val_label.pack(anchor="w")

        def update_label(*_):
            val_label.config(text=fmt.format(var.get()))

        var.trace_add("write", update_label)
        update_label()

    def update_label(*_):
        val_label.config(text=fmt.format(var.get()))

    var.trace_add("write", update_label)
    update_label()
    
    def _debounced(self, fn):
        if self._debounce_job is not None:
            self.after_cancel(self._debounce_job)
        self._debounce_job = self.after(140, fn)

    def _rerun(self):
        self.pause()

        two_layer = bool(self.var_two.get())
        c1 = float(self.var_c0.get())
        c2 = float(self.var_c2.get())

        interface_x = float(np.clip(self.var_int_x.get(), 0.0, self.L))
        src_x = float(round(self.var_src_x.get()))
        rec_x = float(round(self.var_rec_x.get()))

        sponge = float(self.var_spg.get())

        stype = self.source_type.get()
        f0 = float(self.var_f0.get())
        sigma = float(self.var_sigma.get())
        f1 = float(self.var_f1.get())
        f2 = float(self.var_f2.get())
        sweepT = float(self.var_sweepT.get())

        mass_kg = float(self.var_mass.get())
        Q = float(self.var_Q.get())
        spread_mode = self.spread_mode.get()
        noise_rms = float(self.var_noise.get())

        x, c, dt, t, w, rec, t_hist, u_hist, gamma, fdom = simulate_damped_wave(
            nx=self.nx, L=self.L, nt=self.nt,
            c0=c1, two_layer=two_layer, c2=c2, interface_x=interface_x,
            source_type=stype, f0=f0, sigma=sigma, f1=f1, f2=f2, sweep_T=sweepT,
            sponge_strength=sponge,
            src_x=src_x, rec_x=rec_x,
            source_mass_kg=mass_kg,
            Q=Q,
            spread_mode=spread_mode,
            noise_rms=noise_rms,
            record_every=self.record_every
        )

        self.x, self.c, self.dt = x, c, dt
        self.t_full, self.w_full, self.rec_full = t, w, rec
        self.t_hist, self.u_hist = t_hist, u_hist

        self._set_fixed_amplitude_limits_using_early_time()

        # Automatic detection threshold
        if noise_rms > 0.0:
            self.hit_thr = 5.0 * noise_rms  # 5 sigma detection
        else:
            self.hit_thr = 0.03 * max(1e-12, float(np.max(np.abs(self.rec_full))))
        self.hit_t, self.hit_a = pick_hit_events(self.t_full, self.rec_full, self.hit_thr)

        self.frame_idx = 0
        self._update_time_slider_limits()

        self._draw_model(src_x, rec_x, interface_x, two_layer)
        self._draw_source()
        start_idx = 1 if (self.t_hist is not None and len(self.t_hist) > 1) else 0
        self._draw_wavefield(start_idx, src_x, rec_x, interface_x, two_layer)
        
        # keep the time slider consistent with what you drew
        self.time_ms.set(float(self.t_hist[start_idx] * 1000.0))

        db = kg_to_db(mass_kg, ref_kg=1.0) if mass_kg > 0 else -np.inf
        offset = abs(rec_x - src_x)
        self.info_label.configure(
            text=f"mass {mass_kg:.1f} kg ({db:.1f} dB re 1 kg)   Q {Q:.1f}   fdom {fdom:.1f} Hz   gamma {gamma:.4f} 1/s   offset {offset:.1f} m"
        )

        self.canvas.draw_idle()

    def _set_fixed_amplitude_limits_using_early_time(self):
        if self.u_hist is None or len(self.u_hist) == 0:
            self.wave_ylim = 1.0
        else:
            k = max(3, int(0.08 * len(self.u_hist)))
            early = self.u_hist[:k]
            umax0 = float(np.max(np.abs(early)))
            if umax0 < 1e-12:
                umax0 = float(np.max(np.abs(self.u_hist)))
            self.wave_ylim = max(1e-9, 1.35 * umax0)

        if self.rec_full is None or len(self.rec_full) == 0:
            self.rec_ylim = 1.0
        else:
            rmax = float(np.max(np.abs(self.rec_full)))
            self.rec_ylim = max(1e-9, 1.25 * rmax)

        if self.w_full is None or len(self.w_full) == 0:
            self.src_ylim = 1.0
        else:
            wmax = float(np.max(np.abs(self.w_full)))
            self.src_ylim = max(1e-9, 1.25 * wmax)

    def _update_time_slider_limits(self):
        if self.t_hist is None or len(self.t_hist) == 0:
            self.time_slider.configure(from_=0.0, to=1.0)
            self.time_ms.set(0.0)
            self.time_label.configure(text="0.0 ms")
            return
        tmax_ms = float(self.t_hist[-1] * 1000.0)
        self.time_slider.configure(from_=0.0, to=tmax_ms)
        self.time_ms.set(0.0)
        self.time_label.configure(text="0.0 ms")

    def _draw_model(self, src_x, rec_x, interface_x, two_layer):
        self.ax_model.clear()
        self.ax_model.set_title("velocity model and geometry", fontsize=10, loc="left")
        self.ax_model.set_xlabel("x (m)")
        self.ax_model.set_ylabel("velocity (m/s)")
        self.ax_model.grid(True, alpha=0.2)
        self.ax_model.set_xlim(0.0, self.L)

        if two_layer:
            self.ax_model.axvspan(0.0, interface_x, alpha=0.20, color="#4C78A8")
            self.ax_model.axvspan(interface_x, self.L, alpha=0.20, color="#F58518")
            self.ax_model.axvline(interface_x, linestyle=":", linewidth=2.0, color="#111111")
        else:
            self.ax_model.axvspan(0.0, self.L, alpha=0.16, color="#4C78A8")

        self.ax_model.plot(self.x, self.c, linewidth=3.0, color="#111111")
        self.ax_model.axvline(src_x, linestyle="--", linewidth=1.2, color="#111111")
        self.ax_model.axvline(rec_x, linestyle="--", linewidth=1.2, color="#111111")

        cmin, cmax = float(np.min(self.c)), float(np.max(self.c))
        pad = 0.10 * (cmax - cmin + 1e-9)
        self.ax_model.set_ylim(cmin - pad, cmax + pad)

        self.ax_model.text(src_x, cmax + 0.02 * (cmax - cmin + 1e-9), "source", ha="center", va="bottom")
        self.ax_model.text(rec_x, cmax + 0.02 * (cmax - cmin + 1e-9), "receiver", ha="center", va="bottom")

    def _draw_source(self):
        self.ax_src.clear()
        self.ax_src.set_title("source wavelet w(t)", fontsize=10, loc="left")
        self.ax_src.set_xlabel("time (ms)", fontsize=10)
        self.ax_src.set_ylabel("relative amplitude", fontsize=10)
        self.ax_src.grid(True, alpha=0.2)

        t_ms = self.t_full * 1000.0
        self.ax_src.plot(t_ms, self.w_full, linewidth=2.0)

        tmax_ms = min(float(t_ms[-1]), 2000.0)
        self.ax_src.set_xlim(0.0, tmax_ms)
        self.ax_src.set_ylim(-self.src_ylim, self.src_ylim)

    def _draw_receiver_hits(self, current_t):
        self.ax_rec.clear()
        self.ax_rec.set_title("receiver trace", fontsize=10, loc="left")
        self.ax_rec.set_xlabel("time (ms)", fontsize=10)
        self.ax_rec.set_ylabel("relative amplitude", fontsize=10)
        self.ax_rec.grid(True, alpha=0.2)
    
        rec_plot = self.rec_full.copy()
    
        i_now = int(np.searchsorted(self.t_full, float(current_t), side="right"))
        rec_plot[i_now:] = np.nan
    
        self.ax_rec.plot(self.t_full * 1000.0, rec_plot, linewidth=1.6)
        self.ax_rec.set_xlim(0.0, float(self.t_full[-1] * 1000.0))
        self.ax_rec.set_ylim(-self.rec_ylim, self.rec_ylim)
    def _draw_wavefield(self, idx, src_x, rec_x, interface_x, two_layer):
        idx = int(np.clip(idx, 0, len(self.t_hist) - 1))
        self.frame_idx = idx

        self.ax_wave.clear()
        self.ax_wave.set_title("wavefield u(x,t)", fontsize=10, loc="left")
        self.ax_wave.set_xlabel("x (m)")
        self.ax_wave.set_ylabel("relative amplitude")
        self.ax_wave.grid(True, alpha=0.2)

        self.ax_wave.set_xlim(0.0, self.L)
        self.ax_wave.set_ylim(-self.wave_ylim, self.wave_ylim)

        if two_layer:
            self.ax_wave.axvspan(0.0, interface_x, alpha=0.12, color="#4C78A8")
            self.ax_wave.axvspan(interface_x, self.L, alpha=0.12, color="#F58518")
            self.ax_wave.axvline(interface_x, linestyle=":", linewidth=2.0, color="#111111")

        u = self.u_hist[idx]

        c_ref = float(self.var_c0.get())
        norm = np.sqrt(np.clip(self.c / max(c_ref, 1e-9), 1e-9, 1e9))
        u_disp = u / norm

        self.ax_wave.plot(self.x, u_disp, linewidth=2.0, color="#111111")

        self.ax_wave.axvline(src_x, linestyle="--", linewidth=1.2, color="#111111")
        self.ax_wave.axvline(rec_x, linestyle="--", linewidth=1.2, color="#111111")

        self.ax_wave.text(src_x, 0.93 * self.wave_ylim, "source", ha="center", va="top")
        self.ax_wave.text(rec_x, 0.93 * self.wave_ylim, "receiver", ha="center", va="top")

        t_now = float(self.t_hist[idx])
        self.time_label.configure(text=f"{t_now * 1000.0:.1f} ms")

        self._draw_receiver_hits(current_t=t_now)

    def _on_time_slider(self):
        if self.t_hist is None or len(self.t_hist) == 0:
            return
        t_target = float(self.time_ms.get()) / 1000.0
        idx = int(np.argmin(np.abs(self.t_hist - t_target)))

        src_x = float(np.clip(self.var_src_x.get(), 0.0, self.L))
        rec_x = float(np.clip(self.var_rec_x.get(), 0.0, self.L))
        interface_x = float(np.clip(self.var_int_x.get(), 0.0, self.L))
        two_layer = bool(self.var_two.get())

        self._draw_wavefield(idx, src_x, rec_x, interface_x, two_layer)
        self.canvas.draw_idle()

    def play(self):
        if self.t_hist is None or len(self.t_hist) == 0:
            return
        self.playing = True
        self._tick()

    def pause(self):
        self.playing = False
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _tick(self):
        if not self.playing:
            return

        step = int(max(1, self.var_speed.get()))
        next_idx = self.frame_idx + step
        if next_idx >= len(self.t_hist):
            next_idx = 0

        self.time_ms.set(float(self.t_hist[next_idx] * 1000.0))
        self._on_time_slider()
        self._job = self.after(16, self._tick)


if __name__ == "__main__":
    App().mainloop()
