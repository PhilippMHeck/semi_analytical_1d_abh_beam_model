import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from scipy.optimize import least_squares
from numpy import sin, sinh, cos, cosh
from scipy.signal import find_peaks
from scipy import linalg
import cmath
import scipy.integrate as integrat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array
from scipy.sparse import csr_matrix
########################################################################################################################
class ASL_TS_FEM():

    def E_log(self, f, a, b):
        return a * np.log10(f) + b

    def Eta_pot(self, f, a, b, c):
        return a * f ** b + c

    def h_ASL(self, x, h_min, h_max, l_asl, p_h):
        return h_min + ((h_max - h_min) / (l_asl ** p_h)) * x ** p_h

    def E_ASL(self, x, E_min, E_max, l_asl, p_E):
        return E_min + ((E_max - E_min) / (l_asl ** p_E)) * x ** p_E

    def reverse_h(sef, h_vec, h, h_min, p, L_ASL):
        return (2 * h_vec * L_ASL ** p / (h - h_min)) ** (1 / (p))

    def E_eff(self, E1, eta1, E2, eta2, h_1, h_2, h_12):
        E_1 = E1 * (1. + eta1 * 1j)
        E_2 = E2 * (1. + eta2 * 1j)
        E_eff_compl = (2 / 3 * E_2 * (
                    3 / 4 * h_1 ** 2 * h_2 + 3 / 2 * h_1 * h_2 ** 2 + h_2 ** 3) + 1 / 12 * E_1 * h_1 ** 3) * 12 / (
                                  h_12 ** 3)
        return E_eff_compl

    def _build_element(self, h, E, rho_eff, b, nu, Omega):
        """
        Build the dict of derived quantities for a single beam element.

        All array-valued fields have shape (n_freq,) except `zeta` and `psi`,
        which have shape (4, n_freq) — the four roots / mode shapes of the
        Timoshenko characteristic equation.
        """
        elem = {
            'h':     h,
            'E':     E,
            'rho':   rho_eff,
            'I':     b * h ** 3 / 12,
            'A':     b * h,
            'kappa': 10 * (1 + nu) / (12 + 11 * nu),
        }
        elem['k']     = np.sqrt(elem['I'] / elem['A'])
        elem['G']     = elem['E'] / (2 * (1 + nu))
        elem['lam']   = (Omega ** 2 * elem['rho'] / (elem['E'] * elem['k'] ** 2)) ** 0.25
        elem['alpha'] = 0.5 * elem['k'] ** 2 * elem['lam'] ** 2
        elem['mu']    = elem['E'] / (elem['G'] * elem['kappa'])

        # Roots of the characteristic equation. Pairs (1, 2) and (3, 4) are
        # negatives of each other, so we compute each square root once.
        disc_sqrt = np.sqrt(elem['alpha'] ** 2 * (1 - elem['mu']) ** 2 + 1)
        base      = -elem['alpha'] * (1 + elem['mu'])
        root_p    = np.sqrt(base + disc_sqrt)   # |zeta1| = |zeta2|
        root_m    = np.sqrt(base - disc_sqrt)   # |zeta3| = |zeta4|
        elem['zeta'] = np.array([
             elem['lam'] * root_p,
            -elem['lam'] * root_p,
             elem['lam'] * root_m,
            -elem['lam'] * root_m,
        ], dtype=complex)                       # shape (4, n_freq)

        factor       = Omega ** 2 * elem['rho'] / (elem['kappa'] * elem['G'])
        elem['psi']  = -1 / elem['zeta'] * (elem['zeta'] ** 2 + factor)  # (4, n_freq)
        return elem

    def disc_Beam_TS_DL(self, n, n_x_disc, f_vec, l, l_asl, p_h,
                        E_a, E_b, Eta_a, Eta_b,
                        E_a_DL, E_b_DL, Eta_a_DL, Eta_b_DL,
                        nu, b, h_min, h_max, h_DL, rho, rho_DL, F=1):
        """
        Discretized Timoshenko beam model for an Acoustic Black Hole (ABH)
        with an additive damping layer.

        Parameters
        ----------
        n : int
            Number of elements. Must be >= 2 (n - 1 ABH segments + 1 beam segment).
        n_x_disc : int
            x-axis discretisation for h2u.
        f_vec : array_like
            Frequency vector [Hz].
        l : float
            Total beam length.
        l_asl : float
            Length of the ABH region.
        p_h : float
            Power-law exponent of the ABH thickness profile.
        E_a, E_b : float
            log-fit parameters for the ABH base material's elastic modulus.
        Eta_a, Eta_b : float
            Power-fit parameters for the ABH base material's loss factor.
        E_a_DL, E_b_DL, Eta_a_DL, Eta_b_DL : float
            Same fits, for the damping layer material.
        nu : float
            Poisson's ratio.
        b : float
            Beam width.
        h_min : float
            Residual ABH thickness (tip).
        h_max : float
            Beam thickness (base).
        h_DL : float
            Damping-layer thickness.
        rho, rho_DL : float
            Density of base material and damping layer.
        F : float, optional
            Forcing amplitude (default 1).

        Returns
        -------
        f_vec : ndarray
            Echoed frequency vector.
        velocity_vec_x0 : ndarray
            Surface velocity at x = 0 (tip of the ABH).
        h2u : ndarray
            Mean-square mobility on the rear 2/3 of the beam.
        """
        if n < 2:
            raise ValueError(
                "n must be >= 2: need at least one ABH element plus the beam region."
            )

        n_freq = len(f_vec)

        # -------- Frequency-dependent material laws --------
        E_ASL   = self.E_log(f_vec, E_a, E_b)
        Eta_ASL = self.Eta_pot(f_vec, Eta_a, Eta_b, 0)
        E_DL    = self.E_log(f_vec, E_a_DL, E_b_DL)
        Eta_DL  = self.Eta_pot(f_vec, Eta_a_DL, Eta_b_DL, 0)

        Omega = 2 * np.pi * f_vec

        # -------- ABH discretisation: nodal x and segment lengths --------
        x_i        = (np.arange(n) / (n - 1) * l_asl ** p_h) ** (1 / p_h)
        l_asl_disc = np.diff(x_i)

        # -------- Per-element properties (length n) --------
        # Index 0..n-2 are ABH segments (with damping layer), n-1 is the beam region.
        elements = []
        for i in range(n):
            if i == n - 1:
                # Beam region: clean material, no damping layer.
                h_elem   = h_max
                E_elem   = E_ASL * (1 + 1j * Eta_ASL)
                rho_elem = rho
            else:
                # Mean full thickness of the i-th ABH segment (×2 to undo the
                # half-section convention used inside h_ASL).
                h_elem = integrat.quad(
                    lambda x: self.h_ASL(x, h_min / 2 + h_DL, h_max / 2 + h_DL, l_asl, p_h),
                    x_i[i], x_i[i + 1],
                )[0] / l_asl_disc[i] * 2
                E_elem = self.E_eff(
                    E1=E_ASL,  eta1=Eta_ASL,
                    E2=E_DL,   eta2=Eta_DL,
                    h_1=h_elem - 2 * h_DL,
                    h_2=h_DL,
                    h_12=h_elem,
                )
                rho_elem = (rho * (h_elem - 2 * h_DL) + rho_DL * 2 * h_DL) / h_elem

            elements.append(self._build_element(h_elem, E_elem, rho_elem, b, nu, Omega))

        # =========================================================
        # Assemble system matrix M of shape (n_freq, 4n, 4n)
        # =========================================================
        M = np.zeros((n_freq, n * 4, n * 4), dtype=complex)

        # ---- BCs at x = 0 (free tip of ABH) ----
        e0 = elements[0]
        M[:, 0, :4] = (e0['zeta'] * e0['psi']).T              # M_1 = 0
        M[:, 1, :4] = (e0['zeta'] + e0['psi']).T              # Q_1 = 0

        # ---- BCs at x = l (free moment, forced shear) ----
        en    = elements[-1]
        exp_n = np.exp(en['zeta'] * (l - l_asl))              # (4, n_freq)
        M[:, -2, -4:] = (en['zeta'] * en['psi'] * exp_n).T    # M_n = 0
        M[:, -1, -4:] = ((en['zeta'] + en['psi']) * exp_n).T  # Q_n = F input

        # ---- Continuity at the n-1 interior interfaces ----
        for ii in range(len(l_asl_disc)):
            e1, e2 = elements[ii], elements[ii + 1]
            exp1   = np.exp(e1['zeta'] * l_asl_disc[ii])      # (4, n_freq)

            EI1, EI2     = e1['E'] * e1['I'], e2['E'] * e2['I']
            GAk1, GAk2   = (e1['G'] * e1['kappa'] * e1['A'],
                            e2['G'] * e2['kappa'] * e2['A'])

            # Bending moment continuity:  E I zeta psi
            M[:, 2 + ii * 4, ii * 4    : ii * 4 + 4] =  (EI1 * e1['zeta'] * e1['psi'] * exp1).T
            M[:, 2 + ii * 4, ii * 4 + 4: ii * 4 + 8] = -(EI2 * e2['zeta'] * e2['psi']).T

            # Shear force continuity:    G kappa A (zeta + psi)
            M[:, 3 + ii * 4, ii * 4    : ii * 4 + 4] =  (GAk1 * (e1['zeta'] + e1['psi']) * exp1).T
            M[:, 3 + ii * 4, ii * 4 + 4: ii * 4 + 8] = -(GAk2 * (e2['zeta'] + e2['psi'])).T

            # Displacement continuity:   w
            M[:, 4 + ii * 4, ii * 4    : ii * 4 + 4] =  exp1.T
            M[:, 4 + ii * 4, ii * 4 + 4: ii * 4 + 8] = -1

            # Rotation continuity:       psi
            M[:, 5 + ii * 4, ii * 4    : ii * 4 + 4] =  (e1['psi'] * exp1).T
            M[:, 5 + ii * 4, ii * 4 + 4: ii * 4 + 8] = -e2['psi'].T

        # ---- Right-hand side: forcing at x = l ----
        r = np.zeros((n_freq, n * 4, 1), dtype=complex)
        r[:, -1, 0] = F / (en['G'] * en['A'] * en['kappa'])

        # ---- Sparse solve, one frequency at a time ----
        vec = np.array([
            spsolve(csr_matrix(M[i], dtype=complex), r[i])
            for i in range(n_freq)
        ])

        # =========================================================
        # Post-processing
        # =========================================================
        x_vec           = np.linspace(0, l, n_x_disc).round(9)
        x_vec_asl       = np.array([xi for xi in x_vec if xi <= l_asl][1:])  # skip x = 0
        x_vec_after_asl = np.array([
            np.round(xi - l_asl, 9) for xi in x_vec if xi > l_asl
        ])  # local coords starting from end of ABH

        # ---- Displacement on the ABH region ----
        displacement_vec_asl = np.zeros((n_freq, len(x_vec_asl)), dtype=complex)
        for ii in range(n - 1):
            mask = (x_vec_asl > x_i[ii]) & (x_vec_asl <= x_i[ii + 1])
            if not np.any(mask):
                continue
            x_locals  = np.round(x_vec_asl[mask] - x_i[ii], 9)            # (n_local,)
            zeta      = elements[ii]['zeta']                              # (4, n_freq)
            exp_term  = np.exp(zeta[:, :, None] * x_locals[None, None, :])# (4, n_freq, n_local)
            vec_part  = vec[:, ii * 4: ii * 4 + 4]                        # (n_freq, 4)
            displacement_vec_asl[:, mask] = np.einsum('fk,kfx->fx', vec_part, exp_term)

        # ---- Displacement at x = 0 (measurement point) ----
        displacement_vec_x0 = vec[:, :4].sum(axis=1).reshape(n_freq, 1)

        # ---- Displacement after the ABH (beam region) ----
        exp_after = np.exp(en['zeta'][:, :, None] * x_vec_after_asl[None, None, :])
        displacement_vec_after_asl = np.einsum('fk,kfx->fx', vec[:, -4:], exp_after)

        # ---- Stitch the full displacement field together ----
        Disp_vec = np.zeros((n_freq, len(x_vec)), dtype=complex)
        n_asl_pts = displacement_vec_asl.shape[1]
        Disp_vec[:, 0]                 = displacement_vec_x0[:, 0]
        Disp_vec[:, 1: n_asl_pts + 1]  = displacement_vec_asl
        Disp_vec[:, n_asl_pts + 1:]    = displacement_vec_after_asl

        # Velocity = j * Omega * displacement
        Velo_vec        = Disp_vec * 1j * Omega[:, None]
        velocity_vec_x0 = Velo_vec[:, 0]

        # =========================================================
        # Mean-square mobility on the rear 2/3 of the beam
        # =========================================================
        delta_L = np.abs((l - l_asl) - 2 / 3 * l)
        x_vec_2_3 = np.round(np.linspace(0, 2 / 3 * l, n_x_disc) + delta_L, 9)

        exp_2_3  = np.exp(en['zeta'][:, :, None] * x_vec_2_3[None, None, :])
        Disp_2_3 = np.einsum('fk,kfx->fx', vec[:, -4:], exp_2_3)
        Velo_2_3 = Disp_2_3 * 1j * Omega[:, None]

        h2u = np.mean(np.abs(Velo_2_3 ** 2), axis=1) / F ** 2
        return f_vec, velocity_vec_x0, h2u
