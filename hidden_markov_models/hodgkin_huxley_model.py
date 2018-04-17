import numpy as np
import copy

__author__ = 'caro'


class IonChannel:
    def __init__(self, g_max, ep, n_gates, power_gates, inf_gates, tau_gates):
        self.g_max = g_max  # (S/cm2)
        self.ep = ep  # (mV) equilibrium potential  # TODO: how to store ep global, for all ion channels accessible
        self.n_gates = n_gates
        self.power_gates = np.array(power_gates)
        self.inf_gates = inf_gates  # function of V
        self.tau_gates = tau_gates  # function of V

    def compute_current(self, vs, p_gates):
        if self.n_gates == 0:
            return self.g_max * (vs - self.ep)
        else:
            return self.g_max * np.prod(p_gates**self.power_gates) * (vs - self.ep)  # (mA/cm2)

    def compute_gmax_times_gates(self, p_gates, cell_area):
        if self.n_gates == 0:
            return self.g_max * cell_area  # (mA)
        else:
            return self.g_max * np.prod(p_gates**self.power_gates) * cell_area  # (mA)

    def init_gates(self, v0, p_gates0=None):
        if p_gates0 is not None:
            return p_gates0
        else:
            return self.inf_gates(v0)

    def derivative_gates(self, vs, p_gate, n):
        return (self.inf_gates(vs)[n] - p_gate) / self.tau_gates(vs)[n]


class Cell:
    def __init__(self, cm, length, diam, ionchannels):
        self.cm = cm  # (uF/cm2)
        self.length = length  # (um)
        self.diam = diam  # (um)
        self.ionchannels = ionchannels  # list of IonChannels
        self.cell_area = self.length * self.diam * np.pi * 1e-8  # (cm2)

        # unit conversions
        self.cm = self.cm * self.cell_area * 1e-3  # (mF)

    def derivative_v(self, i_ion, i_inj):
        i_ion = copy.copy(i_ion) * self.cell_area  # (mA)
        i_inj = copy.copy(i_inj) * 1e-6  # (mA)
        return (-1 * np.sum(i_ion, 0) + i_inj) / self.cm  # (mV/ms)


class Synapse:

    def __init__(self, ep, tau):
        self.ep = ep
        self.tau = tau


class HMMModel:

    def __init__(self, cell, exc_synapse, inh_synapse):
        self.cell = cell
        self.exc_synapse = exc_synapse
        self.inh_synapse = inh_synapse

        self.C = np.array([  # Linear factor to get from z_n to x_n
            [1, 0, 0]
        ])

    def get_trans(self, V_old, g_e_old, g_i_old, p_gates_old, i_inj, dt):
        g_max_times_gates = np.zeros(len(self.cell.ionchannels))
        p_gates = [0] * len(self.cell.ionchannels)
        eps = np.zeros(len(self.cell.ionchannels))
        for i, ionchannel in enumerate(self.cell.ionchannels):
            p_gates[i] = np.zeros(ionchannel.n_gates)
            for gate_idx in range(ionchannel.n_gates):
                x_old = p_gates_old[i][gate_idx]
                x_new = x_old + dt * ionchannel.derivative_gates(V_old, x_old, gate_idx)
                p_gates[i][gate_idx] = x_new
            g_max_times_gates[i] = ionchannel.compute_gmax_times_gates(p_gates_old[i], self.cell.cell_area)
            eps[i] = ionchannel.ep

        i_inj *= 1e-6  # (mA)
        g_e_old_ = copy.copy(g_e_old) * self.cell.cell_area
        g_i_old_ = copy.copy(g_i_old) * self.cell.cell_area

        trans_mat = np.array([
            [1 - dt * (np.sum(g_max_times_gates) + g_e_old_ + g_i_old_) / self.cell.cm,
             dt * self.exc_synapse.ep, dt * self.inh_synapse.ep],
            [0, 1 - dt / self.exc_synapse.tau, 0],
            [0, 0, 1 - dt / self.inh_synapse.tau]
        ])
        trans_const = np.array([
            [dt * (np.sum(g_max_times_gates * eps) + i_inj) / self.cell.cm],
            [0],
            [0]
        ])
        return trans_mat, trans_const, p_gates