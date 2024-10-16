#!/usr/bin/env python3

from __future__ import print_function

# General
import os
import subprocess
import sys
import tqdm
import yaml

# Analysis
import itertools
import ROOT
import argparse
import numpy as np

sys.path.append('.')

from jetscape_analysis.base import common_base

################################################################
class AnalyzeJetscapeEvents_Base(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file="", input_file="", output_file="", **kwargs):

        super(AnalyzeJetscapeEvents_Base, self).__init__(**kwargs)
        self.config_file = config_file
        self.input_file = input_file
        self.output_file = output_file

        self.initialize_config()

    # ---------------------------------------------------------------
    # Initialize config file into class members
    # ---------------------------------------------------------------
    def initialize_config(self):

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.debug_level = config['debug_level']
        self.n_event_max = config['n_event_max']
        self.reader_type = config['reader']
        self.progress_bar = config['progress_bar']
        self.merge_histograms = config['merge_histograms']
        self.selected_pid = config['selected_pid']
        self.write_Qn_vector_histograms = config['write_Qn_vector_histograms']

        # analysis parameters
        self.eta_min = config['eta_min']
        self.eta_max = config['eta_max']

        self.pT_low = config['pT_low']
        self.pT_high = config['pT_high']
        self.npT = config['npT']

        # parameters of Qn vector writer in JETSCAPE
        self.n_pt_bins = config['n_pt_bins']
        self.pt_min = config['pt_min']
        self.pt_max = config['pt_max']
        self.n_y_bins = config['n_y_bins']
        self.y_min = config['y_min']
        self.y_max = config['y_max']
        self.n_order = config['n_order']
        self.n_oversample = config['n_oversample']

    # ---------------------------------------------------------------
    # Main processing function
    # ---------------------------------------------------------------
    def analyze_jetscape_events(self):

        # Some preparation functions to be added

        # Read JETSCAPE Qn vector output and write histograms to ROOT file
        self.run_jetscape_analysis()

    # ---------------------------------------------------------------
    # Main processing function for events in a single event_QnVector.dat
    # ---------------------------------------------------------------
    def run_jetscape_analysis(self):
        # Create reader class
        all_events, total_events = self.reader_ascii(self.input_file)
        print("total_events", total_events)

        # Open the ROOT file
        output_file = ROOT.TFile(self.output_file, "RECREATE")

        # Create histograms
        histograms = self.initialize_result_histograms(total_events, self.n_pt_bins, self.pt_min, self.pt_max)

        # Iterate through events
        for event_id, event in all_events.items():
            print(f"Processing event_id: {event_id}, number of entries (N_pT times N_rapidity): {len(event)}")

            # Initialize output objects
            self.initialize_qnvector_histogram(event_id)
            
            # Fill histogram
            self.fill_qnvector_histogram(event)

            # Process histogram data to calculate observables
            dNdeta, mean_pt, psi_n_dict, pt_values, N_vn, vn_real_array, vn_imag_array, N_Qn_pT_array, Qn_pT_real_array, Qn_pT_imag_array, N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array = self.process_qnvector_histogram(event_id)

            # Clear histograms for memory efficiency
            self.clear_qnvector_histograms()

            # Fill the results histograms
            self.fill_result_histograms(histograms, event_id, dNdeta, mean_pt, psi_n_dict, pt_values, N_vn, vn_real_array, vn_imag_array, N_Qn_pT_array, Qn_pT_real_array, Qn_pT_imag_array, N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array)

        # Write histograms to file
        self.write_result_histograms(histograms)

        # Close the ROOT file
        output_file.Close()

    # ---------------------------------------------------------------
    # Initialize output objects for each event
    # ---------------------------------------------------------------  
    def initialize_qnvector_histogram(self, event_id):
        self.hist_dN = ROOT.TH2F(f"hist_dN_event_{event_id}", f"dN; pT; y; Event {event_id}", 
                                 self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
        self.hist_dNdpTdy = ROOT.TH2F(f"hist_dNdpTdy_event_{event_id}", f"dNdpTdy; pT; y; Event {event_id}", 
                                 self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
        self.hist_vncos = {}
        self.hist_vnsin = {}
        
        for n in range(1, self.n_order):  # Create histograms for each n
            self.hist_vncos[n] = ROOT.TH2F(f"hist_vncos_event_{event_id}_n{n}", 
                                           f"vncos (n={n}); pT; y; Event {event_id}", 
                                           self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
            self.hist_vnsin[n] = ROOT.TH2F(f"hist_vnsin_event_{event_id}_n{n}", 
                                           f"vnsin (n={n}); pT; y; Event {event_id}", 
                                           self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)

    # ---------------------------------------------------------------
    # Function to fill 2D histogram with Qn vector results for a single event
    # ---------------------------------------------------------------
    def fill_qnvector_histogram(self, qnvector_results):

        # Loop over (N_pT X N_rapidity) entries of each event
        # Note: only results for pid=9999 are read and passed to qnvector_results here
        for result in qnvector_results:

            pt = result[1]
            y = result[3]

            dNdpTdy = result[6] / pt # dN/oversample/(dpTdy)/pT

            # Only consider valid dN entries
            dN = result[-1]
            if dN <= 0:
                continue  # Skip entries with non-positive dN

            # Loop over vn orders
            for n in range(1, self.n_order):  
                vncos = result[8 + (n-1)*4]
                vncos_err = result[9 + (n-1)*4]
                vnsin = result[10 + (n-1)*4]
                vnsin_err = result[11 + (n-1)*4]

                # Fill the histograms with errors
                self.hist_vncos[n].Fill(pt, y, vncos)
                self.hist_vncos[n].SetBinError(self.hist_vncos[n].FindBin(pt, y), vncos_err)
                self.hist_vnsin[n].Fill(pt, y, vnsin)
                self.hist_vnsin[n].SetBinError(self.hist_vnsin[n].FindBin(pt, y), vnsin_err)

            self.hist_dNdpTdy.Fill(pt, y, dN)
            self.hist_dN.Fill(pt, y, dN)

    # ---------------------------------------------------------------
    # Clear output objects after processing each event
    # ---------------------------------------------------------------
    def clear_qnvector_histograms(self):
        self.hist_dN.Delete()
        self.hist_dNdpTdy.Delete()
        for n in range(1, self.n_order):
            self.hist_vncos[n].Delete()
            self.hist_vnsin[n].Delete()

    # ---------------------------------------------------------------
    # Calculate necessary quantities using the histogram of Qn vector results for a single event
    # --------------------------------------------------------------
    def process_qnvector_histogram(self, event_id):
        """
        This function processes 2D histograms (in pT * rapidity bins) of a single event
        and reduces them to 1D histograms in pT by summing contributions over rapidity (y) bins.
        """

        # I. Obtain 1D arrays in pT from 2D histograms in pT and rapidity of a single event (with oversamples included)

        # Lists in pT
        pt_values = []
        dN_values = []
        dNdpTdy_values = []
        vncos_values = {n: [] for n in range(1, self.n_order)}
        vnsin_values = {n: [] for n in range(1, self.n_order)}

        # Loop over pT bins
        for bin_x in range(1, self.hist_dN.GetNbinsX() + 1):
            pt = self.hist_dN.GetXaxis().GetBinCenter(bin_x)

            # Initialize summation for current pT bin
            dN_sum = 0.0
            dNdpTdy_sum = 0.0
            vncos_sum = {n: 0.0 for n in range(1, self.n_order)}
            vnsin_sum = {n: 0.0 for n in range(1, self.n_order)}

            # Loop over rapidity bins
            for bin_y in range(1, self.hist_dN.GetNbinsY() + 1):
                y = self.hist_dN.GetYaxis().GetBinCenter(bin_y)

                # Check if rapidity is within the specified range
                if self.eta_min <= y <= self.eta_max:
                    dN = self.hist_dN.GetBinContent(bin_x, bin_y)
                    dNdpTdy = self.hist_dNdpTdy.GetBinContent(bin_x, bin_y)  # Get dNdpTdy value

                    # Sum dN and dNdpTdy over rapidity bins
                    dN_sum += dN
                    dNdpTdy_sum += dNdpTdy

                    # Sum vncos and vnsin for each harmonic n, weighted by dN
                    for n in range(1, self.n_order):
                        vncos = self.hist_vncos[n].GetBinContent(bin_x, bin_y)
                        vnsin = self.hist_vnsin[n].GetBinContent(bin_x, bin_y)
                        vncos_sum[n] += vncos * dN  # Weighted by dN
                        vnsin_sum[n] += vnsin * dN  # Weighted by dN

            # Append the summed values for this pT bin if dN_sum > 0
            if dN_sum > 0:
                pt_values.append(pt)
                dN_values.append(dN_sum)
                dNdpTdy_values.append(dNdpTdy_sum)

                # Normalize vncos_sum and vnsin_sum by dN_sum for each harmonic n
                for n in range(1, self.n_order):
                    vncos_values[n].append(vncos_sum[n] / dN_sum if dN_sum > 0 else 0.0)
                    vnsin_values[n].append(vnsin_sum[n] / dN_sum if dN_sum > 0 else 0.0)

        # Convert lists to arrays for further processing
        pt_values = np.array(pt_values)
        dN_values = np.array(dN_values)
        dNdpTdy_values = np.array(dNdpTdy_values)

        for n in range(1, self.n_order):
            vncos_values[n] = np.array(vncos_values[n])
            vnsin_values[n] = np.array(vnsin_values[n])

        # II. Calculate mean pT and yield of a single event

        # Calculate total number of particles N_tot
        N_tot = np.sum(dN_values)

        # Calculate dN_ch/deta: N_tot divided by oversample number and rapidity window width
        dNdeta = N_tot / self.n_oversample / (self.eta_max - self.eta_min)

        # Calculate mean pT, weighted by dN
        if N_tot > 0:
            mean_pt = np.sum(pt_values * dN_values) / N_tot
        else:
            mean_pt = 0.0  # Handle the case where no entries are valid

        # III. Calculate quantities for flow calculations

        psi_n_dict = {}

        # Calculate psi_n for each harmonic n
        for n in range(1, self.n_order):
            # Sum vncos and vnsin for psi_n calculation
            vncos_sum = np.sum(vncos_values[n])
            vnsin_sum = np.sum(vnsin_values[n])

            # Calculate psi_n
            psi_n = (1.0 / n) * np.arctan2(vnsin_sum, vncos_sum)
            psi_n_dict[n] = psi_n

        # Ensure there are valid entries for interpolation
        if len(pt_values) == 0 or len(dN_values) == 0:
            raise ValueError("No valid entries for interpolation.")

        # pT array for interpolation
        npT, pT_low, pT_high = self.npT, self.pT_low, self.pT_high
        pT_inte_array = np.linspace(pT_low, pT_high, npT)

        # integrated vn
        vn_real_array = {} 
        vn_imag_array = {}
        N_vn = {}
        
        N_vn, vn_real_array, vn_imag_array = self.calculate_inte_vn_single_event(pT_inte_array, pt_values, dNdpTdy_values, dN_values, vncos_values, vnsin_values)

        # pT-differential vn
        Qn_pT_real_array = {} 
        Qn_pT_imag_array = {}
        N_Qn_pT = {}
        # reference particles
        Qn_ref_real_array = {} 
        Qn_ref_imag_array = {}
        N_Qn_ref = {}

        # here we use all particles within the kinematic ranges as reference particles
        pt_ref_values, dN_ref_values, vncos_ref_values, vnsin_ref_values = pt_values, dN_values, vncos_values, vnsin_values

        N_Qn_pT, Qn_pT_real_array, Qn_pT_imag_array, N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array = self.calculate_diff_vn_single_event(pT_inte_array, pt_values, dN_values, vncos_values, vnsin_values, 
            pt_ref_values, dN_ref_values, vncos_ref_values, vnsin_ref_values)

        return dNdeta, mean_pt, psi_n_dict, pt_values, N_vn, vn_real_array, vn_imag_array, N_Qn_pT, Qn_pT_real_array, Qn_pT_imag_array, N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array

    # ---------------------------------------------------------------
    # Calculate vn for a single event
    # ---------------------------------------------------------------
    def calculate_inte_vn_single_event(self, pT_inte_array, pt_values, dNdpTdy_values, dN_values, vncos_values, vnsin_values):
        """
        This function calculates the pT-integrated vn in a given pT range (pT_low, pT_high)
        using the processed data from process_histogram.
        """
        
        dpT = pT_inte_array[1] - pT_inte_array[0]

        dN_interp = np.exp(np.interp(pT_inte_array, pt_values, np.log(dNdpTdy_values + 1e-30)))
        N_interp = np.exp(np.interp(pT_inte_array, pt_values, np.log(dN_values + 1e-30)))
        N = np.sum(N_interp) * dpT / 0.1

        vn_real_array = []
        vn_imag_array = []

        for iorder in range(1, self.n_order):
            vn_real_interp = np.interp(pT_inte_array, pt_values, vncos_values[iorder])
            vn_imag_interp = np.interp(pT_inte_array, pt_values, vnsin_values[iorder])
            vn_real_inte = (
                np.sum(vn_real_interp * dN_interp * pT_inte_array) / np.sum(dN_interp * pT_inte_array)
            )
            vn_imag_inte = (
                np.sum(vn_imag_interp * dN_interp * pT_inte_array) / np.sum(dN_interp * pT_inte_array)
            )
            vn_real_array.append(vn_real_inte)
            vn_imag_array.append(vn_imag_inte)

        return N, vn_real_array, vn_imag_array

    def calculate_diff_vn_single_event(self, pT_inte_array, pt_values, dN_values, vncos_values, vnsin_values, pt_ref_values, dN_ref_values, vncos_ref_values, vnsin_ref_values):
        """
        This function computes pT-differential vn{4} for a single event using processed data and reference data.
        """
        dpT = pT_inte_array[1] - pT_inte_array[0]

        dN_ref_interp = np.exp(np.interp(pT_inte_array, pt_ref_values, np.log(dN_ref_values + 1e-30)))
        dN_ref = np.sum(dN_ref_interp) * dpT / 0.1

        Qn_pT_real_array = []
        Qn_pT_imag_array = []
        Qn_ref_real_array = []
        Qn_ref_imag_array = []

        for iorder in range(1, self.n_order):
            vn_ref_real_interp = np.interp(pT_inte_array, pt_ref_values, vncos_ref_values[iorder])
            vn_ref_imag_interp = np.interp(pT_inte_array, pt_ref_values, vnsin_ref_values[iorder])
            vn_ref_real_inte = (
                np.sum(vn_ref_real_interp * dN_ref_interp) / np.sum(dN_ref_interp)
            )
            vn_ref_imag_inte = (
                np.sum(vn_ref_imag_interp * dN_ref_interp) / np.sum(dN_ref_interp)
            )
            Qn_ref_real_array.append(vn_ref_real_inte)
            Qn_ref_imag_array.append(vn_ref_imag_inte)

            Qn_pt_real = dN_values * vncos_values[iorder]
            Qn_pt_imag = dN_values * vnsin_values[iorder]

            Qn_pT_real_array.append(Qn_pt_real)
            Qn_pT_imag_array.append(Qn_pt_imag)

        return dN_values, Qn_pT_real_array, Qn_pT_imag_array, dN_ref, Qn_ref_real_array, Qn_ref_imag_array

    # ---------------------------------------------------------------
    # Calculate histograms for results
    # ---------------------------------------------------------------
    def initialize_result_histograms(self, total_events, npT, pT_min, pT_max):
        hist_N_tot = ROOT.TH1F(f"hist_Nch", f"charged multiplicity; Event ID; Nch", total_events, 1, total_events + 1)
        hist_mean_pt = ROOT.TH1F(f"hist_mean_pT", f"mean pT; Event ID; mean_pT", total_events, 1, total_events + 1)

        hist_event_plane_angles = {}

        hist_vn_real = {}
        hist_vn_imag = {}
        hist_N_vn = {}

        hist_Qn_pT_real = {}
        hist_Qn_pT_imag = {}
        hist_N_Qn_pT = {}
        hist_Qn_ref_real = {}
        hist_Qn_ref_imag = {}
        hist_N_Qn_ref = {}

        hist_N_vn = ROOT.TH1F(f"hist_N_vn", f"hist_N_vn; Event ID; N_vn", total_events, 1, total_events + 1)
        hist_N_Qn_pT = ROOT.TH2F(f"hist_N_Qn_pT", f"hist_N_Qn_pT; Event ID; pT; N_Qn_pT", total_events, 1, total_events + 1, npT, pT_min, pT_max)
        hist_N_Qn_ref = ROOT.TH1F(f"hist_N_Qn_ref", f"hist_N_Qn_ref; Event ID; N_Qn_ref", total_events, 1, total_events + 1)

        for n in range(1, self.n_order):
            hist_event_plane_angles[n] = ROOT.TH1F(f"hist_event_plane_angles_n{n}", f"Event Plane Angles (n={n}); Event ID; Psi_{n}", total_events, 1, total_events + 1)

            hist_vn_real[n] = ROOT.TH1F(f"hist_vn_real_n{n}", f"vn real (n={n}); Event ID; v{n}", total_events, 1, total_events + 1)
            hist_vn_imag[n] = ROOT.TH1F(f"hist_vn_imag_n{n}", f"vn imag (n={n}); Event ID; v{n}", total_events, 1, total_events + 1)
            
            # 2D histogram for Qn_pT (pT vs Event ID)
            hist_Qn_pT_real[n] = ROOT.TH2F(f"hist_Qn_pT_real_n{n}", f"Qn pT real (n={n}); Event ID; pT; Qn", total_events, 1, total_events + 1, npT, pT_min, pT_max)
            hist_Qn_pT_imag[n] = ROOT.TH2F(f"hist_Qn_pT_imag_n{n}", f"Qn pT imag (n={n}); Event ID; pT; Qn", total_events, 1, total_events + 1, npT, pT_min, pT_max)
            
            hist_Qn_ref_real[n] = ROOT.TH1F(f"hist_Qn_ref_real_n{n}", f"Qn ref real (n={n}); Event ID; Qn_ref", total_events, 1, total_events + 1)
            hist_Qn_ref_imag[n] = ROOT.TH1F(f"hist_Qn_ref_imag_n{n}", f"Qn ref imag (n={n}); Event ID; Qn_ref", total_events, 1, total_events + 1)
            

        return {
            'hist_N_tot': hist_N_tot,
            'hist_mean_pt': hist_mean_pt,
            'hist_event_plane_angles': hist_event_plane_angles,
            'hist_vn_real': hist_vn_real,
            'hist_vn_imag': hist_vn_imag,
            'hist_N_vn': hist_N_vn,
            'hist_Qn_pT_real': hist_Qn_pT_real,
            'hist_Qn_pT_imag': hist_Qn_pT_imag,
            'hist_N_Qn_pT': hist_N_Qn_pT,
            'hist_Qn_ref_real': hist_Qn_ref_real,
            'hist_Qn_ref_imag': hist_Qn_ref_imag,
            'hist_N_Qn_ref': hist_N_Qn_ref,
        }

    # ---------------------------------------------------------------
    # Fill histograms of results
    # ---------------------------------------------------------------
    def fill_result_histograms(self, histograms, event_id, dNdeta, mean_pt, psi_n_dict, pt_values, N_vn, vn_real_array, vn_imag_array, N_Qn_pT_array, Qn_pT_real_array, Qn_pT_imag_array, N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array):
        histograms['hist_N_tot'].Fill(event_id, dNdeta)
        histograms['hist_mean_pt'].Fill(event_id, mean_pt)

        # Fill N_vn, N_Qn_pT, and N_Qn_ref (shared across all harmonics n)
        histograms['hist_N_vn'].Fill(event_id, N_vn)
        histograms['hist_N_Qn_ref'].Fill(event_id, N_Qn_ref)

        for i, N_Qn_pT in enumerate(N_Qn_pT_array):
            histograms['hist_N_Qn_pT'].Fill(event_id, pt_values[i], N_Qn_pT)

        for n in range(1, self.n_order):
            histograms['hist_event_plane_angles'][n].Fill(event_id, psi_n_dict[n])

            histograms['hist_vn_real'][n].Fill(event_id, float(vn_real_array[n - 1]))
            histograms['hist_vn_imag'][n].Fill(event_id, float(vn_imag_array[n - 1]))

            # Fill the 2D histogram for each pT value
            for i, qn_pT_real in enumerate(Qn_pT_real_array[n - 1]):
                histograms['hist_Qn_pT_real'][n].Fill(event_id, pt_values[i], float(qn_pT_real))
            for i, qn_pT_imag in enumerate(Qn_pT_imag_array[n - 1]):
                histograms['hist_Qn_pT_imag'][n].Fill(event_id, pt_values[i], float(qn_pT_imag))

            histograms['hist_Qn_ref_real'][n].Fill(event_id, float(Qn_ref_real_array[n - 1]))
            histograms['hist_Qn_ref_imag'][n].Fill(event_id, float(Qn_ref_imag_array[n - 1]))

    # ---------------------------------------------------------------
    # Write histograms of results
    # ---------------------------------------------------------------
    def write_result_histograms(self, histograms):
        histograms['hist_N_tot'].Write()
        histograms['hist_mean_pt'].Write()
        histograms['hist_N_vn'].Write()
        histograms['hist_N_Qn_ref'].Write()
        histograms['hist_N_Qn_pT'].Write()

        for n in range(1, self.n_order):
            histograms['hist_event_plane_angles'][n].Write()
            histograms['hist_vn_real'][n].Write()
            histograms['hist_vn_imag'][n].Write()
            histograms['hist_Qn_pT_real'][n].Write()
            histograms['hist_Qn_pT_imag'][n].Write()
            histograms['hist_Qn_ref_real'][n].Write()
            histograms['hist_Qn_ref_imag'][n].Write()

    # ---------------------------------------------------------------
    # Reader of the Qn vector file
    # ---------------------------------------------------------------
    def reader_ascii(self, input_file):
        """
        This function reads Qn vector data from the specified file.
        Each event is stored in a dictionary with the event number as the key.
        Each event has averaged over results of oversampled particle lists
        """
        all_events = {}
        current_event = []
        last_event_number = None

        with open(input_file, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith("#"):
                    if "Event" in stripped_line:
                        event_number = int(stripped_line.split()[2])

                        if current_event and last_event_number is not None:
                            all_events[last_event_number] = current_event

                        current_event = []
                        last_event_number = event_number
                    elif "\tEvent" in stripped_line and "End" in stripped_line:
                        if current_event and last_event_number is not None:
                            all_events[last_event_number] = current_event
                        break
                else:
                    try:
                        data = line.split()
                        if int(data[0]) in self.selected_pid:
                            numeric_data = [int(data[0])] + [float(x) for x in data[1:]]
                            current_event.append(numeric_data)
                    except IndexError:
                        # Skip incomplete lines
                        continue

        if current_event and last_event_number is not None:
            all_events[last_event_number] = current_event

        total_events = len(all_events)
        return all_events, total_events

##################################################################
if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Analyze JETSCAPE events")
    parser.add_argument(
        "-c",
        "--configFile",
        action="store",
        type=str,
        metavar="configFile",
        default="/home/jetscape-user/JETSCAPE-analysis/config/jetscapeAnalysisConfig.yaml",
        help="Path of config file for analysis",
    )
    parser.add_argument(
        "-i",
        "--inputFilename",
        action="store",
        type=str,
        metavar="inputFilename",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Input directory containing JETSCAPE output files",
    )
    parser.add_argument(
        "-o",
        "--outputFilename",
        action="store",
        type=str,
        metavar="outputFilename",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Output directory for output to be written to",
    )

    # Parse the arguments
    args = parser.parse_args()

    print("Analyze the Qn vector to obtain event plane angles and v2s ...")

    # Run the analysis
    analysis = AnalyzeJetscapeEvents_Base(config_file=args.configFile, input_file=args.inputFilename, output_file=args.outputFilename)
    analysis.analyze_jetscape_events()
