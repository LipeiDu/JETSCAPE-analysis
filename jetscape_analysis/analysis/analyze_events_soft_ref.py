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

        self.qnvector_histograms = {}

    # ---------------------------------------------------------------
    # Initialize config file into class members
    # ---------------------------------------------------------------
    def initialize_config(self):

        # Read config file
        with open(self.config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.selected_pid = self.config['selected_pid']

        # centrality of the hydro event
        self.centrality = [40, 50]

        # parameters of Qn vector writer in JETSCAPE
        self.n_pt_bins = self.config['n_pt_bins']
        self.pt_min = self.config['pt_min']
        self.pt_max = self.config['pt_max']
        self.n_y_bins = self.config['n_y_bins']
        self.y_min = self.config['y_min']
        self.y_max = self.config['y_max']
        self.n_order = self.config['n_order']
        self.n_oversample = self.config['n_oversample']

        # pT arrays for interpolation in flow calculation
        self.pT_low = self.config['pT_low']
        self.pT_high = self.config['pT_high']
        self.npT = self.config['npT']

        # to be moved
        self.eta_min_ref = self.config['eta_min_ref']
        self.eta_max_ref = self.config['eta_max_ref']
        self.eta_min = self.config['eta_min']
        self.eta_max = self.config['eta_max']

    # ---------------------------------------------------------------
    # Main processing function
    # ---------------------------------------------------------------
    def analyze_jetscape_events(self):

        # Read JETSCAPE Qn vector output and write histograms to ROOT file
        self.run_jetscape_analysis()

    # ---------------------------------------------------------------
    # Main processing function for events in a single event_QnVector.dat
    # ---------------------------------------------------------------
    def run_jetscape_analysis(self):

        # Read Qn vector data (common for all observables)
        all_events, total_events = self.reader_ascii(self.input_file)
        print(f"Total events in centrality {self.centrality}: {total_events}")

        # Open the ROOT file
        output_file = ROOT.TFile(self.output_file, "RECREATE")

        # Prepare result histograms for all observables before processing events
        result_histograms = {}  # Store histograms for each observable
        for observable_type in self.config:
            if observable_type not in ['soft_differential']:  # Skip unsupported types
                continue
            for observable_type in ['soft_differential']:
                for observable, block in self.config[observable_type].items():
                    for centrality_index, centrality in enumerate(block['centrality']):
                        if centrality != self.centrality:
                            continue  # Skip centrality mismatch

                        # Initialize histograms for the observable
                        result_histograms[(observable_type, observable, tuple(centrality))] = self.initialize_result_histograms(
                            observable_type, observable, centrality, total_events
                        )

        # Process events and loop over observables
        for event_id, event in all_events.items():
            print(f"Processing event_id: {event_id}, entries: {len(event)}")

            # Initialize and fill Qn vector histograms of a single event (common for all observables)
            self.initialize_qnvector_histogram()
            self.fill_qnvector_histogram(event)

            # Loop over observables for event-specific processing
            for (observable_type, observable, centrality), histograms in result_histograms.items():
                print()
                print(f'Histogram {observable_type} observables...')

                # Apply observable-specific processing
                results = self.process_qnvector_histogram(event_id)
                if results:
                    self.fill_result_histograms(observable_type, observable, histograms, event_id, results)

            # Clear Qn vector histograms after processing
            self.clear_qnvector_histograms()

        # Write histograms for all observables to the output ROOT file
        for histograms in result_histograms.values():
            self.write_result_histograms(histograms)

        # Close the ROOT file
        output_file.Close()

    # ---------------------------------------------------------------
    # Initialize output objects for each event
    # ---------------------------------------------------------------  
    def initialize_qnvector_histogram(self):
        if not self.qnvector_histograms:
            self.qnvector_histograms['hist_dN'] = ROOT.TH2F("hist_dN", "dN; pT; y;", self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
            self.qnvector_histograms['hist_vncos'] = {n: ROOT.TH2F(f"hist_vncos_n{n}", f"vncos (n={n}); pT; y;", 
                                                                   self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max) for n in range(1, self.n_order)}
            self.qnvector_histograms['hist_vnsin'] = {n: ROOT.TH2F(f"hist_vnsin_n{n}", f"vnsin (n={n}); pT; y;", 
                                                                   self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max) for n in range(1, self.n_order)}

    # ---------------------------------------------------------------
    # Function to fill 2D histogram with Qn vector results for a single event
    # ---------------------------------------------------------------
    def fill_qnvector_histogram(self, qnvector_results):
        """
        In qnvector histograms, only dNdpTdy is averaged over oversamples; the other ones are summed over oversamples

        """
        # Loop over (N_pT X N_rapidity) entries of each event
        # Note: only results for pid=9999 are read and passed to qnvector_results here

        self.hist_dN = self.qnvector_histograms['hist_dN']
        self.hist_vncos = self.qnvector_histograms['hist_vncos']
        self.hist_vnsin = self.qnvector_histograms['hist_vnsin']

        for result in qnvector_results:
            pt, y, dN = result[1], result[3], result[-1]
            if dN <= 0:
                continue
            self.hist_dN.Fill(pt, y, dN)
            for n in range(1, self.n_order):
                vncos, vnsin = result[8 + (n - 1) * 4], result[10 + (n - 1) * 4]
                self.hist_vncos[n].Fill(pt, y, vncos)
                self.hist_vnsin[n].Fill(pt, y, vnsin)

    # ---------------------------------------------------------------
    # Clear output objects after processing each event
    # ---------------------------------------------------------------
    def clear_qnvector_histograms(self):
        self.qnvector_histograms['hist_dN'].Reset()
        for n in range(1, self.n_order):
            self.qnvector_histograms['hist_vncos'][n].Reset()
            self.qnvector_histograms['hist_vnsin'][n].Reset()

    # ---------------------------------------------------------------
    # Calculate necessary quantities using the histogram of Qn vector results for a single event
    # --------------------------------------------------------------
    def process_qnvector_histogram(self, event_id):
        """
        This function processes 2D histograms (in pT * rapidity bins) of a single event
        and reduces them to 1D histograms in pT by summing contributions over rapidity (y) bins.
        """

        # I. Obtain 1D arrays in pT from 2D histograms in pT and rapidity of a single event (with oversamples summed over)

        # List of particle number in pT around midrapidity
        dN_values = []

        # Reference lists in pT for a different rapidity range
        pt_ref_values = []
        dN_ref_values = []
        vncos_ref_values = {n: [] for n in range(1, self.n_order)}
        vnsin_ref_values = {n: [] for n in range(1, self.n_order)}

        # Loop over pT bins
        for bin_x in range(1, self.hist_dN.GetNbinsX() + 1):

            pt = self.hist_dN.GetXaxis().GetBinCenter(bin_x)
            dN_sum, dN_ref_sum = 0.0, 0.0
            vncos_ref_sum = {n: 0.0 for n in range(1, self.n_order)}
            vnsin_ref_sum = {n: 0.0 for n in range(1, self.n_order)}

            # Loop over rapidity bins
            for bin_y in range(1, self.hist_dN.GetNbinsY() + 1):
                y = self.hist_dN.GetYaxis().GetBinCenter(bin_y)
                dN = self.hist_dN.GetBinContent(bin_x, bin_y)

                # midrapidity multiplicity
                if self.eta_min <= y <= self.eta_max:
                    dN_sum += dN

                # reference particles
                if self.eta_min_ref <= y <= self.eta_max_ref:
                    dN_ref_sum += dN

                    # Sum vncos and vnsin for each harmonic n, weighted by dN for the reference particles
                    for n in range(1, self.n_order):
                        vncos_ref_sum[n] += self.hist_vncos[n].GetBinContent(bin_x, bin_y) * dN
                        vnsin_ref_sum[n] += self.hist_vnsin[n].GetBinContent(bin_x, bin_y) * dN

            # Append the summed values for this pT bin if dN_sum > 0
            if dN_sum > 0:
                dN_values.append(dN_sum)

            # Append the reference values for this pT bin if dN_ref_sum > 0
            if dN_ref_sum > 0:
                pt_ref_values.append(pt)
                dN_ref_values.append(dN_ref_sum)

                # Normalize vncos_ref_sum and vnsin_ref_sum by dN_ref_sum for each harmonic n
                for n in range(1, self.n_order):
                    vncos_ref_values[n].append(vncos_ref_sum[n] / dN_ref_sum if dN_ref_sum > 0 else 0.0)
                    vnsin_ref_values[n].append(vnsin_ref_sum[n] / dN_ref_sum if dN_ref_sum > 0 else 0.0)

        # II. Multiplicity calculation

        # Convert lists to arrays for further processing
        dN_values = np.array(dN_values)

        # Ensure there are valid entries for interpolation
        if len(dN_values) == 0:
            print(f"No valid entries for event ID {event_id}. Skipping...")
            return None

        # Calculate total number of particles N_tot
        N_tot = np.sum(dN_values)

        # Calculate dN_ch/deta: N_tot divided by oversample number and rapidity window width
        dNdeta = N_tot / self.n_oversample / (self.eta_max - self.eta_min)

        # III. Calculate quantities for flow calculations

        pt_ref_values = np.array(pt_ref_values)
        dN_ref_values = np.array(dN_ref_values)

        for n in range(1, self.n_order):
            vncos_ref_values[n] = np.array(vncos_ref_values[n])
            vnsin_ref_values[n] = np.array(vnsin_ref_values[n])

        # pT array for interpolation
        npT, pT_low, pT_high = self.npT, self.pT_low, self.pT_high
        pT_inte_array = np.linspace(pT_low, pT_high, npT)

        # vn for reference particles
        N_Qn_ref = {}
        Qn_ref_real_array = {}
        Qn_ref_imag_array = {}
        
        N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array = self.calculate_diff_vn_single_event(pT_inte_array, pt_ref_values, dN_ref_values, vncos_ref_values, vnsin_ref_values)

        return {
            "dNdeta": dNdeta,
            "N_Qn_ref": N_Qn_ref,
            "Qn_ref_real": Qn_ref_real_array,
            "Qn_ref_imag": Qn_ref_imag_array,
        }

    # ---------------------------------------------------------------
    # Calculate vn for a single event
    # ---------------------------------------------------------------
    def calculate_diff_vn_single_event(self, pT_inte_array, pt_ref_values, dN_ref_values, vncos_ref_values, vnsin_ref_values):
        """
        This function computes pT-differential vn{4} and vn{SP} for a single event using reference particle data.
        """
        dpT = pT_inte_array[1] - pT_inte_array[0]

        dN_ref_interp = np.exp(np.interp(pT_inte_array, pt_ref_values, np.log(dN_ref_values + 1e-30)))
        N_Qn_ref = np.sum(dN_ref_interp) * dpT / 0.1

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
            Qn_ref_real_array.append(N_Qn_ref * vn_ref_real_inte)
            Qn_ref_imag_array.append(N_Qn_ref * vn_ref_imag_inte)

        return N_Qn_ref, Qn_ref_real_array, Qn_ref_imag_array

    # ---------------------------------------------------------------
    # Calculate histograms for results
    # ---------------------------------------------------------------
    def initialize_result_histograms(self, observable_type, observable, centrality, total_events):
        base_name = f"h_{observable_type}_{observable}_{centrality}"
        histograms = {}

        histograms['hist_dNdeta'] = ROOT.TH1F(f"{base_name}_dNchdeta", "Multiplicity; Event ID; Nch", total_events, 1, total_events + 1)

        histograms['hist_N_Qn_ref'] = ROOT.TH1F(f"{base_name}_N_Qn_ref", f"hist_N_Qn_ref; Event ID; N_Qn_ref", total_events, 1, total_events + 1)
        histograms['hist_Qn_ref_real'] =  {n: ROOT.TH1F(f"{base_name}_Qn_ref_real_n{n}", f"Qn ref real (n={n}); Event ID; Qn_ref", total_events, 1, total_events + 1) for n in range(1, self.n_order)}
        histograms['hist_Qn_ref_imag'] =  {n: ROOT.TH1F(f"{base_name}_Qn_ref_imag_n{n}", f"Qn ref imag (n={n}); Event ID; Qn_ref", total_events, 1, total_events + 1) for n in range(1, self.n_order)}

        return histograms

    # ---------------------------------------------------------------
    # Fill histograms of results
    # ---------------------------------------------------------------
    def fill_result_histograms(self, observable_type, observable, histograms, event_id, results):
        histograms['hist_dNdeta'].Fill(event_id, results["dNdeta"])
        histograms['hist_N_Qn_ref'].Fill(event_id, results["N_Qn_ref"])
        for n in range(1, self.n_order):
            histograms['hist_Qn_ref_real'][n].Fill(event_id, results["Qn_ref_real"][n - 1])
            histograms['hist_Qn_ref_imag'][n].Fill(event_id, results["Qn_ref_imag"][n - 1])

    # ---------------------------------------------------------------
    # Write histograms of results
    # ---------------------------------------------------------------
    def write_result_histograms(self, histograms):
        for key, hist in histograms.items():
            if isinstance(hist, dict):
                for subkey, subhist in hist.items():
                    subhist.Write()
            else:
                hist.Write()

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
