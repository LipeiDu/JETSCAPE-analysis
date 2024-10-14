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

        # Create histograms to store charged multiplicity and mean pT
        hist_N_tot = ROOT.TH1F(f"hist_Nch", f"charged multiplicity; Event ID; Nch",
                                                   total_events, 1, total_events + 1)
            
        hist_mean_pt = ROOT.TH1F(f"hist_mean_pT", f"mean pT; Event ID; mean_pT",
                                                   total_events, 1, total_events + 1)

        # Create histograms to store event plane angles and vn for each n
        hist_event_plane_angles = {}
        hist_vn_real = {}
        hist_vn_imag = {}

        for n in range(1, self.n_order):  # Loop over n
            hist_event_plane_angles[n] = ROOT.TH1F(f"hist_event_plane_angles_n{n}",
                                                   f"Event Plane Angles (n={n}); Event ID; Psi_{n}",
                                                   total_events, 1, total_events + 1)
            
            hist_vn_real[n] = ROOT.TH1F(f"hist_vn_real_n{n}", f"vn real (n={n}); Event ID; v{n}",
                                              total_events, 1, total_events + 1)

            hist_vn_imag[n] = ROOT.TH1F(f"hist_vn_imag_n{n}", f"vn imag (n={n}); Event ID; v{n}",
                                              total_events, 1, total_events + 1)

        # Iterate through events
        for event_id, event in all_events.items():
            print(f"Processing event_id: {event_id}, number of entries (N_pT times N_rapidity): {len(event)}")

            # Initialize output objects
            self.initialize_output_objects(event_id)
            
            # Fill histogram
            self.fill_histogram_from_qnvector(event)

            # Process histogram data to calculate observables
            N_tot, mean_pt, psi_n_dict, vn_real_dict, vn_imag_dict = self.process_histogram(event_id)

            # Fill the results histograms
            hist_N_tot.Fill(event_id, N_tot)
            hist_mean_pt.Fill(event_id, mean_pt)

            for n in range(1, self.n_order):
                hist_event_plane_angles[n].Fill(event_id, psi_n_dict[n])
                hist_vn_real[n].Fill(event_id, vn_real_dict[n])
                hist_vn_imag[n].Fill(event_id, vn_imag_dict[n])

            # Clear histograms for memory efficiency
            self.clear_histograms()

        # Write results histograms of all events to file
        hist_N_tot.Write()
        hist_mean_pt.Write()

        for n in range(1, self.n_order):
            hist_event_plane_angles[n].Write()
            hist_vn_real[n].Write()
            hist_vn_imag[n].Write()

        # Close the ROOT file
        output_file.Close()

    # ---------------------------------------------------------------
    # Initialize output objects for each event
    # ---------------------------------------------------------------  
    def initialize_output_objects(self, event_id):
        self.hist_dN = ROOT.TH2F(f"hist_dN_event_{event_id}", f"dN; pT; y; Event {event_id}", 
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
    # Clear output objects after processing each event
    # ---------------------------------------------------------------
    def clear_histograms(self):
        del self.hist_dN
        for n in range(1, self.n_order):
            del self.hist_vncos[n]
            del self.hist_vnsin[n]

    # ---------------------------------------------------------------
    # Function to fill 2D histogram with Qn vector results for a single event
    # ---------------------------------------------------------------
    def fill_histogram_from_qnvector(self, qnvector_results):

        # Loop over (N_pT X N_rapidity) entries of each event
        # Note: only results for pid=9999 are read and passed to qnvector_results here
        for result in qnvector_results:

            pt = result[1]
            y = result[3]

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

            self.hist_dN.Fill(pt, y, dN)

    # ---------------------------------------------------------------
    # Calculate necessary quantities using the histogram of Qn vector results for a single event
    # ---------------------------------------------------------------
    def process_histogram(self, event_id):
        """
        This function processes 2D histograms (in pT * rapidity bins) of a single event
        """
        pt_values = []
        dN_values = []
        vncos_values = {n: [] for n in range(1, self.n_order)}
        vnsin_values = {n: [] for n in range(1, self.n_order)}

        # for psi_n calculation
        vncos_sum = {n: 0.0 for n in range(1, self.n_order)}
        vnsin_sum = {n: 0.0 for n in range(1, self.n_order)}

        for bin_x in range(1, self.hist_dN.GetNbinsX() + 1):
            for bin_y in range(1, self.hist_dN.GetNbinsY() + 1):
                pt = self.hist_dN.GetXaxis().GetBinCenter(bin_x)
                y = self.hist_dN.GetYaxis().GetBinCenter(bin_y)

                if self.eta_min <= y <= self.eta_max:
                    dN = self.hist_dN.GetBinContent(bin_x, bin_y)

                    pt_values.append(pt)
                    dN_values.append(dN)

                    for n in range(1, self.n_order):
                        vncos = self.hist_vncos[n].GetBinContent(bin_x, bin_y, n)
                        vnsin = self.hist_vnsin[n].GetBinContent(bin_x, bin_y, n)

                        vncos_values[n].append(vncos)
                        vnsin_values[n].append(vnsin)

        # Ensure arrays are sorted and non-empty before interpolation
        if len(pt_values) == 0 or len(dN_values) == 0:
            raise ValueError("No valid entries for interpolation.")

        # Convert lists to arrays for interpolation
        pt_values = np.array(pt_values)
        dN_values = np.array(dN_values)

        # Calculate N_tot
        N_tot = np.sum(dN_values)

        # Calculate mean pT, weighted by dN
        if N_tot > 0:
            mean_pt = np.sum(pt_values * dN_values) / N_tot
        else:
            mean_pt = 0.0  # Handle the case where no entries are valid

        # Calculate dN_ch/deta: N_tot to be divided by oversample number and rapidity width
        N_tot = N_tot / self.n_oversample / (self.eta_max - self.eta_min)

        vn_real_dict = {}
        vn_imag_dict = {}
        psi_n_dict = {}
        for n in range(1, self.n_order):
            vncos_values[n] = np.array(vncos_values[n])
            vnsin_values[n] = np.array(vnsin_values[n])

            # Now sum the values for psi_n calculation
            vncos_sum = np.sum(vncos_values[n])
            vnsin_sum = np.sum(vnsin_values[n])

            # Calculate psi_n
            psi_n = (1.0 / n) * np.arctan2(vnsin_sum, vncos_sum)
            psi_n_dict[n] = psi_n

        # Call the integrated vn calculation function for all n; pT cut is done inside the function
        vn_real_dict, vn_imag_dict = self.calculate_vn_single_event(pt_values, dN_values, vncos_values, vnsin_values)

        return N_tot, mean_pt, psi_n_dict, vn_real_dict, vn_imag_dict

    # ---------------------------------------------------------------
    # Calculate vn for a single event
    # ---------------------------------------------------------------
    def calculate_vn_single_event(self, pt_values, dN_values, vncos_values, vnsin_values):

        npT = 50

        pT_inte_array = np.linspace(self.pT_low, self.pT_high, npT)

        # Interpolating dN values in log space
        dN_interp = np.exp(np.interp(pT_inte_array, pt_values, np.log(dN_values + 1e-30)))

        vn_real_dict = {}
        vn_imag_dict = {}

        # Loop over all n
        for n in range(1, self.n_order):
            vn_real_interp = np.interp(pT_inte_array, pt_values, vncos_values[n])
            vn_imag_interp = np.interp(pT_inte_array, pt_values, vnsin_values[n])

            vn_real_inte = np.sum(vn_real_interp * dN_interp) / np.sum(dN_interp)
            vn_imag_inte = np.sum(vn_imag_interp * dN_interp) / np.sum(dN_interp)

            # Store the result for this n
            vn_real_dict[n] = vn_real_inte
            vn_imag_dict[n] = vn_imag_inte

        return vn_real_dict, vn_imag_dict

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
