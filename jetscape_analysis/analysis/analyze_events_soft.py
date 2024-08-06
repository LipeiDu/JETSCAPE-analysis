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

sys.path.append('.')

from jetscape_analysis.base import common_base

################################################################
class AnalyzeJetscapeEvents_Base(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file="", input_dir="", output_dir="", **kwargs):

        super(AnalyzeJetscapeEvents_Base, self).__init__(**kwargs)
        self.config_file = config_file
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not self.input_dir.endswith("/"):
            self.input_dir = self.input_dir + "/"

        # Create output dir
        if not self.output_dir.endswith("/"):
            self.output_dir = self.output_dir + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
        self.write_Qn_histograms = config['write_Qn_histograms']
        
        self.event_id = 0

        self.eta_min = config['eta_min']
        self.eta_max = config['eta_max']

        self.n_pt_bins = config['n_pt_bins']
        self.pt_min = config['pt_min']
        self.pt_max = config['pt_max']
        self.n_y_bins = config['n_y_bins']
        self.y_min = config['y_min']
        self.y_max = config['y_max']
        self.n_order = config['n_order']

    # ---------------------------------------------------------------
    # Main processing function
    # ---------------------------------------------------------------
    def analyze_jetscape_events(self):
  
        # Create outputDir
        if not self.output_dir.endswith("/"):
            self.output_dir = self.output_dir + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Read JETSCAPE Qn vector output and write histograms to ROOT file
        self.input_file_Qnvector = os.path.join(self.input_dir, 'event_QnVector.dat')
        self.run_jetscape_analysis()

    # ---------------------------------------------------------------
    # Main processing function for events in a single event_QnVector.dat
    # ---------------------------------------------------------------
    def run_jetscape_analysis(self):
        # Create reader class
        all_events, total_events = self.reader_ascii(self.input_file_Qnvector)
        print("total_events", total_events)

        # Open the ROOT file
        output_file = ROOT.TFile(self.output_dir + "Qn_vector_results.root", "RECREATE")

        # Create a histogram to store the event plane angles
        hist_event_plane_angles = ROOT.TH1F("hist_event_plane_angles", 
                                            "Event Plane Angles; Event ID; Psi_2", 
                                            total_events + 1,  # Bin count to include all events
                                            0, total_events + 1)  # Edge to include the last event ID

        # Iterate through events
        for event_id, event in all_events.items():
            print(f"Processing event_id: {event_id}, number of entries: {len(event)}")

            # Initialize output objects
            self.initialize_output_objects(event_id)
            # Fill histogram
            self.fill_histogram_from_qnvector(event)

            # Process histogram data directly and get event plane angle
            psi_2 = self.process_histogram(event_id)

            # Fill the event plane angle histogram
            # Use event_id as the bin index
            hist_event_plane_angles.Fill(event_id + 0.5, psi_2)  # Adding 0.5 to align with bin center

            if self.write_Qn_histograms:
                # Write analysis task output to ROOT file
                self.write_histogram_to_file(event_id, output_file)

        # Write the event plane angle histogram to the ROOT file
        hist_event_plane_angles.Write()

        # Print the histogram contents
        # Access histogram from the ROOT file to print its contents
        hist_event_plane_angles = output_file.Get("hist_event_plane_angles")
        
        print("Histogram contents:")
        for bin_num in range(1, hist_event_plane_angles.GetNbinsX() + 1):
            bin_center = hist_event_plane_angles.GetBinCenter(bin_num)
            bin_content = hist_event_plane_angles.GetBinContent(bin_num)
            print(f"Event ID: {int(bin_center)}, Psi_2: {bin_content}")

        # Close the ROOT file if it was opened
        if self.write_Qn_histograms:
            output_file.Close()

    # ---------------------------------------------------------------
    # Initialize output objects
    # ---------------------------------------------------------------  
    def initialize_output_objects(self, event_id):
        # Initialize histograms for each event
        self.hist_dN  = ROOT.TH2F(f"hist_dN_event_{event_id}", f"dN; pT; y; Event {event_id}", self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
        self.hist_vncos = ROOT.TH2F(f"hist_vncos_event_{event_id}", f"vncos; pT; y; Event {event_id}", self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
        self.hist_vnsin = ROOT.TH2F(f"hist_vnsin_event_{event_id}", f"vnsin; pT; y; Event {event_id}", self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)
    
    # ---------------------------------------------------------------
    # Function to fill 2D histogram with Qn vector results
    # ---------------------------------------------------------------
    def fill_histogram_from_qnvector(self, qnvector_results):

        for result in qnvector_results:

            pid = result[0]
            pt = result[1]
            pt_err = result[2]
            y = result[3]
            y_err = result[4]
            et = result[5]

            # n = 1 is skipped
            vncos = result[12]
            vncos_err = result[13]
            vnsin = result[14]
            vnsin_err = result[15]

            dN = result[8+4*(self.n_order-1)]

            # Fill the histograms with errors
            self.hist_dN.Fill(pt, y, dN)

            self.hist_vncos.Fill(pt, y, vncos)
            self.hist_vncos.SetBinError(self.hist_vncos.FindBin(pt, y), vncos_err)

            self.hist_vnsin.Fill(pt, y, vnsin)
            self.hist_vnsin.SetBinError(self.hist_vnsin.FindBin(pt, y), vnsin_err)

    # ---------------------------------------------------------------
    # Calculate necessary quantities using the histogram of Qn vector results
    # ---------------------------------------------------------------
    def process_histogram(self, event_id):
        numerator = 0.0
        denominator = 0.0

        for bin_x in range(1, self.hist_dN.GetNbinsX() + 1):
            for bin_y in range(1, self.hist_dN.GetNbinsY() + 1):
                pt = self.hist_dN.GetXaxis().GetBinCenter(bin_x)
                y = self.hist_dN.GetYaxis().GetBinCenter(bin_y)

                if self.eta_min <= y <= self.eta_max:
                    dN = self.hist_dN.GetBinContent(bin_x, bin_y)
                    vncos = self.hist_vncos.GetBinContent(bin_x, bin_y)
                    vnsin = self.hist_vnsin.GetBinContent(bin_x, bin_y)

                    numerator += vnsin * dN
                    denominator += vncos * dN

        if denominator != 0:
            psi_2 = 0.5 * ROOT.TMath.ATan2(numerator, denominator)
        else:
            psi_2 = None

        print(f"Event: {event_id}, Event Plane Angle (psi_2): {psi_2}")
        return psi_2

    # ---------------------------------------------------------------
    # Save all ROOT histograms and trees to file
    # ---------------------------------------------------------------
    def write_histogram_to_file(self, event_id, output_file):
        output_file.cd()
        self.hist_dN.Write()
        self.hist_vncos.Write()
        self.hist_vnsin.Write()

    # ---------------------------------------------------------------
    # Reader of the Qn vector file
    # ---------------------------------------------------------------
    def reader_ascii(self, input_file):
        """
        This function reads Qn vector data from the specified file.
        Each event is stored in a dictionary with the event number as the key.
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
        "--inputDir",
        action="store",
        type=str,
        metavar="inputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Input directory containing JETSCAPE output files",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        action="store",
        type=str,
        metavar="outputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Output directory for output to be written to",
    )

    # Parse the arguments
    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File "{0}" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # If invalid inputDir is given, exit
    if not os.path.exists(args.inputDir):
        print('File "{0}" does not exist! Exiting!'.format(args.inputDir))
        sys.exit(0)

    analysis = AnalyzeJetscapeEvents_Base(config_file=args.configFile, input_dir=args.inputDir, output_dir=args.outputDir)
    analysis.analyze_jetscape_events()


