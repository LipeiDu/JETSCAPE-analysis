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
        
        self.event_id = 0

        self.n_pt_bins = config['n_pt_bins']
        self.pt_min = config['pt_min']
        self.pt_max = config['pt_max']
        self.n_y_bins = config['n_y_bins']
        self.y_min = config['y_min']
        self.y_max = config['y_max']

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
        output_file = ROOT.TFile(self.output_dir + "results.root", "RECREATE")

        # Iterate through events
        for event_id, event in all_events.items():
            print(f"Processing event_id: {event_id}, number of entries: {len(event)}")

            # Initialize output objects
            self.initialize_output_objects(event_id)
            # Fill histogram
            self.fill_histogram_from_qnvector(event)
            # Write analysis task output to ROOT file
            self.write_histogram_to_file(event_id, output_file)

        # Close the ROOT file
        output_file.Close()

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

    # ---------------------------------------------------------------
    # Initialize output objects
    # ---------------------------------------------------------------
    def initialize_output_objects(self, event_id):
        # Initialize the histogram for each event
        self.hist = ROOT.TH2F(f"hist_dNdpTdy_event_{event_id}", f"dNdpTdy; pT; y; Event {event_id}", self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max)    
    
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
            dNdpTdy = result[6]
            dNdpTdy_err = result[7]
            vncos = result[8]
            vncos_err = result[9]
            vnsin = result[10]
            vnsin_err = result[11]
            dN = result[12]

            # Fill the histogram
            self.hist.Fill(pt, y, dNdpTdy)

    # ---------------------------------------------------------------
    # Save all ROOT histograms and trees to file
    # ---------------------------------------------------------------
    def write_histogram_to_file(self, event_id, output_file):
        # Write the histogram to the file
        output_file.cd()
        self.hist.Write()


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


