#!/usr/bin/env python3

""" Skim a large ascii file and split into smaller files
"""

from __future__ import print_function

# General
import os
import sys
import argparse
import numpy as np
import ROOT

sys.path.append('.')

from jetscape_analysis.analysis.reader import parse_ascii
from jetscape_analysis.base import common_base

################################################################
class SkimAscii(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, input_file="", output_dir="", events_per_chunk=50000, soft_analysis_results_file=None, **kwargs):
        super(SkimAscii, self).__init__(**kwargs)
        self.input_file = input_file
        self.output_dir = output_dir

        self.events_per_chunk = events_per_chunk

        self.soft_analysis_results_file = soft_analysis_results_file
        self.load_soft_analysis_results = self.load_soft_analysis_results() if soft_analysis_results_file else None

    # ---------------------------------------------------------------
    # Load custom event plane angles from a ROOT file
    # ---------------------------------------------------------------
    # Load custom event plane angles and v2 magnitudes from a ROOT file
    def load_soft_analysis_results(self):
        soft_analysis_results = {}

        # Open the ROOT file
        root_file = ROOT.TFile(self.soft_analysis_results_file, "READ")

        # Retrieve the necessary histograms
        hist_psi = root_file.Get("hist_psi_n2")  # Psi_2 for n=2
        hist_Qn_ref_real = root_file.Get("hist_Qn_ref_real_n2")  # Qn_ref real for n=2
        hist_Qn_ref_imag = root_file.Get("hist_Qn_ref_imag_n2")  # Qn_ref imaginary for n=2
        hist_N_Qn_ref = root_file.Get("hist_N_Qn_ref")           # N_Qn_ref for normalization

        if hist_psi and hist_Qn_ref_real and hist_Qn_ref_imag and hist_N_Qn_ref:
            # Loop over the events in the histograms
            for event_id in range(1, hist_psi.GetNbinsX() + 1):
                # Get the event plane angle (Psi_2) for n=2
                psi_2 = hist_psi.GetBinContent(event_id)
                
                # Get the real and imaginary parts of Qn_ref for n=2
                Qn_ref_real = hist_Qn_ref_real.GetBinContent(event_id)
                Qn_ref_imag = hist_Qn_ref_imag.GetBinContent(event_id)
                N_Qn_ref = hist_N_Qn_ref.GetBinContent(event_id)

                # Calculate the magnitude of Qn_ref (v2), taking normalization into account
                if N_Qn_ref > 0:  # Avoid division by zero
                    v2 = np.sqrt(Qn_ref_real**2 + Qn_ref_imag**2) / N_Qn_ref
                else:
                    v2 = 0.0

                # Store the event data
                soft_analysis_results[event_id] = (psi_2, v2)

        # Close the ROOT file
        root_file.Close()

        return soft_analysis_results

    # ---------------------------------------------------------------
    # Main processing function for a single pt-hat bin
    # ---------------------------------------------------------------
    def skim(self):

        # Create reader class for each chunk of events, and iterate through each chunk
        # The parser returns an awkward array of events
        parse_ascii.parse_to_parquet(base_output_filename=self.output_dir,
                                     store_only_necessary_columns=True,
                                     input_filename=self.input_file,
                                     events_per_chunk=self.events_per_chunk,
                                     load_soft_analysis_results=self.load_soft_analysis_results)

##################################################################
if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Generate JETSCAPE events")
    parser.add_argument(
        "-i",
        "--inputFile",
        action="store",
        type=str,
        metavar="inputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/test.out",
        help="Input directory containing JETSCAPE output files",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        action="store",
        type=str,
        metavar="outputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Output directory and filename template for output to be written to",
    )
    parser.add_argument(
        "-n",
        "--nEventsPerFile",
        action="store",
        type=int,
        metavar="nEventsPerFile",
        default=50000,
        help="Number of events to store in each parquet file",
    )
    parser.add_argument(
        "-e",
        "--softAnalysisResultsFile",
        action="store",
        type=str,
        metavar="softAnalysisResultsFile",
        default=None,
        help="File containing custom event plane angles",
    )

    # Parse the arguments
    args = parser.parse_args()

    # If invalid inputDir is given, exit
    if not os.path.exists(args.inputFile):
        print('File "{0}" does not exist! Exiting!'.format(args.inputFile))
        sys.exit(0)

    analysis = SkimAscii(input_file=args.inputFile, output_dir=args.outputDir, events_per_chunk=args.nEventsPerFile,
        soft_analysis_results_file=args.softAnalysisResultsFile)
    analysis.skim()
