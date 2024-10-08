#!/usr/bin/env python3

""" Skim a large ascii file and split into smaller files
"""

from __future__ import print_function

# General
import os
import sys
import argparse
# import uproot
import ROOT

sys.path.append('.')

from jetscape_analysis.analysis.reader import parse_ascii
from jetscape_analysis.base import common_base

################################################################
class SkimAscii(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, input_file="", output_dir="", events_per_chunk=50000, event_plane_angle_file=None, **kwargs):
        super(SkimAscii, self).__init__(**kwargs)
        self.input_file = input_file
        self.output_dir = output_dir

        self.event_id = 0
        self.events_per_chunk = events_per_chunk

        self.event_plane_angle_file = event_plane_angle_file
        self.custom_event_plane_angles = self.load_custom_event_plane_angles() if event_plane_angle_file else None

    # ---------------------------------------------------------------
    # Load custom event plane angles from a ROOT file
    # ---------------------------------------------------------------
    # Load custom event plane angles and v2 magnitudes from a ROOT file
    def load_custom_event_plane_angles(self):
        event_plane_data = {}
        root_file = ROOT.TFile(self.event_plane_angle_file, "READ")
        hist_psi_2 = root_file.Get("hist_event_plane_angles")
        hist_v2 = root_file.Get("hist_v2_magnitudes")
        
        if hist_psi_2 and hist_v2:
            for bin_num in range(1, hist_psi_2.GetNbinsX() + 1):
                event_id = bin_num - 1  # Assuming event ID starts from 0
                psi_2 = hist_psi_2.GetBinContent(bin_num)
                v2 = hist_v2.GetBinContent(bin_num)
                event_plane_data[event_id] = (psi_2, v2)  # Store both Psi_2 and v2 for each event
        root_file.Close()
        return event_plane_data

    # def load_custom_event_plane_angles(self):
    #     event_plane_data = {}
        
    #     # Open the ROOT file using uproot
    #     with uproot.open(self.event_plane_angle_file) as root_file:
    #         # Access the Psi_2 and v2 histograms
    #         hist_psi_2 = root_file["hist_event_plane_angles"]
    #         hist_v2 = root_file["hist_v2_magnitudes"]
            
    #         # Convert histograms to numpy arrays
    #         psi_2_contents, bin_edges = hist_psi_2.to_numpy()
    #         v2_contents, _ = hist_v2.to_numpy()
            
    #         # Calculate the bin centers
    #         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
    #         # Iterate through the bins and store both psi_2 and v2
    #         for i, psi_2 in enumerate(psi_2_contents):
    #             event_id = int(bin_centers[i])  # Use bin centers for event ID
    #             v2 = v2_contents[i]  # Corresponding v2 magnitude
    #             event_plane_data[event_id] = (psi_2, v2)
    #             print(f"Event ID: {event_id}, Psi_2: {psi_2}, v2: {v2}")
                
    #     return event_plane_data

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
                                     custom_event_plane_angles=self.custom_event_plane_angles)

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
        "--eventPlaneAngleFile",
        action="store",
        type=str,
        metavar="eventPlaneAngleFile",
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
        event_plane_angle_file=args.eventPlaneAngleFile)
    analysis.skim()
