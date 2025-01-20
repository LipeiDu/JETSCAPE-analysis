#!/usr/bin/env python3

from __future__ import print_function

# General
import os
import subprocess
import sys
import tqdm
import yaml
import awkward as ak
from pathlib import Path

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
    def __init__(self, config_file="", input_file="", output_dir="", **kwargs):

        super(AnalyzeJetscapeEvents_Base, self).__init__(**kwargs)

        # Initialize input, output, and config paths
        self.config_file = config_file
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        # Generate the output file path by replacing ".parquet" with ".root"
        self.output_file = self.output_dir / self.input_file.name.replace(".parquet", ".root")

        # Read run info from the corresponding YAML file
        self._Qn_vector_path = self.input_file
        _run_number = self._Qn_vector_path.stem.split("_")[2]
        _file_index = int(self._Qn_vector_path.name.split('_')[4])
        run_info_path = self._Qn_vector_path.parent / f"{_run_number}_info.yaml"

        if not run_info_path.exists():
            raise FileNotFoundError(f"Run info file not found: {run_info_path}")

        # Read and parse the YAML file
        with open(run_info_path, 'r') as f:
            _run_info = yaml.safe_load(f)

        # Assign parameters from the YAML file
        self.centrality = _run_info.get("centrality", None)
        self.soft_sector_execution_type = _run_info.get("soft_sector_execution_type", None)
        self.n_oversample = _run_info.get("number_of_repeated_sampling", None)

        self.initialize_config()

        self.qnvector_histograms = {}

    # ---------------------------------------------------------------
    # Initialize config file into class members
    # ---------------------------------------------------------------
    def initialize_config(self):

        # Read config file
        with open(self.config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)

        # parameters of Qn vector writer in JETSCAPE
        self.n_pt_bins = self.config['n_pt_bins']
        self.pt_min = self.config['pt_min']
        self.pt_max = self.config['pt_max']
        self.n_y_bins = self.config['n_y_bins']
        self.y_min = self.config['y_min']
        self.y_max = self.config['y_max']
        self.norder = self.config['norder']

        # select charged particles in the Qn vector file
        self.selected_pid = self.config['selected_pid']

        # pT arrays for interpolation in flow calculation
        self.pT_low = self.config['pT_low']
        self.pT_high = self.config['pT_high']
        self.npT = self.config['npT']

    # ---------------------------------------------------------------
    # Main processing function
    # ---------------------------------------------------------------
    def analyze_jetscape_events(self):

        print(f"Analyze the Qn vector of the soft sector ...")

        # Read JETSCAPE Qn vector output and write histograms to ROOT file
        self.run_jetscape_analysis()

    # ---------------------------------------------------------------
    # Main processing function for events in a single event_QnVector.dat
    # ---------------------------------------------------------------
    def run_jetscape_analysis(self):
        # Read Qn vector data
        all_events, total_events = self.reader_parquet(self.input_file)

        print(f"Total events in centrality {self.centrality}: {total_events}")
        min_event_id = min(all_events.keys())
        max_event_id = max(all_events.keys())

        # Open the ROOT file
        output_file = ROOT.TFile(str(self.output_file), "RECREATE")

        # Prepare result histograms for all observables before processing events
        result_histograms = {}  # Store histograms for each observable
        kinematic_cuts = {}  # Store kinematic cuts for each observable

        observable_type = 'hadron_correlations'
        # Loop through each observable within the observable type
        for observable, block in self.config[observable_type].items():
            if "v2" in observable:  # only process "v2" observables
                # Loop through methods (EP, SP, etc.)
                for method, method_block in block.items():
                    if not self.centrality_accepted(method_block['centrality']):
                        continue  # Skip if centrality doesn't match

                    # Extract kinematic cuts
                    eta_cut = method_block['eta_cut']
                    eta_min_ref = method_block['eta_min_ref']
                    eta_max_ref = method_block['eta_max_ref']

                    kinematic_cuts[(observable_type, observable, method, tuple(self.centrality))] = (eta_cut, eta_min_ref, eta_max_ref)

                    # Initialize histograms for the observable
                    result_histograms[(observable_type, observable, method, tuple(self.centrality))] = self.initialize_result_histograms(
                        observable_type, observable, method, self.centrality, min_event_id, max_event_id
                    )

        # Process events and loop over observables
        for event_id, event in all_events.items():
            if event_id % 1000 == 0:
                print(f"Processing event_id: {event_id}, entries: {len(event)}")

            # Initialize and fill Qn vector histograms of a single event (common for all observables)
            self.initialize_qnvector_histogram()

            self.fill_parquet_qnvector_histogram(event)

            # Loop over observables for event-specific processing
            for (observable_type, observable, method, centrality), histograms in result_histograms.items():
                print()
                print(f'Histogram {observable_type} observables...')

                # Get the kinematic cuts for this observable
                eta_cut, eta_min_ref, eta_max_ref = kinematic_cuts[(observable_type, observable, method, centrality)]

                # Apply observable-specific processing with the correct kinematic cuts
                results = self.process_qnvector_histogram(event_id, eta_cut, eta_min_ref, eta_max_ref)
                if results:
                    self.fill_result_histograms(histograms, event_id, results)

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
                                                                   self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max) for n in range(1, self.norder)}
            self.qnvector_histograms['hist_vnsin'] = {n: ROOT.TH2F(f"hist_vnsin_n{n}", f"vnsin (n={n}); pT; y;", 
                                                                   self.n_pt_bins, self.pt_min, self.pt_max, self.n_y_bins, self.y_min, self.y_max) for n in range(1, self.norder)}

    # ---------------------------------------------------------------
    # Function to fill 2D histogram with Qn vector results for a single event
    # ---------------------------------------------------------------
    def fill_parquet_qnvector_histogram(self, qnvector_results):
        """
        In qnvector histograms, only dNdpTdy is averaged over oversamples; the other ones are summed over oversamples.
        """
        # Loop over (N_pT X N_rapidity) entries of each event
        # Note: only results for pid=9999 are read and passed to qnvector_results here

        self.hist_dN = self.qnvector_histograms['hist_dN']
        self.hist_vncos = self.qnvector_histograms['hist_vncos']
        self.hist_vnsin = self.qnvector_histograms['hist_vnsin']

        for result in qnvector_results:
            pt = result.get("pT", None)
            y = result.get("y", None)
            dN = result.get("dN", None)

            if dN is None or dN <= 0 or pt is None or y is None:
                continue

            self.hist_dN.Fill(pt, y, dN)

            vn_cos_list = result.get("vn_cos", [])
            vn_sin_list = result.get("vn_sin", [])

            for n in range(1, self.norder):
                if n - 1 < len(vn_cos_list) and n - 1 < len(vn_sin_list):
                    vncos = vn_cos_list[n - 1]
                    vnsin = vn_sin_list[n - 1]
                    self.hist_vncos[n].Fill(pt, y, vncos)
                    self.hist_vnsin[n].Fill(pt, y, vnsin)

    # ---------------------------------------------------------------
    # Clear output objects after processing each event
    # ---------------------------------------------------------------
    def clear_qnvector_histograms(self):
        self.qnvector_histograms['hist_dN'].Reset()
        for n in range(1, self.norder):
            self.qnvector_histograms['hist_vncos'][n].Reset()
            self.qnvector_histograms['hist_vnsin'][n].Reset()

    # ---------------------------------------------------------------
    # Calculate necessary quantities using the histogram of Qn vector results for a single event
    # --------------------------------------------------------------
    def process_qnvector_histogram(self, event_id, eta_cut, eta_min_ref, eta_max_ref):
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
        vncos_ref_values = {n: [] for n in range(1, self.norder)}
        vnsin_ref_values = {n: [] for n in range(1, self.norder)}

        # Loop over pT bins
        for bin_x in range(1, self.hist_dN.GetNbinsX() + 1):

            pt = self.hist_dN.GetXaxis().GetBinCenter(bin_x)
            dN_sum, dN_ref_sum = 0.0, 0.0
            vncos_ref_sum = {n: 0.0 for n in range(1, self.norder)}
            vnsin_ref_sum = {n: 0.0 for n in range(1, self.norder)}

            # Loop over rapidity bins
            for bin_y in range(1, self.hist_dN.GetNbinsY() + 1):
                y = self.hist_dN.GetYaxis().GetBinCenter(bin_y)
                dN = self.hist_dN.GetBinContent(bin_x, bin_y)

                # midrapidity multiplicity
                if abs(y) < eta_cut:
                    dN_sum += dN

                # reference particles
                if eta_min_ref <= y <= eta_max_ref:
                    dN_ref_sum += dN

                    # Sum vncos and vnsin for each harmonic n, weighted by dN for the reference particles
                    for n in range(1, self.norder):
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
                for n in range(1, self.norder):
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
        dNdeta = N_tot / self.n_oversample / (2 * eta_cut)

        # III. Calculate quantities for flow calculations

        pt_ref_values = np.array(pt_ref_values)
        dN_ref_values = np.array(dN_ref_values)

        for n in range(1, self.norder):
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

        for iorder in range(1, self.norder):
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
    def initialize_result_histograms(self, observable_type, observable, method, centrality, min_event_id, max_event_id):
        base_name = f"h_{observable_type}_{observable}_{method}_{centrality}"
        histograms = {}

        # Use min_event_id and max_event_id for the x-axis range
        histograms['hist_dNdeta'] = ROOT.TH1F(f"{base_name}_dNchdeta", "Multiplicity; Event ID; Nch", max_event_id - min_event_id + 1, min_event_id - 0.5, max_event_id + 0.5)
        histograms['hist_N_Qn_ref'] = ROOT.TH1F(f"{base_name}_N_Qn_ref", "N_Qn_ref; Event ID; N_Qn_ref", max_event_id - min_event_id + 1, min_event_id - 0.5, max_event_id + 0.5)

        histograms['hist_Qn_ref_real'] = {
            n: ROOT.TH1F(f"{base_name}_Qn_ref_real_n{n}", f"Qn ref real (n={n}); Event ID; Qn_ref", max_event_id - min_event_id + 1, min_event_id - 0.5, max_event_id + 0.5)
            for n in range(1, self.norder)
        }

        histograms['hist_Qn_ref_imag'] = {
            n: ROOT.TH1F(f"{base_name}_Qn_ref_imag_n{n}", f"Qn ref imag (n={n}); Event ID; Qn_ref", max_event_id - min_event_id + 1, min_event_id - 0.5, max_event_id + 0.5)
            for n in range(1, self.norder)
        }

        return histograms

    # ---------------------------------------------------------------
    # Fill histograms of results
    # ---------------------------------------------------------------
    def fill_result_histograms(self, histograms, event_id, results):
        histograms['hist_dNdeta'].Fill(event_id, results["dNdeta"])
        histograms['hist_N_Qn_ref'].Fill(event_id, results["N_Qn_ref"])
        for n in range(1, self.norder):
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
    # Reader of the Qn vector Parquet file
    # ---------------------------------------------------------------
    def reader_parquet(self, input_file):
        """
        This function reads Qn vector data from a Parquet file.
        Each event is stored as a list of dictionaries, grouped by `event_ID`.
        """
        print(f"Reading Parquet file: {input_file}")

        # Read the entire Parquet file into an awkward array
        data = ak.from_parquet(input_file)

        # Convert awkward array into a dictionary grouped by `event_ID`
        all_events = {}
        for event_id in np.unique(ak.to_numpy(data["event_ID"])):
            event_mask = data["event_ID"] == event_id
            event_data = data[event_mask].to_list()  # Convert to Python list for compatibility
            all_events[int(event_id)] = event_data

        total_events = len(all_events)
        return all_events, total_events

    # ---------------------------------------------------------------
    # Check if event centrality is within observable's centrality
    # ---------------------------------------------------------------
    def centrality_accepted(self, observable_centrality_list):

        for observable_centrality in observable_centrality_list:
            if self.centrality[0] >= observable_centrality[0]:
                if self.centrality[1] <= observable_centrality[1]:
                    return True
        return False

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
        "--outputDir",
        action="store",
        type=str,
        metavar="outputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Output directory for output to be written to",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the analysis
    analysis = AnalyzeJetscapeEvents_Base(
        config_file=args.configFile,
        input_file=args.inputFilename,
        output_dir=args.outputDir
    )
    analysis.analyze_jetscape_events()
