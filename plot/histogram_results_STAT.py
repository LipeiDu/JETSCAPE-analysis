"""
  macro for plotting analyzed jetscape events
  """

# This script plots histograms created in the analysis of Jetscape events
#
# Author: James Mulligan (james.mulligan@berkeley.edu)

# General
import os
import sys
import yaml
import argparse

# Data analysis and plotting
import ROOT
import numpy as np
import pandas as pd

# Base class
sys.path.append('.')
from jetscape_analysis.base import common_base
from plot import plot_results_STAT_utils

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class HistogramResults(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file='', input_file='', output_dir='', **kwargs):
        super(HistogramResults, self).__init__(**kwargs)
        self.input_file = input_file
        self.output_dir = output_dir

        self.plot_utils = plot_results_STAT_utils.PlotUtils()

        #------------------------------------------------------
        # Check whether pp or AA
        if 'PbPb' in self.input_file or 'AuAu' in self.input_file:
            self.is_AA = True
        else:
            self.is_AA = False

        #------------------------------------------------------
        # Read config file
        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.sqrts = self.config['sqrt_s']
        self.power = self.config['power']
        self.pt_ref = self.config['pt_ref']

        self.norder = self.config['norder']

        # If AA, set different options for hole subtraction treatment
        if self.is_AA:
            self.jet_collection_labels = self.config['jet_collection_labels']
        else:
            self.jet_collection_labels = ['']

        #------------------------------------------------------
        # Read input file
        self.observables_df = pd.read_parquet(self.input_file)
        # It's possible that there are no entries in observables_df, so we use get here and
        # return an empty list so that the code doesn't crash. In this case, we'll have to
        # bail out immediately when processing.
        # NOTE: We should still write an empty ROOT file when we go to run the histogramming.
        #       Otherwise the steering will fail because it will see the missing file and think
        #       that the jobs failed.
        self.weights = self.observables_df.get('event_weight', [])
        self.pt_hat = self.observables_df.get('pt_hat', [])
        self.soft_particle_v2 = self.observables_df.get('soft_v2', [])

        #------------------------------------------------------
        # Read cross-section file
        cross_section_file = 'cross_section'.join(self.input_file.rsplit('observables', 1))
        cross_section_df = pd.read_parquet(cross_section_file)
        self.cross_section = cross_section_df['cross_section'][0]
        self.cross_section_error = cross_section_df['cross_section_error'][0]
        self.n_events_generated = cross_section_df['n_events'][0]
        self.sum_weights = cross_section_df['weight_sum'][0]
        self.soft_v2_average = cross_section_df['soft_v2_average'][0]
        if self.is_AA:
            self.centrality = [ int(cross_section_df['centrality_min'][0]), int(cross_section_df['centrality_max'][0]) ]
            self.observable_centrality_list = []

        print(f'xsec: {self.cross_section}')
        print(f'weights: {self.sum_weights}')

        #------------------------------------------------------
        # Create output list to store histograms
        self.output_list = []

        #print(self)
        #print(f'keys: {self.observables_df.keys()}')

    #-------------------------------------------------------------------------------------------
    # Main function
    #-------------------------------------------------------------------------------------------
    def histogram_results(self):

        if len(self.observables_df) != 0:
            # Hadron histograms
            self.histogram_hadron_observables(observable_type='hadron')

            self.histogram_hadron_correlation_observables(observable_type='hadron_correlations')

            # Jet histograms: loop through different hole subtraction treatments
            for jet_collection_label in self.jet_collection_labels:

                self.histogram_jet_observables(observable_type='inclusive_chjet', jet_collection_label=jet_collection_label)

                if 'inclusive_jet' in self.config:
                    self.histogram_jet_observables(observable_type='inclusive_jet', jet_collection_label=jet_collection_label)

                if 'semi_inclusive_chjet' in self.config:
                    self.histogram_semi_inclusive_chjet_observables(observable_type='semi_inclusive_chjet', jet_collection_label=jet_collection_label)

                if 'dijet' in self.config:
                    self.histogram_jet_observables(observable_type='dijet', jet_collection_label=jet_collection_label)
        else:
            print("\tWARNING: There are no entries in the observables df. Will only write event level QA.")

        # QA histograms
        # NOTE: These histograms still work even if there are no observables (well, at least many of them,
        #       but not all), so we might, as well collect event level QA.
        self.histogram_event_qa()

        # Write output to ROOT file
        self.write_output_objects()

    #-------------------------------------------------------------------------------------------
    # Create event-level histograms
    #-------------------------------------------------------------------------------------------
    def histogram_event_qa(self):

        #---------------------------------
        # Save xsec and sum_weights for normalization and uncertainty

        # For AA, we need to loop through all centrality bins and save the xsec and weight_sum for each.
        # This is needed so that when we merge histograms of different centralities, we retain access to
        #   the normalization factors for each observable's centrality bin
        if self.is_AA:
            for centrality in self.observable_centrality_list:
                if self.centrality_accepted(centrality):

                    h = ROOT.TH1F(f'h_xsec_{centrality}', f'h_xsec_{centrality}', 1, 0, 1)
                    h.SetBinContent(1, self.cross_section)
                    self.output_list.append(h)

                    h = ROOT.TH1F(f'h_xsec_error_{centrality}', f'h_xsec_error_{centrality}', 1, 0, 1)
                    h.SetBinContent(1, self.cross_section_error)
                    self.output_list.append(h)

                    # Save sum of weights (effective number of events), in order to keep track of normalization uncertainty
                    h = ROOT.TH1F(f'h_weight_sum_{centrality}', f'h_weight_sum_{centrality}', 1, 0, 1)
                    h.SetBinContent(1, self.sum_weights)
                    self.output_list.append(h)

                    # Save event weights
                    bins = np.logspace(np.log10( np.power(self.pt_ref/(self.sqrts/2),self.power) ), np.log10( np.power(self.pt_ref/2,self.power)  ), 10000)
                    h = ROOT.TH1F(f'h_weights_{centrality}', f'h_weights_{centrality}', bins.size-1, bins)
                    for weight in self.weights:
                        h.Fill(weight)
                    self.output_list.append(h)

                    # Save soft particle v2 average
                    h = ROOT.TH1F(f'h_soft_v2_average_{centrality}', f'h_soft_v2_average_{centrality}', 1, 0, 1)
                    h.SetBinContent(1, self.soft_v2_average)
                    self.output_list.append(h)

        # For pp, we can just save a single histogram for each
        else:

            h = ROOT.TH1F('h_xsec', 'h_xsec', 1, 0, 1)
            h.SetBinContent(1, self.cross_section)
            self.output_list.append(h)

            h = ROOT.TH1F('h_xsec_error', 'h_xsec_error', 1, 0, 1)
            h.SetBinContent(1, self.cross_section_error)
            self.output_list.append(h)

            # Save sum of weights (effective number of events), in order to keep track of normalization uncertainty
            h = ROOT.TH1F('h_weight_sum', 'h_weight_sum', 1, 0, 1)
            h.SetBinContent(1, self.sum_weights)
            self.output_list.append(h)

            # Save event weights
            bins = np.logspace(np.log10( np.power(self.pt_ref/(self.sqrts/2),self.power) ), np.log10( np.power(self.pt_ref/2,self.power)  ), 10000)
            h = ROOT.TH1F('h_weights', 'h_weights', bins.size-1, bins)
            for weight in self.weights:
                h.Fill(weight)
            self.output_list.append(h)

        #---------------------------------
        # Save additional histograms for QA

        if self.is_AA:
            h = ROOT.TH1F('h_centrality_generated', 'h_centrality_generated', 100, 0, 100)
            h.SetBinContent(self.centrality[1], self.n_events_generated)
            self.output_list.append(h)
        else:
            h = ROOT.TH1F('h_n_events_generated', 'h_n_events_generated', 1, 0, 1)
            h.SetBinContent(1, self.n_events_generated)
            self.output_list.append(h)

        # Save unweighted pt-hat
        bins = np.logspace(np.log10(1.), np.log10(self.sqrts/2), 100)
        h = ROOT.TH1F('h_pt_hat', 'h_pt_hat', bins.size-1, bins)
        h.Sumw2()
        for pt_hat in self.pt_hat:
            h.Fill(pt_hat)
        self.output_list.append(h)

        # Save weighted pt-hat
        h = ROOT.TH1F('h_pt_hat_weighted', 'h_pt_hat_weighted', bins.size-1, bins)
        h.Sumw2()
        for i,pt_hat in enumerate(self.pt_hat):
            h.Fill(pt_hat, self.weights[i])
        self.output_list.append(h)

    #-------------------------------------------------------------------------------------------
    # Histogram hadron observables
    #-------------------------------------------------------------------------------------------
    def histogram_hadron_observables(self, observable_type=''):
        print()
        print(f'Histogram {observable_type} observables...')

        for observable, block in self.config[observable_type].items():
            for centrality_index,centrality in enumerate(block['centrality']):

                # Add centrality bin to list, if needed
                if self.is_AA and centrality not in self.observable_centrality_list:
                    self.observable_centrality_list.append(centrality)

                # Construct appropriate binning
                bins = self.plot_utils.bins_from_config(block, self.sqrts, observable_type, observable, centrality, centrality_index)
                if not bins.any():
                    continue

                # Histogram observable
                self.histogram_observable(column_name=f'{observable_type}_{observable}', bins=bins, centrality=centrality)
                if self.is_AA:
                    self.histogram_observable(column_name=f'{observable_type}_{observable}_holes', bins=bins, centrality=centrality)

    #-------------------------------------------------------------------------------------------
    # Histogram hadron correlation observables
    #-------------------------------------------------------------------------------------------
    def histogram_hadron_correlation_observables(self, observable_type=''):
        print()
        print(f'Histogram {observable_type} observables...')

        def compute_h_total_NpT(column_name, centrality, bins, compute_total_NpT_only=True):
            """
            Helper function to compute h_total_NpT if not already done.
            """
            h_total_NpT_name = f'h_{column_name}_NpT_{centrality}'
            if not any(h.GetName() == h_total_NpT_name for h in self.output_list):
                self.histogram_observable(
                    column_name=column_name,
                    bins=bins,
                    centrality=centrality,
                    compute_total_NpT_only=compute_total_NpT_only,
                )
            return h_total_NpT_name

        for observable, block in self.config[observable_type].items():
            if "v2" in observable:
                for method, method_block in block.items():
                    for centrality_index, centrality in enumerate(method_block['centrality']):
                        # Add centrality bin to list, if needed
                        if self.is_AA and centrality not in self.observable_centrality_list:
                            self.observable_centrality_list.append(centrality)

                        # Construct appropriate binning for v2 observables
                        bins = self.plot_utils.bins_from_config(method_block, self.sqrts, observable_type, observable, centrality, centrality_index)
                        if not bins.any():
                            continue

                        # Compute h_total_NpT for the main data
                        total_NpT_column = f'{observable_type}_{observable}_{method}_Qn0_total'
                        h_total_NpT_name = compute_h_total_NpT(total_NpT_column, centrality, bins)

                        # Compute h_total_NpT for holes if in AA mode
                        if self.is_AA:
                            total_NpT_holes_column = f'{observable_type}_{observable}_{method}_Qn0_total_holes'
                            h_total_NpT_holes_name = compute_h_total_NpT(total_NpT_holes_column, centrality, bins)

                        # Loop over harmonics n
                        for n in range(1, self.norder):
                            # Handle both real and imaginary components
                            for component in ['real', 'img']:
                                self.histogram_observable(
                                    column_name=f'{observable_type}_{observable}_{method}_Qn{n}_{component}',
                                    bins=bins,
                                    centrality=centrality,
                                    h_total_NpT_name=h_total_NpT_name,  # Pass precomputed h_total_NpT
                                )
                                if self.is_AA:
                                    self.histogram_observable(
                                        column_name=f'{observable_type}_{observable}_{method}_Qn{n}_{component}_holes',
                                        bins=bins,
                                        centrality=centrality,
                                        h_total_NpT_name=h_total_NpT_holes_name,  # Pass precomputed h_total_NpT for holes
                                    )

            # Handle dihadron_star separately (no methods loop)
            # STAR dihadron
            elif observable == 'dihadron_star':
                for centrality_index,centrality in enumerate(block['centrality']):

                    # Add centrality bin to list, if needed
                    if self.is_AA and centrality not in self.observable_centrality_list:
                        self.observable_centrality_list.append(centrality)

                    # Construct appropriate binning
                    dphi_bins = np.array(block["dphi_bins"])

                    # Histogram observable
                    pt_trigger_ranges = block["pt_trig"]
                    pt_associated_ranges = block["pt_assoc"]
                    # Loop over trigger and associated ranges
                    # NOTE: n_trig will be calculated when histogram observable
                    for i_trig_bin, pt_trig_range in enumerate(pt_trigger_ranges):
                        histogrammed_n_trig = False
                        pt_trig_min, pt_trig_max = pt_trig_range
                        for pt_assoc_range in pt_associated_ranges:
                            pt_assoc_min, pt_assoc_max = pt_assoc_range
                            # If the upper range has -1, it's unbounded, so we make it large enough not to matter
                            pt_assoc_max = 1000 if pt_assoc_max == -1 else pt_assoc_max

                            label = f"pt_trig_{pt_trig_min:g}_{pt_trig_max:g}_pt_assoc_{pt_assoc_min:g}_{pt_assoc_max:g}"
                            self.histogram_observable(
                                column_name=f'{observable_type}_{observable}_{label}',
                                bins=dphi_bins, centrality=centrality,
                                # We only want to histogram the number of triggers once. Otherwise, we're just
                                # repeatedly replacing the histogram.
                                pt_bin=i_trig_bin if histogrammed_n_trig == False else None, block=block
                            )
                            if self.is_AA:
                                self.histogram_observable(
                                    column_name=f'{observable_type}_{observable}_{label}_holes',
                                    bins=dphi_bins, centrality=centrality,
                                    # We only want to histogram the number of triggers once. Otherwise, we're just
                                    # repeatedly replacing the histogram.
                                    pt_bin=i_trig_bin if histogrammed_n_trig == False else None, block=block
                                )
                            histogrammed_n_trig = True

    #-------------------------------------------------------------------------------------------
    # Histogram inclusive jet observables
    #-------------------------------------------------------------------------------------------
    def histogram_jet_observables(self, observable_type='', jet_collection_label=''):
        print()
        print(f'Histogram {observable_type} observables...')

        for observable, block in self.config[observable_type].items():
            # Check if the observable is v2-related
            if 'v2' in observable:
                # Loop over methods within each observable
                for method, method_block in block.items():
                    for centrality_index, centrality in enumerate(method_block['centrality']):

                        # Add centrality bin to list, if needed
                        if self.is_AA and centrality not in self.observable_centrality_list:
                            self.observable_centrality_list.append(centrality)

                        for jet_R in method_block['jet_R']:

                            self.suffix = f'_R{jet_R}'

                            bins = self.plot_utils.bins_from_config(method_block, self.sqrts, observable_type, observable,
                                                                    centrality, centrality_index, suffix=f'{self.suffix}')
                            if not bins.any():
                                continue

                            self.histogram_observable(column_name=f'{observable_type}_{observable}_{method}{self.suffix}{jet_collection_label}',
                                                      bins=bins, centrality=centrality, block=method_block)
                            if jet_collection_label in ['_shower_recoil']:
                                self.histogram_observable(column_name=f'{observable_type}_{observable}_{method}{self.suffix}{jet_collection_label}_unsubtracted',
                                                          bins=bins, centrality=centrality, block=method_block)

            else: # Non-v2 observables
                for centrality_index,centrality in enumerate(block['centrality']):

                    # Add centrality bin to list, if needed
                    if self.is_AA and centrality not in self.observable_centrality_list:
                        self.observable_centrality_list.append(centrality)

                    for jet_R in block['jet_R']:

                        # Custom skip
                        if observable == 'zg_alice' or observable == 'tg_alice':
                            if np.isclose(jet_R, 0.4) and centrality_index == 0:
                                continue
                            if np.isclose(jet_R, 0.2) and centrality_index == 1:
                                continue

                        # Optional: Loop through pt bins
                        for pt_bin in range(len(block['pt'])-1):

                            # Custom skip
                            if observable == 'xj_atlas':
                                if centrality_index > 0 and pt_bin !=0:
                                    continue

                            if len(block['pt']) > 2:
                                pt_suffix = f'_pt{pt_bin}'
                            else:
                                pt_suffix = ''

                            # Optional: subobservable
                            subobservable_label_list = ['']
                            if 'kappa' in block:
                                subobservable_label_list = [f'_k{kappa}' for kappa in block['kappa']]
                            if 'r' in block:
                                subobservable_label_list = [f'_r{r}' for r in block['r']]
                            for subobservable_label in subobservable_label_list:

                                if 'SoftDrop' in block:
                                    for grooming_setting in block['SoftDrop']:
                                        zcut = grooming_setting['zcut']
                                        beta = grooming_setting['beta']

                                        self.suffix = f'_R{jet_R}_zcut{zcut}_beta{beta}{subobservable_label}'
                                        bins = self.plot_utils.bins_from_config(block, self.sqrts, observable_type, observable,
                                                                                centrality, centrality_index,
                                                                                suffix=f'{self.suffix}{pt_suffix}')
                                        if not bins.any():
                                            continue

                                        self.histogram_observable(column_name=f'{observable_type}_{observable}{self.suffix}{jet_collection_label}',
                                                                    bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)
                                        if jet_collection_label in ['_shower_recoil']:
                                            self.histogram_observable(column_name=f'{observable_type}_{observable}{self.suffix}{jet_collection_label}_unsubtracted',
                                                                        bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)
                                else:
                                    self.suffix = f'_R{jet_R}{subobservable_label}'

                                    bins = self.plot_utils.bins_from_config(block, self.sqrts, observable_type, observable, centrality,
                                                                            centrality_index, suffix=f'{self.suffix}{pt_suffix}')

                                    if not bins.any():
                                        continue

                                    self.histogram_observable(column_name=f'{observable_type}_{observable}{self.suffix}{jet_collection_label}',
                                                              bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)
                                    if jet_collection_label in ['_shower_recoil']:
                                        self.histogram_observable(column_name=f'{observable_type}_{observable}{self.suffix}{jet_collection_label}_unsubtracted',
                                                                bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)

    #-------------------------------------------------------------------------------------------
    # Histogram semi-inclusive jet observables
    #-------------------------------------------------------------------------------------------
    def histogram_semi_inclusive_chjet_observables(self, observable_type='', jet_collection_label=''):
        print()
        print(f'Histogram {observable_type} observables...')

        for observable, block in self.config[observable_type].items():
            for centrality_index,centrality in enumerate(block['centrality']):

                # Add centrality bin to list, if needed
                if self.is_AA and centrality not in self.observable_centrality_list:
                    self.observable_centrality_list.append(centrality)

                for jet_R in block['jet_R']:
                    self.suffix = f'_R{jet_R}'

                    if self.sqrts == 2760 or self.sqrts == 5020:

                        if 'dphi' in observable:

                            # loop over pt bins for dphi observable
                            for pt_bin in range(len(block['pt'])-1):

                                pt_suffix = f'_pt{pt_bin}'

                                # Construct appropriate binning
                                bins = self.plot_utils.bins_from_config(block, self.sqrts, observable_type, observable,
                                                                        centrality, centrality_index,
                                                                        # suffix=f'{self.suffix}{pt_suffix}')
                                                                        self.suffix)
                                if not bins.any():
                                    continue

                                self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_lowTrigger{jet_collection_label}',
                                                          bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)
                                self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_highTrigger{jet_collection_label}',
                                                          bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)

                                if jet_collection_label in ['_shower_recoil']:
                                    self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_lowTrigger{jet_collection_label}_unsubtracted',
                                            bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)
                                    self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_highTrigger{jet_collection_label}_unsubtracted',
                                            bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)




                                self.histogram_observable(column_name=f'{observable_type}_{observable}{self.suffix}{jet_collection_label}',
                                                          bins=bins, centrality=centrality, pt_suffix=pt_suffix, pt_bin=pt_bin, block=block)

                        else:

                            # Construct appropriate binning
                            bins = self.plot_utils.bins_from_config(block, self.sqrts, observable_type, observable,
                                                                    centrality, centrality_index, self.suffix)

                            if not bins.any():
                                continue

                            self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_lowTrigger{jet_collection_label}',
                                                      bins=bins, centrality=centrality)
                            self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_highTrigger{jet_collection_label}',
                                                      bins=bins, centrality=centrality)

                            if jet_collection_label in ['_shower_recoil']:
                                self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_lowTrigger{jet_collection_label}_unsubtracted',
                                        bins=bins, centrality=centrality)
                                self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}_highTrigger{jet_collection_label}_unsubtracted',
                                        bins=bins, centrality=centrality)


                        if np.isclose(jet_R, block['jet_R'][0]):
                            column_name = f'{observable_type}_{observable}_trigger_pt{jet_collection_label}'
                            bins = np.array(block['low_trigger_range'] + block['high_trigger_range']).astype(np.float64)
                            self.histogram_observable(column_name=column_name, bins=bins, centrality=centrality, observable=observable)

                    elif self.sqrts == 200:
                        bins = self.plot_utils.bins_from_config(block, self.sqrts, observable_type, observable,
                                                                centrality, centrality_index, self.suffix)


                        self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}{jet_collection_label}',
                                                  bins=bins, centrality=centrality)
                        if jet_collection_label in ['_shower_recoil']:
                            self.histogram_observable(column_name=f'{observable_type}_{observable}_R{jet_R}{jet_collection_label}_unsubtracted',
                                                    bins=bins, centrality=centrality)

                        if observable == 'IAA_star' and np.isclose(jet_R, block['jet_R'][0]):
                            column_name = f'{observable_type}_star_trigger_pt{jet_collection_label}'
                            bins = np.array(block['trigger_range'])
                            self.histogram_observable(column_name=column_name, bins=bins, centrality=centrality)

    #-------------------------------------------------------------------------------------------
    # Histogram a single observable
    #-------------------------------------------------------------------------------------------
    def histogram_observable(self, column_name=None, bins=None, centrality=None, pt_suffix='',
                             pt_bin=None, block=None, observable='', 
                             compute_total_NpT_only=False, h_total_NpT_name=None):

        # Check if event centrality is within observable centrality bin
        if not self.centrality_accepted(centrality):
            return

        # Get column
        if column_name in self.observables_df.keys():
            col = self.observables_df[column_name]
        else:
            return

        # Find dimension of observable
        dim_observable = 0
        for i,_ in enumerate(col):
            if col[i] is not None:
                dim_observable = col[i][0].size
                break

        # Construct histogram
        if dim_observable == 1:
            self.histogram_1d_observable(col, column_name=column_name, bins=bins, centrality=centrality, pt_suffix=pt_suffix, observable=observable)
        elif dim_observable == 2:
            self.histogram_2d_observable(col, column_name=column_name, bins=bins, centrality=centrality, pt_suffix=pt_suffix, block=block, 
                compute_total_NpT_only=compute_total_NpT_only, h_total_NpT_name=h_total_NpT_name)
        else:
            return

        # Also store N_jets for D(z) observables
        if 'Dz' in column_name:
            column_name = f'{column_name}_Njets'
            col = self.observables_df[column_name]
            bins = np.array([block['pt'][pt_bin], block['pt'][pt_bin+1]])
            self.histogram_1d_observable(col, column_name=column_name, bins=bins, centrality=centrality, pt_suffix=pt_suffix)

        # Also store N_trig for dihadron correlations
        # NOTE: We use pt_bin being set as a signal whether we should histogram the triggers.
        #       We then only set it sometimes so that we don't repeating histogram the same quantity.
        if "dihadron_" in column_name and pt_bin is not None:
            start_of_label = column_name.find("_pt_trig")
            column_name_suffix = "_holes" if "_holes" in column_name else ""
            column_name = f'{column_name[:start_of_label]}_Ntrig{column_name_suffix}'
            col = self.observables_df[column_name]
            # We want the same binning for all triggers, so we flatten all of the trigger binning to a flat list
            # NOTE: This assumes that the triggers ranges do not overlap.
            # NOTE: The unique ensures that the binning is valid if the bin edges match up.
            bins = np.unique(np.array([v for trig_range in block["pt_trig"] for v in trig_range]))  # type: ignore
            self.histogram_1d_observable(col, column_name=column_name, bins=bins, centrality=centrality, pt_suffix=pt_suffix)

    #-------------------------------------------------------------------------------------------
    # Histogram a single observable
    #-------------------------------------------------------------------------------------------
    def histogram_1d_observable(self, col, column_name=None, bins=None, centrality=None, pt_suffix='', observable=''):

        hname = f'h_{column_name}{observable}_{centrality}{pt_suffix}'
        h = ROOT.TH1F(hname, hname, len(bins)-1, bins)
        h.Sumw2()

        # Fill histogram
        for i,_ in enumerate(col):
            if col[i] is not None:
                for value in col[i]:
                    h.Fill(value, self.weights[i])

        # Save histogram to output list
        self.output_list.append(h)

    #-------------------------------------------------------------------------------------------
    # Histogram a single observable
    #-------------------------------------------------------------------------------------------
    def histogram_2d_observable(self, col, column_name=None, bins=None, centrality=None, pt_suffix='', block=None, 
                compute_total_NpT_only=False, h_total_NpT_name=None):

        # Check if the observable is v2-related
        if "hadron_correlations_v2" in column_name or "dijet_v2" in column_name:
            # Create and normalize histograms for Qn vector components (real or imaginary)
            # or calculate total_NpT only if compute_total_NpT_only is True.

            h_total_NpT = None

            # Case 1: Compute total_NpT only
            if compute_total_NpT_only:
                h_total_NpT_name = f'h_{column_name}_NpT_{centrality}'
                h_total_NpT = ROOT.TH1F(h_total_NpT_name, h_total_NpT_name, len(bins) - 1, bins)
                h_total_NpT.Sumw2()

                # Fill the total_NpT histogram
                for i, particle_data in enumerate(col):
                    if particle_data is not None:
                        for value in particle_data:
                            pt = value[0]  # pT value
                            h_total_NpT.Fill(pt, 1)  # Increment particle count

                # Append the total_NpT histogram to the output list
                if h_total_NpT_name not in [h.GetName() for h in self.output_list]:
                    self.output_list.append(h_total_NpT)

                return

            # Case 2: Normalize Qn vectors using precomputed total_NpT
            # Fetch the precomputed h_total_NpT
            if h_total_NpT_name:
                h_total_NpT = next((h for h in self.output_list if h.GetName() == h_total_NpT_name), None)
            if not h_total_NpT:
                raise ValueError(f"h_total_NpT histogram '{h_total_NpT_name}' not found in output list.")

            # Define histogram names
            hname = f'h_{column_name}_{centrality}'
            # Create the Qn component histogram
            h_vn_component = ROOT.TH1F(hname, hname, len(bins) - 1, bins)
            h_vn_component.Sumw2()

            # Fill the component histogram
            for i, particle_data in enumerate(col):
                if particle_data is not None:
                    for value in particle_data:
                        pt = value[0]  # pT value
                        component = value[1]  # Real or imaginary component
                        h_vn_component.Fill(pt, component)

            # Normalize the component histogram by total_NpT
            for ibin in range(1, h_total_NpT.GetNbinsX() + 1):
                total_NpT = h_total_NpT.GetBinContent(ibin)
                if total_NpT > 0:
                    h_vn_component.SetBinContent(ibin, h_vn_component.GetBinContent(ibin) / total_NpT)
                else:
                    h_vn_component.SetBinContent(ibin, 0)

            # Append the normalized histogram to the output list
            self.output_list.append(h_vn_component)

            return

        # Non-v2 variables
        hname = f'h_{column_name}_{centrality}{pt_suffix}'
        h = ROOT.TH1F(hname, hname, len(bins)-1, bins)
        h.Sumw2()

        # Get pt bin
        pt_index = int(pt_suffix[-1])
        pt_min = block['pt'][pt_index]
        pt_max = block['pt'][pt_index+1]

        # Fill histogram for non-v2 observables
        for i, _ in enumerate(col):
            if col[i] is not None:
                for value in col[i]:
                    if pt_min < value[0] < pt_max:
                        h.Fill(value[1], self.weights[i])

        # Save histogram to output list
        self.output_list.append(h)

    # ---------------------------------------------------------------
    # Check if event centrality is within observable's centrality
    # ---------------------------------------------------------------
    def centrality_accepted(self, observable_centrality):

        # AA
        if self.is_AA:

            if self.centrality[0] >= observable_centrality[0]:
                if self.centrality[1] <= observable_centrality[1]:
                    return True
            return False

        # pp
        else:
            return True

    # ---------------------------------------------------------------
    # Save all ROOT histograms to file
    # ---------------------------------------------------------------
    def write_output_objects(self):

        # Save output objects
        output_file = self.input_file.replace('observables', 'histograms').replace('parquet', 'root')
        output_path = os.path.join(self.output_dir, output_file)
        fout = ROOT.TFile(output_path, 'recreate')
        fout.cd()
        for obj in self.output_list:

            types = (ROOT.TH1, ROOT.THnBase)
            if isinstance(obj, types):
                obj.Write()
                obj.SetDirectory(0)
                del obj

        fout.Close()

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print()
    print('Executing histogram_results_STAT.py...')

    # Define arguments
    parser = argparse.ArgumentParser(description='Histogram JETSCAPE observables')
    parser.add_argument(
        '-c',
        '--configFile',
        action='store',
        type=str,
        metavar='configFile',
        default='config/TG3.yaml',
        help='Config file'
    )
    parser.add_argument(
        '-i',
        '--inputFile',
        action='store',
        type=str,
        metavar='inputFile',
        default='observables_5020_0000_00.parquet',
        help='Input file'
    )
    parser.add_argument(
        '-o',
        '--outputDir',
        action='store',
        type=str,
        metavar='outputDir',
        default='/home/jetscape-user/JETSCAPE-analysis/TestOutput',
        help='Output directory for output to be written to'
    )

    # Parse the arguments
    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File "{0}" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # If invalid inputDir is given, exit
    if not os.path.exists(args.inputFile):
        print('File "{0}" does not exist! Exiting!'.format(args.inputFile))
        sys.exit(0)

    # If output dir doesn't exist, create it
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    analysis = HistogramResults(config_file=args.configFile, input_file=args.inputFile, output_dir=args.outputDir)
    analysis.histogram_results()
