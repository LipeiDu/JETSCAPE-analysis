# This is a cleaned-up version focusing solely on soft sector observables
# Jet observables and pp/RAA-related logic are removed

import os
import sys
import yaml
import argparse
import ROOT
import numpy as np
import pandas as pd

sys.path.append('.')
from jetscape_analysis.base import common_base
from plot import plot_results_STAT_utils

ROOT.gROOT.SetBatch(True)

class PlotSoftSectorResults(common_base.CommonBase):

    def __init__(self, config_file='', input_file='', output_dir='', **kwargs):
        super().__init__(**kwargs)

        self.output_dir = output_dir or os.path.dirname(input_file)
        os.makedirs(self.output_dir, exist_ok=True)

        self.plot_utils = plot_results_STAT_utils.PlotUtils()
        self.plot_utils.setOptions()
        ROOT.gROOT.ForceStyle()

        self.input_file = ROOT.TFile(input_file, 'READ')

        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.sqrts = self.config['sqrt_s']
        self.norder = self.config['norder']
        self.include_pT_spectra = False

        self.is_AA = True

        self.output_dict = {}

        # Style
        self.data_color = ROOT.kGray+3
        self.data_marker = 21
        self.jetscape_color = [ROOT.kViolet-8, ROOT.kRed-7, ROOT.kTeal-8, ROOT.kCyan-2]
        self.jetscape_fillstyle = [1001, 3144, 1001, 3144]
        self.jetscape_alpha = [0.7] * 4
        self.jetscape_marker = 20
        self.marker_size = 1.5
        self.line_width = 2
        self.line_style = 1
        self.file_format = '.pdf'

    #-------------------------------------------------------------------------------------------
    # Functions
    #-------------------------------------------------------------------------------------------
    def plot_results(self):
        self.plot_hadron_observables()
        self.plot_hadron_correlation_observables()
        self.write_output_objects()

    def plot_hadron_observables(self):
        for observable, block in self.config['hadron'].items():
            for centrality_index, centrality in enumerate(block['centrality']):
                if 'hepdata' not in block and 'custom_data' not in block:
                    continue
                self.init_observable('hadron', observable, block, centrality, centrality_index)
                self.plot_observable('hadron', observable, centrality)

    def plot_hadron_correlation_observables(self):
        for observable, block in self.config['hadron_correlations'].items():
            if 'v2' in observable:
                for method, method_block in block.items():
                    for centrality_index, centrality in enumerate(method_block['centrality']):
                        if 'hepdata' not in method_block and 'custom_data' not in method_block:
                            continue
                        self.init_observable('hadron_correlations', observable, method_block, centrality, centrality_index, method=f'_{method}')
                        self.plot_observable('hadron_correlations', observable, centrality, method=f'_{method}')

    #-------------------------------------------------------------------------------------------
    # Functions
    #-------------------------------------------------------------------------------------------
    def init_observable(self, observable_type, observable, block, centrality, centrality_index, method='', pt_suffix='', self_normalize=False):
        self.observable_settings = {}
        self.xtitle = block.get('xtitle', '')
        self.ytitle = block.get('ytitle_AA', '')
        self.y_min = block.get('y_min_AA', 0.)
        self.y_max = block.get('y_max_AA', 0.3)
        self.y_ratio_min = block.get('y_ratio_min', 0.)
        self.y_ratio_max = block.get('y_ratio_max', 1.99)
        self.eta_cut = block.get('eta_cut', 1.0)
        self.logy = block.get('logy', False)

        self.suffix = ''
        self.init_data_distribution(block, observable_type, observable, centrality_index, pt_suffix)
        self.init_model_distribution(observable_type, observable, method, block, centrality, pt_suffix, self_normalize)

    def init_data_distribution(self, block, observable_type, observable, centrality_index, pt_suffix):
        if observable_type == "hadron":
            include_pT_spectra = True
        if observable_type == "hadron_correlations":
            include_pT_spectra = False
        if 'hepdata' in block:
            self.observable_settings['data_distribution'] = self.plot_utils.tgraph_from_hepdata(block, self.is_AA, self.sqrts, observable_type, observable, centrality_index, suffix=self.suffix, pt_suffix=pt_suffix, pT_spectra=include_pT_spectra)
        elif 'custom_data' in block:
            self.observable_settings['data_distribution'] = self.plot_utils.tgraph_from_yaml(
                block, True, self.sqrts, observable_type, observable, centrality_index, suffix=self.suffix, pt_suffix=pt_suffix)
        else:
            self.observable_settings['data_distribution'] = None

    def init_model_distribution(self, observable_type, observable, method, block, centrality, pt_suffix, self_normalize):
        if observable_type == 'hadron':
            hname = f"h_{observable_type}_{observable}_{centrality}"
            h = self.input_file.Get(hname)
            if not h or not h.InheritsFrom("TH1"):
                return False
            h.SetDirectory(0)

            # Apply physical scaling for soft hadron pT spectra
            if observable.startswith('pt_'):  # apply to pT spectra only
                if hasattr(self, 'eta_cut'):
                    h.Scale(1. / (2. * self.eta_cut))
                else:
                    print(f"[WARN] eta_cut not defined; skipping eta normalization for {hname}")
                h.Scale(1. / (2. * np.pi))

            self.observable_settings['jetscape_distribution'] = h
            return True

        if observable_type == 'hadron_correlations' and 'v2' in observable:
            base_name = f"h_{observable_type}_{observable}{method}_{centrality}"
            settings = {}

            def safe_get(name):
                h = self.input_file.Get(name)
                if not h or not h.InheritsFrom("TH1"):
                    return None
                h.SetDirectory(0)
                return h

            settings['h_N_ref'] = safe_get(f"{base_name}_Qn0_ref")
            settings['h_Qn_ref_real'] = {n: safe_get(f"{base_name}_Qn{n}_real_ref") for n in range(1, self.norder)}
            settings['h_Qn_ref_imag'] = {n: safe_get(f"{base_name}_Qn{n}_imag_ref") for n in range(1, self.norder)}

            settings['h_N_pT'] = safe_get(f"{base_name}_Qn0")
            settings['h_Qn_pT_real'] = {n: safe_get(f"{base_name}_Qn{n}_real") for n in range(1, self.norder)}
            settings['h_Qn_pT_imag'] = {n: safe_get(f"{base_name}_Qn{n}_imag") for n in range(1, self.norder)}

            # Check if any required histogram is missing
            if not settings['h_N_ref'] or not settings['h_N_pT']:
                return False
            if any(v is None for v in settings['h_Qn_ref_real'].values()) or any(v is None for v in settings['h_Qn_ref_imag'].values()):
                return False
            if any(v is None for v in settings['h_Qn_pT_real'].values()) or any(v is None for v in settings['h_Qn_pT_imag'].values()):
                return False

            self.observable_settings['jetscape_distribution'] = settings
            return True

        return False

    #-------------------------------------------------------------------------------------------
    # Functions
    #-------------------------------------------------------------------------------------------
    def plot_observable(self, observable_type, observable, centrality, method='', pt_suffix='', logy=False):
        label = f'{observable_type}_{observable}{method}_{centrality}{pt_suffix}'

        # Try initializing model data
        success = self.init_model_distribution(observable_type, observable, method, {}, centrality, pt_suffix, self_normalize=False)
        if not success:
            print(f"[INFO] Skipping {label} due to missing histograms.")
            return

        # Plot depending on type
        if observable_type == 'hadron_correlations' and 'v2' in observable:
            self.plot_v2_distribution(label, logy=False)
        elif observable_type == 'hadron':
            self.plot_pT_spectra(label)

    def plot_v2_distribution(self, label, logy=False):
        settings = self.observable_settings['jetscape_distribution']

        n = 2  # harmonic order
        h_N_ref = settings['h_N_ref']
        h_Qn_ref_real = settings['h_Qn_ref_real'][n]
        h_Qn_ref_imag = settings['h_Qn_ref_imag'][n]
        h_N_pT = settings['h_N_pT']
        h_Qn_pT_real = settings['h_Qn_pT_real'][n]
        h_Qn_pT_imag = settings['h_Qn_pT_imag'][n]

        QnpT_diff_array = []
        Qnref_array = []

        n_event_bins = h_N_pT.GetNbinsX()
        n_pt_bins = h_N_pT.GetNbinsY()
        pt_bin_centers = [h_N_pT.GetYaxis().GetBinCenter(j) for j in range(1, n_pt_bins+1)]

        for i in range(1, n_event_bins + 1):
            N_pT = np.array([h_N_pT.GetBinContent(i, j) for j in range(1, n_pt_bins + 1)])
            if np.sum(N_pT) < 1e-6:
                continue

            QnpT_event = [N_pT]
            for order in range(1, self.norder):
                real = np.array([settings['h_Qn_pT_real'][order].GetBinContent(i, j) for j in range(1, n_pt_bins + 1)])
                imag = np.array([settings['h_Qn_pT_imag'][order].GetBinContent(i, j) for j in range(1, n_pt_bins + 1)])
                QnpT_event.append(real + 1j * imag)
            QnpT_diff_array.append(QnpT_event)

            N_ref = h_N_ref.GetBinContent(i)
            if N_ref < 1e-6:
                continue
            Qn_event = [N_ref]
            for order in range(1, self.norder):
                real = settings['h_Qn_ref_real'][order].GetBinContent(i)
                imag = settings['h_Qn_ref_imag'][order].GetBinContent(i)
                Qn_event.append(real + 1j * imag)
            Qnref_array.append(Qn_event)

        # Determine method
        method_key = label.split('_')[-2]
        if method_key == 'ep':
            vn_values, vn_errors = self.calculate_vn_event_plane_diff(QnpT_diff_array, Qnref_array)
        elif method_key == 'sp':
            vn_values, vn_errors = self.calculate_vn_scalar_product_diff(QnpT_diff_array, Qnref_array)
        elif method_key == 'four':
            vn_values, vn_errors = self.calculate_vn_four_cumulant_diff(QnpT_diff_array, Qnref_array)
        else:
            raise ValueError(f"Unknown method: {method_key}")

        vn_vals = vn_values[n-1]
        vn_errs = vn_errors[n-1]

        # Skip plotting if all values are invalid
        if np.all(np.array(vn_vals) == 0) or len(vn_vals) != len(pt_bin_centers):
            print(f"[WARN] {label} has invalid or empty vn values, skipping plot.")
            return

        # Check for invalid values for log scale
        has_negative = np.any(np.array(vn_vals) <= 0)
        if logy and has_negative:
            print(f"[WARN] {label} contains non-positive vn values; switching to linear scale.")
            logy = False

        # Create TGraphErrors for model
        graph = ROOT.TGraphErrors(len(pt_bin_centers))
        for i, (pt, val, err) in enumerate(zip(pt_bin_centers, vn_vals, vn_errs)):
            graph.SetPoint(i, pt, val)
            graph.SetPointError(i, 0, err)

        c = ROOT.TCanvas(f"c_{label}", f"Canvas {label}", 600, 450)
        if logy:
            c.SetLogy()
        c.SetLeftMargin(0.15)
        c.SetBottomMargin(0.15)
        c.cd()

        frame = ROOT.TH1F("frame", "", 1, pt_bin_centers[0]*0.8, pt_bin_centers[-1]*1.2)
        frame.SetMinimum(self.y_min)
        frame.SetMaximum(self.y_max)
        frame.GetXaxis().SetTitle(self.xtitle)
        frame.GetYaxis().SetTitle(self.ytitle)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetYaxis().SetTitleOffset(1.2)
        frame.SetNdivisions(505)
        frame.Draw("AXIS")

        # Plot model
        graph.SetFillColor(self.jetscape_color[0])
        graph.SetFillColorAlpha(self.jetscape_color[0], self.jetscape_alpha[0])
        graph.SetFillStyle(self.jetscape_fillstyle[0])
        graph.SetMarkerStyle(0)
        graph.SetLineWidth(0)
        graph.Draw("E3 SAME")

        # Plot data if exists
        h_data = self.observable_settings.get("data_distribution")
        legend = ROOT.TLegend(0.4, 0.65, 0.75, 0.88)
        self.plot_utils.setup_legend(legend, 0.045, sep=-0.1)

        legend.AddEntry(graph, "JETSCAPE", "f")
        if h_data:
            h_data.SetMarkerStyle(self.data_marker)
            h_data.SetMarkerSize(self.marker_size)
            h_data.SetMarkerColor(self.data_color)
            h_data.SetLineColor(self.data_color)
            h_data.SetLineWidth(self.line_width)
            h_data.Draw("PE Z SAME")
            legend.AddEntry(h_data, "Data", "p")

        legend.Draw()

        text_latex = ROOT.TLatex()
        text_latex.SetNDC()
        text_latex.SetTextSize(0.06)
        text_latex.DrawLatex(0.18, 0.88, f"#bf{{{label}}}  #sqrt{{s}} = {self.sqrts / 1000.} TeV")

        c.SaveAs(os.path.join(self.output_dir, f"{label}_v2{self.file_format}"))
        c.Close()

    def plot_pT_spectra(self, label):
        settings = self.observable_settings
        h_model = settings.get('jetscape_distribution')
        h_data = settings.get('data_distribution')
        if not h_model:
            print(f"[WARN] Missing pT spectra histogram: {label}")
            return

        c = ROOT.TCanvas(f"c_{label}", f"Canvas {label}", 600, 450)
        c.SetLogy()
        c.SetLeftMargin(0.15)
        c.SetBottomMargin(0.15)
        c.SetTopMargin(0.05)
        c.SetRightMargin(0.05)

        # Define binning
        bins = np.array(h_model.GetXaxis().GetXbins())
        if bins.size == 0:  # for uniform binning
            xlow = h_model.GetXaxis().GetXmin()
            xup = h_model.GetXaxis().GetXmax()
            nbins = h_model.GetNbinsX()
            bins = np.linspace(xlow, xup, nbins+1)

        # Blank histogram frame
        h_frame = ROOT.TH1F('h_frame', '', 1, bins[0], bins[-1])
        h_frame.SetMinimum(1e-8)
        h_frame.SetMaximum(1e5)
        h_frame.GetXaxis().SetTitle(self.xtitle)
        h_frame.GetYaxis().SetTitle(self.ytitle)
        h_frame.GetYaxis().SetTitleOffset(1.2)
        h_frame.GetXaxis().SetTitleOffset(1.2)
        h_frame.SetNdivisions(505)
        h_frame.Draw("AXIS")

        legend = ROOT.TLegend(0.4, 0.65, 0.75, 0.88)
        self.plot_utils.setup_legend(legend, 0.045, sep=-0.1)

        # Draw model as band
        h_model.SetFillColor(self.jetscape_color[0])
        h_model.SetFillColorAlpha(self.jetscape_color[0], self.jetscape_alpha[0])
        h_model.SetFillStyle(self.jetscape_fillstyle[0])
        h_model.SetLineWidth(0)
        h_model.Draw("E3 SAME")
        legend.AddEntry(h_model, "JETSCAPE", "f")

        # Draw data
        if h_data:
            h_data.SetMarkerStyle(self.data_marker)
            h_data.SetMarkerSize(self.marker_size)
            h_data.SetMarkerColor(self.data_color)
            h_data.SetLineWidth(self.line_width)
            h_data.SetLineColor(self.data_color)
            h_data.Draw("P SAME")
            legend.AddEntry(h_data, "Data", "p")

        legend.Draw()

        # Annotate
        text_latex = ROOT.TLatex()
        text_latex.SetNDC()
        text_latex.SetTextSize(0.06)
        text_latex.DrawLatex(0.18, 0.88, f"#bf{{{label}}}  #sqrt{{s}} = {self.sqrts/1000.} TeV")

        c.SaveAs(os.path.join(self.output_dir, f"{label}_spectra{self.file_format}"))
        c.Close()


    def write_output_objects(self):
        pass  # Implement saving of results if needed

    #-------------------------------------------------------------------------------------------
    # Functions for flow calculations
    #-------------------------------------------------------------------------------------------
    def calculate_vn_event_plane_diff(self, QnpT_diff, Qnref):
        """
        This function calculates the event-plane vn(pT) using the Event Plane (EP) method.
        Assumption: No overlap between particles of interest and reference flow Qn vectors.
        
        Inputs:
            QnpT_diff: [nev, norder, npT], flow vectors for particles of interest (POI) in each pT bin.
            Qnref: [nev, norder], reference flow vectors for event plane reconstruction.
        
        Returns:
            [vn{EP}(pT), vn{EP}(pT)_err]: Mean and error of vn{EP}(pT) for each harmonic order and pT bin.
        """
        QnpT_diff = np.array(QnpT_diff)
        Qnref = np.array(Qnref)
        nev, norder, npT = QnpT_diff.shape

        vn_values = []
        vn_errors = []

        Nref = np.real(Qnref[:, 0])  # Number of reference particles
        N2refPairs = Nref * (Nref - 1.)  # Number of reference particle pairs
        NpTPOI = np.real(QnpT_diff[:, 0, :])  # Number of POI in each pT bin
        N2POIPairs = NpTPOI * Nref.reshape(nev, 1)  # Number of POI-reference pairs

        for iorder in range(1, norder):  # Loop over harmonic orders (n >= 1)
            # Normalize reference flow vectors
            QnRef_tmp = Qnref[:, iorder]
            QnRef_norm = QnRef_tmp / np.abs(QnRef_tmp)  # QnA / |QnA|

            # Compute event plane resolution: <QnA/|QnA| * QnB*/|QnB|>
            n2ref = np.real(QnRef_norm * np.conj(QnRef_norm))  # QnA/|QnA| * QnA*/|QnA| = 1

            # Compute numerator: <QnPOI * QnA*/|QnA|>
            QnpT_tmp = QnpT_diff[:, iorder, :]
            n2pT = np.real(QnpT_tmp * np.conj(QnRef_norm.reshape(nev, 1)))

            # Calculate observables with Jackknife resampling
            vnEPpT_arr = np.zeros([nev, npT])
            for iev in range(nev):
                array_idx = [True] * nev
                array_idx[iev] = False
                array_idx = np.array(array_idx)

                # Event plane resolution term
                Cn2ref_arr = np.mean(n2ref[array_idx]) / np.mean(N2refPairs[array_idx])

                # vn{EP}(pT) for this event subset
                vnEPpT_arr[iev, :] = (np.mean(n2pT[array_idx], 0) 
                                      / (np.mean(N2POIPairs[array_idx], 0)+1.e-20)
                                      / (np.sqrt(Cn2ref_arr))+1.e-20)

            # Compute mean and error of vn{EP}(pT)
            vnEPpT_mean = np.mean(vnEPpT_arr, 0)
            vnEPpT_err = np.sqrt((nev - 1.) / nev * np.sum((vnEPpT_arr - vnEPpT_mean)**2., 0))

            vn_values.append(vnEPpT_mean)
            vn_errors.append(vnEPpT_err)

        return [vn_values, vn_errors]

    def calculate_vn_scalar_product_diff(self, QnpT_diff, Qnref):
        """
            this funciton calculates the scalar-product vn
            assumption: no overlap between particles of interest
                        and reference flow Qn vectors
            inputs: QnpT_diff[nev, norder, npT], Qnref[nev, norder]
            return: [vn{SP}(pT), vn{SP}(pT)_err]
        """
        QnpT_diff = np.array(QnpT_diff)
        Qnref = np.array(Qnref)
        nev, norder, npT = QnpT_diff.shape

        vn_values = []
        vn_errors = []

        Nref = np.real(Qnref[:, 0])
        N2refPairs = Nref*(Nref - 1.)
        NpTPOI = np.real(QnpT_diff[:, 0, :])
        N2POIPairs = NpTPOI*Nref.reshape(nev, 1)
        for iorder in range(1, norder):
            # compute Cn^ref{2}
            QnRef_tmp = Qnref[:, iorder]
            n2ref = np.abs(QnRef_tmp)**2. - Nref

            # compute vn{SP}(pT)
            QnpT_tmp = QnpT_diff[:, iorder, :]
            n2pT = np.real(QnpT_tmp*np.conj(QnRef_tmp.reshape(nev, 1)))

            # calcualte observables with Jackknife resampling method
            vnSPpT_arr = np.zeros([nev, npT])
            for iev in range(nev):
                array_idx = [True]*nev
                array_idx[iev] = False
                array_idx = np.array(array_idx)

                Cn2ref_arr = np.mean(n2ref[array_idx])/np.mean(N2refPairs[array_idx])
                vnSPpT_arr[iev, :] = (np.mean(n2pT[array_idx], 0)
                        /(np.mean(N2POIPairs[array_idx], 0)+1.e-20)/(np.sqrt(Cn2ref_arr))+1.e-20)
            vnSPpT_mean = np.mean(vnSPpT_arr, 0)
            vnSPpT_err  = np.sqrt((nev - 1.)/nev
                               *np.sum((vnSPpT_arr - vnSPpT_mean)**2., 0))
            
            vn_values.append(vnSPpT_mean)
            vn_errors.append(vnSPpT_err)

        return [vn_values, vn_errors]

    def calculate_vn_four_cumulant_diff(self, QnpT_diff, Qnref):
        """
            This function calculates the 4-particle vn(pT) using the scalar-product method.
            Assumption: No overlap between particles of interest and reference flow Qn vectors.
            Inputs:
                - QnpT_diff: Shape [nev, norder, npT]
                - Qnref: Shape [nev, norder]
            Returns:
                - vn_values: List of mean vn values for different harmonic orders (one array per harmonic order)
                - vn_errors: List of vn errors for different harmonic orders (one array per harmonic order)
        """
        QnpT_diff = np.array(QnpT_diff)
        Qnref = np.array(Qnref)
        nev, norder, npT = QnpT_diff.shape

        vn_values = []
        vn_errors = []

        for iorder in range(1, 4):  # Process for orders 1 to 3
            # compute Cn^ref{4}
            Nref = np.real(Qnref[:, 0])
            QnRef_tmp = Qnref[:, iorder]
            Q2nRef_tmp = Qnref[:, 2 * iorder]
            N4refPairs = Nref * (Nref - 1.) * (Nref - 2.) * (Nref - 3.)
            n4ref = (np.abs(QnRef_tmp)**4.
                     - 2. * np.real(Q2nRef_tmp * np.conj(QnRef_tmp) * np.conj(QnRef_tmp))
                     - 4. * (Nref - 2) * np.abs(QnRef_tmp)**2. + np.abs(Q2nRef_tmp)**2.
                     + 2. * Nref * (Nref - 3))
            N2refPairs = Nref * (Nref - 1.)
            n2ref = np.abs(QnRef_tmp)**2. - Nref

            # compute dn{4}(pT)
            NpTPOI = np.real(QnpT_diff[:, 0, :])
            QnpT_tmp = QnpT_diff[:, iorder, :]
            Nref = Nref.reshape(len(Nref), 1)
            QnRef_tmp = QnRef_tmp.reshape(len(QnRef_tmp), 1)
            Q2nRef_tmp = Q2nRef_tmp.reshape(len(Q2nRef_tmp), 1)
            N4POIPairs = NpTPOI * (Nref - 1.) * (Nref - 2.) * (Nref - 3.) + 1e-30
            n4pT = np.real(QnpT_tmp * QnRef_tmp * np.conj(QnRef_tmp) * np.conj(QnRef_tmp)
                           - 2. * (Nref - 1) * QnpT_tmp * np.conj(QnRef_tmp)
                           - QnpT_tmp * QnRef_tmp * np.conj(Q2nRef_tmp))
            N2POIPairs = NpTPOI * Nref + 1e-30
            n2pT = np.real(QnpT_tmp * np.conj(QnRef_tmp))

            # Calculate observables with Jackknife resampling
            Cn2ref_arr = np.zeros(nev)
            Cn4ref_arr = np.zeros(nev)
            dn4pT_arr = np.zeros(npT)
            vn4pT4_arr = np.zeros([nev, npT])
            
            for iev in range(nev):
                array_idx = [True] * nev
                array_idx[iev] = False
                array_idx = np.array(array_idx)

                Cn2ref_arr[iev] = (
                    np.mean(n2ref[array_idx]) / np.mean(N2refPairs[array_idx])
                )
                Cn4ref_arr[iev] = (
                    np.mean(n4ref[array_idx]) / np.mean(N4refPairs[array_idx])
                    - 2. * (Cn2ref_arr[iev])**2.
                )

                dn4pT_arr = (
                    np.mean(n4pT[array_idx, :], 0) / np.mean(N4POIPairs[array_idx, :], 0)
                    - 2. * np.mean(n2pT[array_idx, :], 0) / np.mean(N2POIPairs[array_idx, :], 0) * Cn2ref_arr[iev]
                )

                vn4pT4_arr[iev, :] = (-dn4pT_arr)**4. / ((-Cn4ref_arr[iev])**3.)

            vn4pT4_mean = np.mean(vn4pT4_arr, axis=0)
            vn4pT4_err = np.sqrt((nev - 1.) / nev * np.sum((vn4pT4_arr - vn4pT4_mean)**2., axis=0))

            vn4pT = np.zeros(npT)
            vn4pT_err = np.zeros(npT)
            idx = vn4pT4_mean > 0
            vn4pT[idx] = vn4pT4_mean[idx]**(0.25)
            vn4pT_err[idx] = vn4pT4_err[idx] / (4. * vn4pT4_mean[idx]**(0.75))

            # Append calculated values and errors for this harmonic order
            vn_values.append(vn4pT)
            vn_errors.append(vn4pT_err)

        return [vn_values, vn_errors]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Soft Sector Observables from JETSCAPE')
    parser.add_argument('-c', '--configFile', required=True)
    parser.add_argument('-i', '--inputFile', required=True)
    parser.add_argument('-o', '--outputDir', default='')
    args = parser.parse_args()

    analysis = PlotSoftSectorResults(
        config_file=args.configFile,
        input_file=args.inputFile,
        output_dir=args.outputDir
    )
    analysis.plot_results()
