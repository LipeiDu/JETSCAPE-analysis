"""
  macro for plotting analyzed jetscape events
  """

# This script plots histograms created in the analysis of Jetscape events
#
# Author: James Mulligan (james.mulligan@berkeley.edu)

# General
import os
import sys
import argparse
import yaml

# Data analysis and plotting
import ROOT
from array import *
import numpy as np

# Base class
sys.path.append('.')
from jetscape_analysis.base import common_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)


################################################################
class PlotResults(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file='', input_file='', output_dir="", **kwargs):
        super(PlotResults, self).__init__(**kwargs)

        # Read config file
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        
        if not output_dir.endswith("/"):
            output_dir = output_dir + "/"
        self.output_dir = output_dir
        
        self.nEvents = 1000
        self.file_format = '.pdf'
        
        # Filename from my output
        self.filename = input_file
        
        # Filenames for reference data
        self.filename_amit = '/home/jetscape-user/JETSCAPE-analysis-output/AnalysisResultsFinal.root'
        self.filename_gojko = '/home/jetscape-user/JETSCAPE-analysis-output/AnalysisResultsFinal.root'

        self.prompt_photon_pt_ranges = config['prompt_photon_pt_ranges']
        self.file_ATLAS_gamma_Jet_5020 = config['ATLAS_gamma_Jet_5020']

        print(self)

    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    def plot_results(self):
      
        self.setOptions()
        ROOT.gROOT.ForceStyle()

        #self.plot_jet_cross_section('2760', eta_cut = 2.0)
        self.plot_jet_cross_section('5020', eta_cut = 0.3)
        self.plot_jet_cross_section('5020', eta_cut = 2.8)

        #self.plot_ch_hadron_cross_section('2760', eta_cut = 1.0)
        self.plot_ch_hadron_cross_section('5020', eta_cut = 1.0)

        self.plot_photon_cross_section('5020', eta_cut = 1.0)

        for i, (pt_lower, pt_upper) in enumerate(self.prompt_photon_pt_ranges, start=1):
            self.plot_xjgamma_distribution('5020', i, pt_lower, pt_upper, jetR = 0.4, eta_cut=None)

    #-------------------------------------------------------------------------------------------
    def plot_xjgamma_distribution(self, sqrts, i, pt_lower, pt_upper, jetR, eta_cut):

        # Open the ROOT file of the experimental data
        f1 = ROOT.TFile(self.file_ATLAS_gamma_Jet_5020, 'READ')

        # Get the table corresponding to the current pt range
        table_name = 'Table {}'.format(i)
        dir = f1.Get(table_name)

        # Get histograms for central values and statistical errors
        h_data_central = dir.Get('Hist1D_y1')
        h_data_stat_error = dir.Get('Hist1D_y1_e1')

        # Create a new histogram to store combined data which combine the central values and statistical errors

        # Get the bin edges from h_data_central
        bin_edges = []
        for bin_idx in range(1, h_data_central.GetNbinsX() + 2):
            bin_edges.append(h_data_central.GetXaxis().GetBinLowEdge(bin_idx))
        bin_edges.append(h_data_central.GetXaxis().GetBinUpEdge(h_data_central.GetNbinsX()))

        # Create a new histogram with binning matching h_data_central
        h_combined_data = ROOT.TH1F('h_combined_data', 'Combined Data', len(bin_edges) - 1,
                                    array('d', bin_edges))

        # Loop over bins and set central values and statistical errors
        for bin_idx in range(1, h_data_central.GetNbinsX() + 1):
            central_value = h_data_central.GetBinContent(bin_idx)
            stat_error = h_data_stat_error.GetBinContent(bin_idx)
            h_combined_data.SetBinContent(bin_idx, central_value)
            h_combined_data.SetBinError(bin_idx, stat_error)

        # Get the histogram of model calculations
        f2 = ROOT.TFile(self.filename, 'READ')

        hXjgamma = f2.Get('hXjgamma_PhotonPt_{:.1f}_{:.1f}_R{}Scaled'.format(pt_lower, pt_upper, jetR))
        
        # Obtain the number of prompt photons within the pT range
        prompt_photon_hist = f2.Get('hPromptPhoton_PtScaled')
        bin_label = '{:.1f}-{:.1f}'.format(pt_lower, pt_upper)
        bin_index = prompt_photon_hist.GetXaxis().FindBin(bin_label)
        n_prompt_photons = prompt_photon_hist.GetBinContent(bin_index)

        # Normalize the distributions by the number of prompt photons
        hXjgamma.Scale(1/n_prompt_photons, 'width')
        
        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hXjgammaPP_Ratio_{}Pt_{}_{}_R{}{}'.format(sqrts, pt_lower, pt_upper, jetR, self.file_format))
        xtitle = '#it{x}_{J#gamma}'
        ytitle = '(1/N_{#gamma})(dN/d#it{x}_{J#gamma})'

        # hXjgamma: model calculations; h_combined_data: experimental measurements
        self.plot_ratio(hXjgamma, h_combined_data, output_filename, xtitle, ytitle, sqrts, eta_cut, pt_lower, pt_upper, label='Xjgamma')

        f1.Close()
        f2.Close()

    #-------------------------------------------------------------------------------------------
    def plot_jet_cross_section(self, sqrts, eta_cut, pt_lower=None, pt_upper=None):
    
        # Get my histogram
        f = ROOT.TFile(self.filename, 'READ')

        hJetPt_eta = f.Get('hJetPt_eta_R0.4Scaled')
        
        hJetPt_eta.GetYaxis().SetRangeUser(-1.*eta_cut, eta_cut)
        hJetPt_finebinned = hJetPt_eta.ProjectionX()
        hJetPt_finebinned.SetName('{}_{}_{}'.format(hJetPt_finebinned, sqrts, eta_cut))
        
        pt_bins = []
        if sqrts == '2760':
            pt_bins = [70, 80, 90, 100, 110, 130, 150, 170, 190, 210, 240, 270, 300]
        elif sqrts == '5020':
            pt_bins = [40, 50, 63, 79, 100, 126, 158, 200, 251, 316]#, 398, 501, 631, 800, 1000]
        n_pt_bins = len(pt_bins) - 1
        pt_bin_array = array('d', pt_bins)
        hJetPt = hJetPt_finebinned.Rebin(n_pt_bins, '{}{}'.format(hJetPt_finebinned.GetName(), 'rebinned'), pt_bin_array)
        
        eta_acc = 2*eta_cut
        hJetPt.Scale(1/(self.nEvents * eta_acc), 'width')
        
        # Get reference histogram
        f_amit = ROOT.TFile(self.filename_amit, 'READ')
        
        hname = ''
        if sqrts == '2760':
            hname = 'hJetRAA_CMS_2760'
        elif sqrts == '5020':
            if eta_cut == 0.3:
                hname = 'hJetRAA_ATLAS_5020_eta03'
            elif eta_cut == 2.8:
                hname = 'hJetRAA_ATLAS_5020_eta28'
                
        hJetPt_amit = hJetPt#f_amit.Get(hname)

        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hJetCrossSectionPP_Ratio_{}_eta{}{}'.format(sqrts, self.remove_periods(eta_cut), self.file_format))
        xtitle = '#it{p}_{T,jet} (GeV/#it{c})'
        ytitle = '#frac{d^{2}#sigma}{d#it{p}_{T,jet}d#it{#eta}_{jet}} #left[mb (GeV/c)^{-1}#right]'
        self.plot_ratio(hJetPt, hJetPt_amit, output_filename, xtitle, ytitle, sqrts, eta_cut, pt_lower, pt_upper, label = 'Jet')

    #-------------------------------------------------------------------------------------------
    def plot_ch_hadron_cross_section(self, sqrts, eta_cut, pt_lower=None, pt_upper=None):

        # Get my histogram
        f = ROOT.TFile(self.filename, 'READ')

        hHadronPt_eta = f.Get('hChHadronPt_etaScaled')
        
        hHadronPt_eta.GetYaxis().SetRangeUser(-1.*eta_cut, eta_cut)
        hHadronPt_finebinned = hHadronPt_eta.ProjectionX()
        hHadronPt_finebinned.SetName('{}_{}_{}'.format(hHadronPt_finebinned, sqrts, eta_cut))
        
        if sqrts == '2760':
            pt_bins = [9.6, 12.0, 14.4, 19.2, 24.0, 28.8, 35.2, 41.6, 48.0, 60.8, 73.6, 86.4, 103.6]
        else:
            pt_bins = [9.6, 12.0, 14.4, 19.2, 24.0, 28.8, 35.2, 41.6, 48.0, 60.8, 73.6, 86.4, 103.6]#, 120.8, 140.0, 165.0, 250.0, 400.0]
        n_pt_bins = len(pt_bins) - 1
        pt_bin_array = array('d', pt_bins)
        hHadronPt = hHadronPt_finebinned.Rebin(n_pt_bins, '{}{}'.format(hHadronPt_finebinned.GetName(), 'rebinned'), pt_bin_array)
        
        # dSigma/(2*pi*pT*dpT*dy*70mb)
        eta_acc = 2*eta_cut
        hHadronPt.Scale(1/(2 * np.pi * self.nEvents * eta_acc * 70.), 'width')
        
        # Get reference histogram
        f_amit = ROOT.TFile(self.filename_amit, 'READ')
        
        hname = ''
        if sqrts == '2760':
            hname = 'hHadronRAA_CMS_2760'
        elif sqrts == '5020':
            hname = 'hHadronRAA_CMS_5020'
                
        hHadronPt_amit = hHadronPt#f_amit.Get(hname)

        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hChHadronCrossSectionPP_Ratio_{}_eta{}{}'.format(sqrts, self.remove_periods(eta_cut), self.file_format))
        xtitle = '#it{p}_{T} (GeV/#it{c})'
        ytitle = '#frac{1}{2 #pi #it{p}_{T} #times 70mb} #frac{d^{2}#sigma}{d#it{p}_{T}d#it{#eta}} #left[(GeV/c)^{-2}#right]'
        self.plot_ratio(hHadronPt, hHadronPt_amit, output_filename, xtitle, ytitle, sqrts, eta_cut, pt_lower, pt_upper, label = 'Hadron')

    #-------------------------------------------------------------------------------------------
    def plot_photon_cross_section(self, sqrts, eta_cut, pt_lower=None, pt_upper=None):

        # Get my histogram
        f = ROOT.TFile(self.filename, 'READ')

        hPhotonPt_eta = f.Get('hPhotonPt_etaScaled')
        
        hPhotonPt_eta.GetYaxis().SetRangeUser(-1.*eta_cut, eta_cut)
        hPhotonPt_finebinned = hPhotonPt_eta.ProjectionX()
        hPhotonPt_finebinned.SetName('{}_{}_{}'.format(hPhotonPt_finebinned, sqrts, eta_cut))
        
        if sqrts == '2760':
            pt_bins = [9.6, 12.0, 14.4, 19.2, 24.0, 28.8, 35.2, 41.6, 48.0, 60.8, 73.6, 86.4, 103.6]
        else:
            pt_bins = [9.6, 12.0, 14.4, 19.2, 24.0, 28.8, 35.2, 41.6, 48.0, 60.8, 73.6, 86.4, 103.6]#, 120.8, 140.0, 165.0, 250.0, 400.0]
        n_pt_bins = len(pt_bins) - 1
        pt_bin_array = array('d', pt_bins)
        hPhotonPt = hPhotonPt_finebinned.Rebin(n_pt_bins, '{}{}'.format(hPhotonPt_finebinned.GetName(), 'rebinned'), pt_bin_array)
        
        # dSigma/(2*pi*pT*dpT*dy*70mb)
        eta_acc = 2*eta_cut
        hPhotonPt.Scale(1/(2 * np.pi * self.nEvents * eta_acc * 70.), 'width')
        
        # Get reference histogram
        f_amit = ROOT.TFile(self.filename_amit, 'READ')
        
        hname = ''
        if sqrts == '2760':
            hname = 'hPhotonRAA_CMS_2760'
        elif sqrts == '5020':
            hname = 'hPhotonRAA_CMS_5020'
                
        hPhotonPt_amit = hPhotonPt#f_amit.Get(hname)

        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hPhotonCrossSectionPP_Ratio_{}_eta{}{}'.format(sqrts, self.remove_periods(eta_cut), self.file_format))
        xtitle = '#it{p}_{T} (GeV/#it{c})'
        ytitle = '#frac{1}{2 #pi #it{p}_{T} #times 70mb} #frac{d^{2}#sigma}{d#it{p}_{T}d#it{#eta}} #left[(GeV/c)^{-2}#right]'
        self.plot_ratio(hPhotonPt, hPhotonPt_amit, output_filename, xtitle, ytitle, sqrts, eta_cut, pt_lower, pt_upper, label = 'Photon')
     
    #-------------------------------------------------------------------------------------------
    def plot_ratio(self, h1, h2, outputFilename, xtitle, ytitle, sqrts, eta_cut, pt_lower, pt_upper, label='Jet'):
        # Create canvas
        cname = 'c_{}_{}'.format(h1.GetName(), h2.GetName())
        c = ROOT.TCanvas(cname,cname,800,600)  # Adjust canvas size to remove the lower panel
        ROOT.SetOwnership(c, False)  # For some reason this is necessary to avoid a segfault...
                                     # Supposedly fixed in https://github.com/root-project/root/pull/3787
        c.cd()
        pad1 = ROOT.TPad('pad1', 'pad1', 0, 0, 1, 1.0)  # Adjust pad size to cover the whole canvas
        pad1.SetBottomMargin(0.15)  # Adjust bottom margin to leave space for x-axis labels
        pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.05)
        pad1.SetTopMargin(0.05)
        if 'Xjgamma' not in label:
            pad1.SetLogy()
        pad1.Draw()
        pad1.cd()

        # Set pad and histogram arrangement
        myPad = ROOT.TPad('myPad', 'The pad', 0, 0, 1, 1)
        myPad.SetLeftMargin(0.15)
        myPad.SetTopMargin(0.04)
        myPad.SetRightMargin(0.04)
        myPad.SetBottomMargin(0.15)

        # Set spectra styles
        # Remove marker settings for h1
        h1.SetMarkerSize(0)
        h1.SetMarkerStyle(0)
        h1.SetMarkerColor(0)
        h1.SetLineStyle(2)
        h1.SetLineWidth(2)
        h1.SetLineColor(ROOT.kRed)
        
        h2.SetMarkerSize(1)
        h2.SetMarkerStyle(21)
        h2.SetMarkerColor(1)
        h2.SetLineStyle(1)
        h2.SetLineWidth(2)
        h2.SetLineColor(1)

        # Draw spectra
        h2.SetXTitle(xtitle)
        h2.GetXaxis().SetTitle(xtitle)
        h2.GetYaxis().SetTitleOffset(1.)
        h2.SetYTitle(ytitle)
        if 'Jet' in label:
            h2.SetMaximum(9e-4)
            h2.SetMinimum(2e-2)
        elif 'Hadron' in label:
            h2.SetMaximum(9e-5)
            h2.SetMinimum(2e-2)
        elif 'D0' in label:
            h2.SetMaximum(9e-4)
            h2.SetMinimum(2e-14)
        elif 'Xjgamma' in label:
            h2.SetMaximum(2.2)
            h2.SetMinimum(-0.1)

        h2.Draw('PE X0 same')  # Draw experimental measurements with markers
        h1.Draw('hist same')  # Draw model results with lines

        # Add horizontal bars to the experimental measurements with markers
        # Create a TGraphErrors object to hold the data points and errors
        graph = ROOT.TGraphErrors()

        # Fill the TGraphErrors with data points and errors
        for bin_idx in range(1, h2.GetNbinsX() + 1):
            if h2.GetBinContent(bin_idx) != 0:  # Check if the bin has non-zero content (marker)
                x_center = h2.GetBinCenter(bin_idx)
                y = h2.GetBinContent(bin_idx)
                y_error = h2.GetBinError(bin_idx)
                x_error = h2.GetBinWidth(bin_idx) / 2  # Half of the bin width as x error

                # Add the point with errors to the TGraphErrors
                point_idx = graph.GetN()
                graph.SetPoint(point_idx, x_center, y)
                graph.SetPointError(point_idx, x_error, y_error)

        # Set marker and line properties for the TGraphErrors
        graph.SetMarkerStyle(20)
        graph.SetMarkerColor(1)
        graph.SetLineColor(1)

        # Draw the TGraphErrors on the canvas
        graph.Draw("PE")  # Draw with error bars and markers

        # Add legends and text
        # system = ROOT.TLatex(0.65, 0.90, 'JETSCAPE')
        # system.SetNDC()
        # system.SetTextSize(0.044)
        # system.Draw()
        
        system2 = ROOT.TLatex(0.65, 0.835, 'pp  #sqrt{#it{s}} = ' + str(float(sqrts)/1000) + ' TeV')
        system2.SetNDC()
        system2.SetTextSize(0.044)
        system2.Draw()

        if 'Jet' in label:
            system3 = ROOT.TLatex(0.49, 0.765, 'Anti-#it{k}_{T} #it{R} = 0.4 | #it{#eta}_{jet}| < ' + str(eta_cut))
        elif 'Hadron' in label:
            system3 = ROOT.TLatex(0.49, 0.765, 'Charged particles | #it{#eta}| < ' + str(eta_cut))
        elif 'Photon' in label:
            system3 = ROOT.TLatex(0.49, 0.765, 'Photon | #it{#eta}| < ' + str(eta_cut))
        elif 'Xjgamma' in label:
            system3 = ROOT.TLatex(0.65, 0.765, 'p_{{T}}^{{#gamma}} = {} - {} GeV'.format(pt_lower, pt_upper))
        elif 'D0' in label:
            system3 = ROOT.TLatex(0.49, 0.765, '(D^{0} + #bar{D^{0}})/2 | #it{#eta}| < ' + str(eta_cut))
        system3.SetNDC()
        system3.SetTextSize(0.044)
        system3.Draw()

        myLegend2pp = ROOT.TLegend(0.65, 0.6, 0.8, 0.68)
        self.setupLegend(myLegend2pp, 0.04)

        # Add entries to the legend with markers
        # Assuming h1 corresponds to model results with lines and h2 corresponds to experimental measurements with markers
        myLegend2pp.AddEntry(h1, "JETSCAPE", "l")  # "l" for line
        myLegend2pp.AddEntry(h2, "ATLAS", "lp")  # "lp" for line with markers

        # Set the separation between legend entries
        myLegend2pp.SetEntrySeparation(0.5)

        myLegend2pp.Draw()

        # Draw axis ticks and labels
        c.Update()  # Required to force drawing of the objects
        c.Modified()
        pad1.Update()
        pad1.Modified()
        c.RedrawAxis()

        line = ROOT.TLine(2., 1, 100., 1)
        line.SetLineColor(920+2)
        line.SetLineStyle(2)
        line.SetLineWidth(4)
        line.Draw()

        c.SaveAs(outputFilename)
        
    # Set legend parameters
    #-------------------------------------------------------------------------------------------
    def setupLegend(self, leg, textSize):
      
        leg.SetTextFont(42);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.SetFillColor(0);
        leg.SetMargin(0.25);
        leg.SetTextSize(textSize);
        leg.SetEntrySeparation(0.5);

    #-------------------------------------------------------------------------------------------
    def setOptions(self):
      
        font = 42

        ROOT.gStyle.SetFrameBorderMode(0)
        ROOT.gStyle.SetFrameFillColor(0)
        ROOT.gStyle.SetCanvasBorderMode(0)
        ROOT.gStyle.SetPadBorderMode(0)
        ROOT.gStyle.SetPadColor(10)
        ROOT.gStyle.SetCanvasColor(10)
        ROOT.gStyle.SetTitleFillColor(10)
        ROOT.gStyle.SetTitleBorderSize(1)
        ROOT.gStyle.SetStatColor(10)
        ROOT.gStyle.SetStatBorderSize(1)
        ROOT.gStyle.SetLegendBorderSize(1)

        ROOT.gStyle.SetDrawBorder(0)
        ROOT.gStyle.SetTextFont(font)
        ROOT.gStyle.SetStatFont(font)
        ROOT.gStyle.SetStatFontSize(0.05)
        ROOT.gStyle.SetStatX(0.97)
        ROOT.gStyle.SetStatY(0.98)
        ROOT.gStyle.SetStatH(0.03)
        ROOT.gStyle.SetStatW(0.3)
        ROOT.gStyle.SetTickLength(0.02,"y")
        ROOT.gStyle.SetEndErrorSize(3)
        ROOT.gStyle.SetLabelSize(0.05,"xyz")
        ROOT.gStyle.SetLabelFont(font,"xyz")
        ROOT.gStyle.SetLabelOffset(0.01,"xyz")
        ROOT.gStyle.SetTitleFont(font,"xyz")
        ROOT.gStyle.SetTitleOffset(1.2,"xyz")
        ROOT.gStyle.SetTitleSize(0.045,"xyz")
        ROOT.gStyle.SetMarkerSize(1)
        ROOT.gStyle.SetPalette(1)

        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)

    # ---------------------------------------------------------------
    # Remove periods from a label
    # ---------------------------------------------------------------
    def remove_periods(self, text):

        string = str(text)
        return string.replace('.', '')

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Executing plot_results_pp.py...')
    print('')
    
    # Define arguments
    parser = argparse.ArgumentParser(description='Generate JETSCAPE events')
    parser.add_argument(
        '-c',
        '--configFile',
        action='store',
        type=str,
        metavar='configFile',
        default='/home/jetscape-user/JETSCAPE-analysis/config/PP_gamma_jet.yaml',
        help='Config file'
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        action="store",
        type=str,
        metavar="inputFile",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Input directory containing JETSCAPE output files",
    )
    parser.add_argument(
        '-o',
        '--outputDir',
        action='store',
        type=str,
        metavar='outputDir',
        default='/home/jetscape-user/JETSCAPE-analysis/TestOutput',
        help='Output directory for output to be written to',
    )

    # Parse the arguments
    args = parser.parse_args()

    # If invalid inputFile is given, exit
    if not os.path.exists(args.inputFile):
        print('File "{0}" does not exist! Exiting!'.format(args.inputFile))
        sys.exit(0)

    analysis = PlotResults(config_file=args.configFile, input_file=args.inputFile, output_dir=args.outputDir)
    analysis.plot_results()
