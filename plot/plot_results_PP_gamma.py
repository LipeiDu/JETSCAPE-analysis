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

# Data analysis and plotting
import ROOT
from array import *
import numpy as np

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)


################################################################
class PlotResults():

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, output_dir="", **kwargs):
        super(PlotResults, self).__init__(**kwargs)
        
        if not output_dir.endswith("/"):
            output_dir = output_dir + "/"
        self.output_dir = output_dir
        
        self.nEvents = 90000
        self.file_format = '.pdf'
        
        # Filename from my output
        self.filename = 'AnalysisResults_{}_weakdecays{}.root'
        
        # Filenames for reference data
        self.filename_amit = '/home/jetscape-user/JETSCAPE-analysis/plot/AnalysisResults_5020_weakdecaysON.root'
        self.filename_gojko = '/home/jetscape-user/JETSCAPE-analysis/plot/AnalysisResults_5020_weakdecaysON.root'

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

        self.plot_D0_cross_section('5020', eta_cut = 1.0)
        
    #-------------------------------------------------------------------------------------------
    def plot_jet_cross_section(self, sqrts, eta_cut):
    
        # Get my histogram
        filename = os.path.join(self.output_dir, self.filename.format(sqrts, 'ON'))
        f = ROOT.TFile(filename, 'READ')

        hJetPt_eta = f.Get('hJetPt_eta_R0.4Scaled')
        
        hJetPt_eta.GetYaxis().SetRangeUser(-1.*eta_cut, eta_cut)
        hJetPt_finebinned = hJetPt_eta.ProjectionX()
        hJetPt_finebinned.SetName('{}_{}_{}'.format(hJetPt_finebinned, sqrts, eta_cut))
        
        pt_bins = []
        if sqrts == '2760':
            pt_bins = [70, 80, 90, 100, 110, 130, 150, 170, 190, 210, 240, 270, 300]
        elif sqrts == '5020':
            pt_bins = [40, 50, 63, 79, 100, 126, 158, 200, 251, 316, 398, 501, 631, 800, 1000]
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
                
        hJetPt_amit = f_amit.Get(hname)

        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hJetCrossSectionPP_Ratio_{}_eta{}{}'.format(sqrts, self.remove_periods(eta_cut), self.file_format))
        xtitle = '#it{p}_{T,jet} (GeV/#it{c})'
        ytitle = '#frac{d^{2}#sigma}{d#it{p}_{T,jet}d#it{#eta}_{jet}} #left[mb (GeV/c)^{-1}#right]'
        self.plot_ratio(hJetPt, hJetPt_amit, output_filename, xtitle, ytitle, sqrts, eta_cut, label = 'Jet')

    #-------------------------------------------------------------------------------------------
    def plot_ch_hadron_cross_section(self, sqrts, eta_cut):

        # Get my histogram
        filename = os.path.join(self.output_dir, self.filename.format(sqrts, 'ON'))
        f = ROOT.TFile(filename, 'READ')

        hHadronPt_eta = f.Get('hChHadronPt_etaScaled')
        
        hHadronPt_eta.GetYaxis().SetRangeUser(-1.*eta_cut, eta_cut)
        hHadronPt_finebinned = hHadronPt_eta.ProjectionX()
        hHadronPt_finebinned.SetName('{}_{}_{}'.format(hHadronPt_finebinned, sqrts, eta_cut))
        
        if sqrts == '2760':
            pt_bins = [9.6, 12.0, 14.4, 19.2, 24.0, 28.8, 35.2, 41.6, 48.0, 60.8, 73.6, 86.4, 103.6]
        else:
            pt_bins = [9.6, 12.0, 14.4, 19.2, 24.0, 28.8, 35.2, 41.6, 48.0, 60.8, 73.6, 86.4, 103.6, 120.8, 140.0, 165.0, 250.0, 400.0]
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
                
        hHadronPt_amit = f_amit.Get(hname)

        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hChHadronCrossSectionPP_Ratio_{}_eta{}{}'.format(sqrts, self.remove_periods(eta_cut), self.file_format))
        xtitle = '#it{p}_{T} (GeV/#it{c})'
        ytitle = '#frac{1}{2 #pi #it{p}_{T} #times 70mb} #frac{d^{2}#sigma}{d#it{p}_{T}d#it{#eta}} #left[(GeV/c)^{-2}#right]'
        self.plot_ratio(hHadronPt, hHadronPt_amit, output_filename, xtitle, ytitle, sqrts, eta_cut, label = 'Hadron')
        
    #-------------------------------------------------------------------------------------------
    def plot_D0_cross_section(self, sqrts, eta_cut):

        # Get my histogram
        filename = os.path.join(self.output_dir, self.filename.format(sqrts, 'OFF'))
        f = ROOT.TFile(filename, 'READ')

        hD0Pt_eta = f.Get('hD0Pt_etaScaled')
        
        hD0Pt_eta.GetYaxis().SetRangeUser(-1.*eta_cut, eta_cut)
        hD0Pt_finebinned = hD0Pt_eta.ProjectionX()
        hD0Pt_finebinned.SetName('{}_{}_{}'.format(hD0Pt_finebinned, sqrts, eta_cut))
        
        pt_bins = [2., 3., 4., 5., 6., 8., 10., 12.5, 15., 20., 25., 30., 40., 60., 100.]
        n_pt_bins = len(pt_bins) - 1
        pt_bin_array = array('d', pt_bins)
        hD0Pt = hD0Pt_finebinned.Rebin(n_pt_bins, '{}{}'.format(hD0Pt_finebinned.GetName(), 'rebinned'), pt_bin_array)
        
        # dN/(2pi pT dpT) of (D0+anti-D0)/2
        # dSigma/(2*pi*pT*dpT*dy*70mb)
        #eta_acc = 2*eta_cut
        hD0Pt.Scale(1/(2 * np.pi * self.nEvents * 70. * 2), 'width')
        
        # Get reference histogram
        f_gojko = ROOT.TFile(self.filename_gojko, 'READ')
        hname = 'hD0Pt'
        hD0Pt_gojko = f_gojko.Get(hname)

        # Plot the ratio
        output_filename = os.path.join(self.output_dir, 'hD0CrossSectionPP_Ratio_{}_eta{}{}'.format(sqrts, self.remove_periods(eta_cut), self.file_format))
        xtitle = '#it{p}_{T} (GeV/#it{c})'
        ytitle = '#frac{1}{2 #pi #it{p}_{T} #times 70mb} #frac{d^{2}#sigma}{d#it{p}_{T}d#it{#eta}} #left[(GeV/c)^{-2}#right]'
        self.plot_ratio(hD0Pt, hD0Pt_gojko, output_filename, xtitle, ytitle, sqrts, eta_cut, label = 'D0')
        
    #-------------------------------------------------------------------------------------------
    # Plot ratio h1/h2
    def plot_ratio(self, h1, h2, outputFilename, xtitle, ytitle, sqrts, eta_cut, label='Jet'):

        # Create canvas
        cname = 'c_{}_{}'.format(h1.GetName(), h2.GetName())
        c = ROOT.TCanvas(cname,cname,800,850)
        ROOT.SetOwnership(c, False) # For some reason this is necessary to avoid a segfault...some bug in ROOT or pyroot
                                    # Supposedly fixed in https://github.com/root-project/root/pull/3787
        c.cd()
        pad1 = ROOT.TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0)
        pad1.SetLeftMargin(0.2)
        pad1.SetRightMargin(0.05)
        pad1.SetTopMargin(0.05)
        pad1.SetLogy()
        pad1.Draw()
        pad1.cd()

        # Set pad and histo arrangement
        myPad = ROOT.TPad('myPad', 'The pad',0,0,1,1)
        myPad.SetLeftMargin(0.22)
        myPad.SetTopMargin(0.04)
        myPad.SetRightMargin(0.04)
        myPad.SetBottomMargin(0.15)

        # Set spectra styles
        h1.SetMarkerSize(3)
        h1.SetMarkerStyle(33)
        h1.SetMarkerColor(600-6)
        h1.SetLineStyle(1)
        h1.SetLineWidth(2)
        h1.SetLineColor(600-6)
        
        h2.SetMarkerSize(2)
        h2.SetMarkerStyle(21)
        h2.SetMarkerColor(1)
        h2.SetLineStyle(1)
        h2.SetLineWidth(2)
        h2.SetLineColor(1)

        # Draw spectra
        h2.SetXTitle(xtitle)
        h2.GetYaxis().SetTitleOffset(2.2)
        h2.SetYTitle(ytitle)
        if 'Jet' in label:
            h2.SetMaximum(9e-4)
            h2.SetMinimum(2e-13)
        elif 'Hadron' in label:
            h2.SetMaximum(9e-5)
            h2.SetMinimum(2e-17)
        elif 'D0' in label:
            h2.SetMaximum(9e-4)
            h2.SetMinimum(2e-14)

        h2.Draw('PE X0 same')
        h1.Draw('PE X0 same')

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Add legends and text
        # # # # # # # # # # # # # # # # # # # # # # # #
        system = ROOT.TLatex(0.49,0.90,'JETSCAPE')
        system.SetNDC()
        system.SetTextSize(0.044)
        system.Draw()
        
        system2 = ROOT.TLatex(0.49,0.835,'pp  #sqrt{#it{s}} = ' + str(float(sqrts)/1000) + ' TeV')
        system2.SetNDC()
        system2.SetTextSize(0.044)
        system2.Draw()

        if 'Jet' in label:
            system3 = ROOT.TLatex(0.49 ,0.765, 'Anti-#it{k}_{T} #it{R} = 0.4 | #it{#eta}_{jet}| < ' + str(eta_cut))
        elif 'Hadron' in label:
            system3 = ROOT.TLatex(0.49 ,0.765, 'Charged particles | #it{#eta}| < ' + str(eta_cut))
        elif 'D0' in label:
            system3 = ROOT.TLatex(0.49 ,0.765, '(D^{0} + #bar{D^{0}})/2 | #it{#eta}| < ' + str(eta_cut))
        system3.SetNDC()
        system3.SetTextSize(0.044)
        system3.Draw()

        myLegend2pp = ROOT.TLegend(0.65,0.6,0.8,0.68)
        self.setupLegend(myLegend2pp,0.04)
        myLegend2pp.AddEntry(h1,'JETSCAPE 3.0','Pe')
        myLegend2pp.AddEntry(h2,'ReleaseAApaper','Pe')
        myLegend2pp.Draw()

        c.cd()
        pad2 = ROOT.TPad('pad2', 'pad2', 0, 0.05, 1, 0.3)
        pad2.SetTopMargin(0)
        pad2.SetBottomMargin(0.35)
        pad2.SetLeftMargin(0.2)
        pad2.SetRightMargin(0.05)
        pad2.Draw()
        pad2.cd()

        # plot ratio
        hRatio = h1.Clone()
        hRatio.SetName('hRatio_{}'.format(cname))
        hRatio.Divide(h2)
        hRatio.SetMarkerStyle(21)
        hRatio.SetMarkerSize(2)

        hRatio.GetXaxis().SetTitleSize(30)
        hRatio.GetXaxis().SetTitleFont(43)
        hRatio.GetXaxis().SetTitleOffset(4.)
        hRatio.GetXaxis().SetLabelFont(43)
        hRatio.GetXaxis().SetLabelSize(20)
        hRatio.GetXaxis().SetTitle(xtitle)

        hRatio.GetYaxis().SetTitle('3.0 / AApaper')
        hRatio.GetYaxis().SetTitleSize(20)
        hRatio.GetYaxis().SetTitleFont(43)
        hRatio.GetYaxis().SetTitleOffset(2.2)
        hRatio.GetYaxis().SetLabelFont(43)
        hRatio.GetYaxis().SetLabelSize(20)
        hRatio.GetYaxis().SetNdivisions(505)
        
        hRatio.SetMinimum(0.5)
        hRatio.SetMaximum(1.7)
        if 'D0' in label:
            hRatio.SetMinimum(0.1)
        hRatio.Draw('P E')
        
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

    analysis = PlotResults(output_dir=args.outputDir)
    analysis.plot_results()
