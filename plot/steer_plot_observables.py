"""
  Macro to steer calculation of observables from final_state_hadrons

  The workflow is as follows:

   (1) For each final_state_hadrons parquet file, compute observables with analyze_events_STAT.py.
       This will produce an "observables" parquet file for each final_state_hadrons file.

       In the AA case, the centrality is retrieved for each file, and the observables filled only for relevant centralities.
         (the centrality for each file is recorded in Run*_info.yaml in the index_to_hydro_event map,
         e.g. '3: cent_00_01/08' for file 3 0-1%)

       Usually this step should be done on XSEDE in the same job as the event generation.
       In case we need to compute observables manually, we provide an option to loop over all final_state_hadron
       files within a specified directory.

   (2) For each observable parquet file, fill histograms with histogram_results_STAT.py.
       This will produce a "histograms" ROOT file for each observable parquet file.

       In the AA case, the centrality is retrieved from the cross-section parquet file, and the observables filled into the appropriate centrality-binned histogram.
       In the pp case, all centrality-binned histograms are filled with the same observables.

       The histogram binnings are retrieved from experimental data. We support:
         - HEPData (preferred)
             - We support two different HEPData file structures:
               (1) Centralities in same dir, different hname/gname
               (2) Centralities in different dir, same hname/gname
               See the STAT_{sqrts}.yaml files for documentation on how to specify the centralities to be looped over.
         - Custom format -- see STAT_{sqrts}.yaml files

       Usually this step should be done on XSEDE in the same job as the event generation.
       In case we need to histogram observables manually, we provide an option to loop over all observable
       files within a specified directory.

   (3) Merge all histograms together from Step 2.

       TODO: We determine the set of histogram files to merge based on
       the analysis name (e.g. 'Analysis0'), which contains a set of runs (e.g. 'Run0001', 'Run0002', ...),
       specified at https://github.com/JETSCAPE/STAT-XSEDE-2021/tree/main/docs/DataManagement.
       We loop over all runs from all facilities in the Analysis, including all sqrt{s}.

   (4) Plot final observables and write table for input to Bayesian analysis.

       In the AA case, we plot the AA/pp ratios
       In the pp case, we plot the pp distributions

       We support the same specifications for retrieval of experimental as described for histogram binning.
       See the STAT_{sqrts}.yaml files for documentation on how to specify the centralities to be looped over.

       TODO: Implement machinery to write and aggregate tables for Bayesian analysis 

  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

# General
import os
import subprocess
import shutil

# ---------------------------------------------------------------
def main():

    # Specify input directory containing final_state_hadrons files
    sqrts = 5020

    soft_sector_analysis = True

    final_state_hadron_dir = '/home/jetscape-user/JETSCAPE-STAT-output/Dirac_mar31/Run1020_june23'
    final_state_hadron_files = [file for file in os.listdir(final_state_hadron_dir) if 'jetscape' in file]
    system = final_state_hadron_files[0].split('_')[1]

    # If AA, supply pp reference results in order to construct RAA
    pp_reference_filename = '/home/jetscape-user/JETSCAPE-STAT-output/STAT_designs/plot/5020_pp/final_results.root'

    # Note: the construction of observables and histograms is usually done on XSEDE,
    #       and only the merging/plotting step is needed to be run locally
    construct_observables = False
    construct_histograms = False
    merge_histograms = False
    plot_histograms = True

    #-----------------------------------------------------------------
    # Loop through final_state_hadron files, and construct observables
    if construct_observables:

        # Clear output directory
        observables_dir = os.path.join(final_state_hadron_dir, 'observables')
        if os.path.exists(observables_dir):
            shutil.rmtree(observables_dir)

        for file in os.listdir(final_state_hadron_dir):
            if 'final_state_hadrons' in file and file.endswith('.parquet'):
                cmd = f'python3 jetscape_analysis/analysis/analyze_events_STAT.py -c config/STAT_{sqrts}.yaml -i {final_state_hadron_dir}/{file} -o {observables_dir}'
                print(cmd)
                subprocess.run(cmd, check=True, shell=True)

    #-----------------------------------------------------------------
    # Loop through observable files, and construct histograms
    if construct_histograms:

        inputdir = os.path.join(final_state_hadron_dir, 'observables')
        outputdir = os.path.join(final_state_hadron_dir, 'histograms')

        for file in os.listdir(inputdir):
            if 'observables' in file:
                cmd = f'python3 plot/histogram_results_STAT.py -c config/STAT_{sqrts}.yaml -i {inputdir}/{file} -o {outputdir}'
                print(cmd)
                subprocess.run(cmd, check=True, shell=True)

        if soft_sector_analysis:
            # Analyze soft sector parquet files and obtain histogram root files
            inputdir = final_state_hadron_dir
            outputdir = final_state_hadron_dir

            # Find all parquet files in the soft sector with "QnVector" in the name
            parquet_files = [
                file for file in os.listdir(inputdir)
                if 'QnVector' in file and file.endswith('.parquet')
            ]

            for file in parquet_files:
                input_path = os.path.join(inputdir, file)
                cmd = f'python3 jetscape_analysis/analysis/analyze_events_soft_ref.py -c config/STAT_{sqrts}.yaml -i {input_path} -o {outputdir}'
                print(cmd)
                subprocess.run(cmd, check=True, shell=True)

    #-----------------------------------------------------------------
    # Merge histograms
    if merge_histograms:

        # Merge final state histograms (hard sector)
        inputdir = os.path.join(final_state_hadron_dir, 'histograms')
        outputdir = os.path.join(final_state_hadron_dir, 'plot')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        ROOT_files = os.listdir(inputdir)
        fname = f'histograms_{system}_{sqrts}_merged.root'
        cmd = f'hadd -f {os.path.join(outputdir, fname)}'
        for file in ROOT_files:
            if '.root' in file:
                cmd += f' {os.path.join(inputdir, file)}'
        subprocess.run(cmd, check=True, shell=True)

        # Soft sector histogram merging (e.g., QnVector-based files)
        if soft_sector_analysis:
            # Find all files with "QnVector" and ".root" in their names
            qnvector_files = [
                file for file in os.listdir(final_state_hadron_dir)
                if 'QnVector' in file and file.endswith('.root')
            ]

            # Define output file for soft sector
            soft_fname = f'histograms_{system}_{sqrts}_soft_reference.root'
            soft_reference_filename = os.path.join(outputdir, soft_fname)

            # Construct and run the hadd command
            qn_merge_cmd = f'hadd -f {soft_reference_filename}'
            for file in qnvector_files:
                qn_merge_cmd += f' {os.path.join(final_state_hadron_dir, file)}'
            subprocess.run(qn_merge_cmd, check=True, shell=True)

        else:
            soft_reference_filename = ''

    #-----------------------------------------------------------------
    # Plot histograms
    if plot_histograms:

        inputdir = os.path.join(final_state_hadron_dir, 'plot')
        fname = f'histograms_{system}_{sqrts}_merged.root'
        input_file = os.path.join(inputdir, fname)

        if not merge_histograms:
            if soft_sector_analysis:
                soft_fname = f'histograms_{system}_{sqrts}_soft_reference.root'
                soft_reference_filename = os.path.join(inputdir, soft_fname)
            else:
                soft_reference_filename = ''

        if system == 'pp':
            cmd = f'python3 plot/plot_results_STAT.py -c config/STAT_{sqrts}.yaml -i {input_file}'
        else:
            cmd = f'python3 plot/plot_results_STAT.py -c config/STAT_{sqrts}.yaml -i {input_file} -r {pp_reference_filename}'
            if soft_sector_analysis:
                cmd += f' -s {soft_reference_filename}'

        print(cmd)
        subprocess.run(cmd, check=True, shell=True)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print()
    main()
