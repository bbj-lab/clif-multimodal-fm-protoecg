I want to build a multi-modal foundation modal based on: https://github.com/bbj-lab/protoecg-fm-multi-modal

But instead of using the raw MIMIC data we will use the CLIF version (in ./data/, all start with clif_), the CLIF data model is described here: https://clif-icu.com/data-dictionary we'll use version 2.1.0

There are several key steps:
  1. Mapping CLIF to a MEDS-like format where we have a sequence list 
  2. Defining Labels of interest (use the existing repository for inspiration + as necessary to define labels)
    Labels of interest should be in 2 categories: 
      1. CLIF-only required
      2. Requires additional EMR data (e.g. MIMIC Concepts or other data which don't make it into CLIF)
  3. Inserting the labels of interest at their first occurence for a particular hospitalization
  4. Inserting the ECG prototype information at their occurrence (./data/ecg_prototypes_with_shifted_dates.csv)
  5. Adding the clock and gap tokens 
  6. Model training (with filtering of the ECG events so we have 3 routes - no_ecg, fusion, all_branches)
  7. Inference (ideally with vLLM for speed) for a horizon period (stopping at discharge, death or after the time period has ended, e.g., if horizon period is 48 hours, stop after 6 of the 8-hour clock tokens)
  8. Evaluation - window-based AUROCs 

  Think deeply and come up with a comprehensive implementation plan, asking thorough questions where you need more info. 
