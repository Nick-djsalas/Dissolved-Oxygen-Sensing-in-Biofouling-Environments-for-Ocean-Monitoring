================================================================================
 COMPANION SOFTWARE — README
================================================================================

 Manuscript:  "Deep Learning-Enabled Dissolved Oxygen Sensing in Biofouling
               Environments for Ocean Monitoring"
               (also referenced as: "A New Paradigm for Robust, Low-Cost
               Dissolved Oxygen Sensing in Biofouling Environments")

 Authors:     Salaris N., Desjardins A., Tiwari M.K.
              Nanoengineered Systems Laboratory,
              UCL Mechanical Engineering & UCL Hawkes Institute

 Public code repository (identical scripts, always up to date):
   https://github.com/Nick21-Sparti/Deep-Learning-Enabled-Dissolved-Oxygen-Sensing-in-Biofouling-Environments-for-Ocean-Monitoring

 Corresponding author: Manish K. Tiwari  (m.tiwari@ucl.ac.uk)

--------------------------------------------------------------------------------
 0. WHAT THIS ARCHIVE IS
--------------------------------------------------------------------------------
 Dear Reviewer/Editor,

 This self-contained archive lets you install, run, and test the full software
 used in the manuscript without any additional downloads (apart from the Python
 packages listed below and, on first run, the pretrained Vision-Transformer
 weights). The same code is also browsable in the public GitHub repository
 linked above; this zip is provided as the offline, "everything-in-one-place"
 copy requested during submission.

 You can run any script by either:
   (a) opening the .py file in your IDE (PyCharm, VS Code, Spyder, ...) and
       pressing the green "Run" button; or
   (b) running it from a terminal:   python <script_name>.py

 All paths inside the scripts are RELATIVE to the script's own location, so the
 pipeline works immediately after unzipping — no path editing required.

--------------------------------------------------------------------------------
 1. CONTENTS OF THIS ARCHIVE
--------------------------------------------------------------------------------
   NatureComms_DO_Sensing_Submission/
   |
   |-- README.txt                          <- this file
   |-- requirements.txt                    <- Python package list
   |
   |-- algae_github_ViT_ensemble.py        <- MAIN METHOD: Physics-Informed
   |                                          Vision Transformer (ViT-PINN) with
   |                                          Leave-One-Day-Out cross-validation
   |                                          and deep ensembling + uncertainty.
   |
   |-- algae_scalability_chrono.py         <- SCALABILITY / TEMPORAL-FORECASTING
   |                                          experiment: strict chronological
   |                                          train/test split + test-time
   |                                          adaptation. (See its own header.)
   |
   |-- algae_github_CLASSICAL_ML.py        <- COMPARISON / CONTROL GROUP: classical
   |                                          Stern-Volmer calibration, "Best
   |                                          Pixels" selection, and the
   |                                          physics-reinforced LightGBM baseline.
   |
   |-- data/
   |   |-- raw/                            <- input for the ViT-ensemble &
   |   |   |-- <DD-MM-YYYY>/                  classical scripts. One folder per
   |   |   |   |-- *_ROI.mp4                   experimental day, each containing
   |   |   |   |-- *_arduino_*.txt             a cropped UV-excitation video, the
   |   |   |   |-- *temperature*.csv           DO sensor log, and a temperature log.
   |   |   |-- ...
   |   |
   |   |-- aligned_experiments/            <- input for the chronological-scaling
   |       |-- Analysis_Results/Data/          script ONLY (different layout; see
   |       |   |-- Exp_<N>_Final*.csv          section 5C). Present only if the
   |       |-- *experiment_<N>*ROInew_Output/  authors included it.
   |           |-- *.mp4
   |
   |-- example_outputs/    (optional)      <- a small set of pre-computed figures
   |                                          and CSVs showing expected results.
   |
   (Folders such as outputs/, cache_features/ and hpo_cache/ are created
    automatically on first run.)

 NOTE ON DATA SIZE: the raw UV videos are large. To keep this archive a
 manageable size, the authors have included an EXAMPLE SUBSET of experimental
 days that is sufficient to install, run, and verify the pipeline end-to-end.
 The complete multi-day dataset is available from the corresponding author on
 request (see the manuscript's Data Availability statement).

--------------------------------------------------------------------------------
 2. SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------
   - Python 3.9 - 3.11 (64-bit).
   - Operating system: Windows, Linux, or macOS.
   - RAM: 16 GB recommended.
   - GPU: a CUDA-capable NVIDIA GPU is recommended for the deep-learning scripts
     but is NOT required. On CPU the deep-learning scripts run correctly, just
     more slowly; the classical baseline script is fast on CPU.
   - Internet access is needed ONCE: the first run of either deep-learning script
     downloads the pretrained ViT weights (~85 MB) via the `timm` library.

--------------------------------------------------------------------------------
 3. INSTALLATION (one-time, ~5 minutes)
--------------------------------------------------------------------------------
   Step 1.  Unzip this archive. Keep all files together — the scripts expect the
            `data/` folder to sit next to them.

   Step 2.  (Recommended) create a clean virtual environment:
              python -m venv venv
              # Windows:
              venv\Scripts\activate
              # Linux / macOS:
              source venv/bin/activate

   Step 3.  Install the dependencies:
              pip install -r requirements.txt

            Required packages:
              numpy, pandas, polars, scipy, scikit-learn, lightgbm,
              opencv-python, matplotlib, seaborn, tqdm, joblib, optuna,
              torch, torchvision, timm

            GPU users: for best performance install the CUDA build of PyTorch
            first, following https://pytorch.org/get-started/locally/, then run
            the command above for the remaining packages.

--------------------------------------------------------------------------------
 4. QUICK START
--------------------------------------------------------------------------------
   The fastest way to verify the software works is the classical baseline script,
   which is CPU-friendly and finishes quickly on the example data:

       python algae_github_CLASSICAL_ML.py

   Then run the main deep-learning method:

       python algae_github_ViT_ensemble.py

   (Optional, and only if aligned data was included — see 5C:)

       python algae_scalability_chrono.py

   Every script prints its progress to the console and writes all figures and
   the underlying numerical data (as .csv) into an `outputs/` folder created
   next to the script.

--------------------------------------------------------------------------------
 5. THE THREE SCRIPTS IN DETAIL
--------------------------------------------------------------------------------

 5A. algae_github_ViT_ensemble.py   — MAIN METHOD
     ............................................................................
     A Physics-Informed Vision Transformer that predicts dissolved oxygen (DO)
     directly from video frames. Key features:
       - ViT feature extractor with four heads: DO regression, biofouling mask,
         per-pixel confidence map, and Stern-Volmer physics-parameter estimation.
       - A hybrid loss combining data (MSE) and a Stern-Volmer physics residual.
       - Leave-One-(Day-)Out Cross-Validation: each experimental day is held out
         in turn, so the reported accuracy reflects performance on a completely
         unseen day.
       - A deep ensemble (default 3 models per fold) that yields a predictive
         uncertainty (ensemble standard deviation) used to self-diagnose
         low-confidence predictions (e.g. severe biofouling).
       - Optuna hyperparameter optimisation (cached after the first run).
     Outputs (in ./outputs/): per-fold convergence plots, parity plots,
     an uncertainty-vs-error diagnostic, red-intensity heatmaps, and a final
     cross-validated results summary (loocv_final_results.csv).
     Reproduces the PINN/ViT results (uncertainty quantification and accuracy
     figures) of the manuscript.

 5B. algae_github_CLASSICAL_ML.py   — COMPARISON / CONTROL GROUP
     ............................................................................
     Implements every classical and physics-REINFORCED baseline, i.e. the
     methods the deep-learning approach is benchmarked against:
       - GA  : Global Average — a single linear Stern-Volmer fit to the mean
               pixel intensity (equivalent to industry-standard single-point
               calibration).
       - Best Pixels : top-N pixels ranked by physics-derived metrics
               (R^2, I0, K_SV, dynamic range, limit of detection), averaged
               into a "super-pixel".
       - LGA : a physics-reinforced LightGBM that receives aggregated physics
               parameters as INPUT FEATURES (note: these are NOT enforced through
               the loss function — that is the defining difference from a PINN).
     Uses the same Leave-One-Day-Out cross-validation as the main method.
     Outputs (in ./outputs/): parity plots, error distributions, MAE bar charts,
     spatial Stern-Volmer parameter heatmaps, LightGBM feature importances, and
     the publication-figure composites. This is the quantitative control that
     demonstrates the limitation of averaging/physics-only methods.

 5C. algae_scalability_chrono.py    — SCALABILITY / TEMPORAL FORECASTING
     ............................................................................
     Demonstrates how the ViT-PINN scales and generalises in CHRONOLOGICAL order:
     models are trained on earlier experiments and tested on a strictly later
     (future) experiment — the realistic ocean-monitoring deployment scenario,
     where biofouling and sensor response drift over time. A full description for
     reviewers is given in the header comment at the top of the script itself.

     IMPORTANT — this script uses a DIFFERENT input layout from the other two.
     It consumes PRE-ALIGNED sensor data, expected under:
         ./data/aligned_experiments/Analysis_Results/Data/Exp_<N>_Final*.csv
         ./data/aligned_experiments/*experiment_<N>*ROInew_Output/*.mp4
     If this folder is not present in the archive, the other two scripts still
     run independently; this experiment can then be reproduced from the public
     GitHub repository together with the aligned data available from the authors.

--------------------------------------------------------------------------------
 6. NOTES ON REPRODUCIBILITY
--------------------------------------------------------------------------------
   - All scripts fix random seeds (SEED = 42) for NumPy, Python, and PyTorch.
   - Results are cached: the first run extracts features and runs HPO, which is
     slow; subsequent runs reuse the caches and are much faster. To force a fresh
     run, set the FORCE_* flags near the top of a script to True.
   - Minor numerical differences between runs are expected on different hardware
     (GPU non-determinism, BLAS/cuDNN versions), but trends and conclusions are
     stable.

--------------------------------------------------------------------------------
 7. TROUBLESHOOTING
--------------------------------------------------------------------------------
   * "FileNotFoundError / No complete experiments found"
       -> Make sure the `data/raw/` folder sits in the SAME directory as the
          script, and that each dated sub-folder contains all three files
          (*_ROI.mp4, *_arduino_*.txt, *temperature*.csv).

   * First deep-learning run hangs at model creation
       -> It is downloading the pretrained ViT weights. Ensure internet access
          for the first run; afterwards it works offline.

   * Out-of-memory on GPU
       -> Lower BATCH_SIZE near the top of the script.

   * LightGBM GPU error
       -> The classical script auto-detects the GPU and falls back to CPU; no
          action needed.

--------------------------------------------------------------------------------
 8. CITATION
--------------------------------------------------------------------------------
   @article{salaris_do_pinn,
     title   = {A New Paradigm for Robust, Low-Cost Dissolved Oxygen Sensing
                in Biofouling Environments},
     author  = {Salaris, Nikolaos and Desjardins, Adrien and Tiwari, Manish K.},
     journal = {},
     year    = {},
     doi     = {}
   }

================================================================================
