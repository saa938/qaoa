Layout
  src/      all scripts (run these; they resolve data/results/figures relative to their own file location)
  data/     input data (MaxCutMAQAOAData.csv, graphs.csv, AshayMAQAOAData/)
  results/  generated json/csv, and results/shells/ for the cached .npz minima
  figures/  generated png plots

Scripts can be run from any working directory, e.g. either of:
  python src/verify_all.py
  cd src && python verify_all.py

Run order for latest experiment
1. python src/verify_all.py
2. python src/lowest_norm_survey.py
3. python src/quantization_final.py
4. python src/sphere_uniformity.py
5. python src/layers_corrected.py 13 1
   python src/layers_corrected.py 13 2
   python src/layers_corrected.py 13 3
