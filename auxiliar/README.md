# Auxiliar helpers

This directory contains auxiliary notebooks and scripts that support the
project but are not needed to run the live demonstration.  The
contents fall into two folders:

* **Graphics and analysis** (`graphics/`): Jupyter notebooks and
  training histories used to explore the raw Traffic4Cast data and
  generate the figures shown in the report.  
  The `train_histories/` folder
  contains JSON files with the training and validation loss curves for
  the baseline and enriched models. 

* **Hugging Face downloader** (`hugging_face_down/download_from_hf.py`):
  The small script that downloads the Traffic4Cast, LBCS and weather
  data from Hugging Face. 

