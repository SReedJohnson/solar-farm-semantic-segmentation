## Solar Farm Semantic Segmentation

This project applies a U-Net convolutional neural network (CNN) to detect and segment solar farms in aerial imagery. It's the final practicum project for the MSDS program at Regis University, developed by Reed Johnson.

## Project Overview

The goal is to automate the identification of solar farms from high-resolution aerial imagery, supporting large-scale energy infrastructure analysis and land use monitoring.

Manual solar farm detection is labor-intensive and not scalable. This project uses semantic segmentation to:
- Detect solar farms at the pixel level
- Estimate total acreage of solar infrastructure
- Provide geospatial visualizations of ground truth vs. model predictions

## Project Structure

├── 01_image_processing_pipeline.ipynb   # Preprocessing and tile/mask cleaning
├── 02_main_unet_pipeline.py             # Model training, stratified split, validation
├── 03_practicum_save_doublets.ipynb     # Visualization: prediction vs ground truth overlays
├── 04_acreage_calculation.ipynb         # Area estimation of solar farm predictions
├── best_model.pth*                      # [ignored] Trained U-Net weights (~118 MB)
├── unet_solarfarm.pth*                  # [ignored] Backup model weights
├── MSDS692_Final_Pres_ReedJohnson.pptx* # [ignored] Final presentation slide deck
├── .gitignore
└── README.md                            # Project documentation

Large files are excluded from GitHub due to size constraints.

## Data Sources

The imagery data was sourced from NCOneMap > NC Orthoimagery 2021 > Robeson County > 01/01/2021.  The county specific download link is here:    
https://www.nconemap.gov/#directdatadownloads.  Cultural data was sourced from Natural Earth.  

NATURAL EARTH. (2025). 1:10M CULTURAL VECTORS. HTTPS://WWW.NATURALEARTHDATA.COM/

NC ONEMAP. (2025). NORTH CAROLINA DEPARTMENT OF INFORMATION TECHNOLOGY, GOVERNMENT DATA ANALYTICS CENTER, CENTER FOR GEOGRAPHIC INFORMATION AND ANALYSIS. HTTPS://WWW.NCONEMAP.GOV/

## Environment Setup

This project was developed across two different environments:
	1.) 	Local Mac (Preprocessing Only) including the following: 
			• QGIS-LTR for visualizing, downsampling, and exporting data
			• GDAL operating on local geospatial environment
			• 01_image_processing_pipeline.ipynb
				• Local file paths and I/O were tailored for macOS development
				• Responsible for tiling, cleaning, and preparing image/mask datasets before upload to RunPod	
	2.) RunPod Cloud (Training, Visualization, Acreage)
			• Executed using a GPU instance on RunPod
			• File paths are set for the /workspace/ directory
			• 02_main_unet_pipeline.py, 03_practicum_save_doublets.ipynb, and 04_acreage_calculation.ipynb

Note: If you’re trying to replicate this project locally or on another platform, you’ll need to adjust the file paths in each script accordingly.

## Preprocessing

Please see the header of 01_image_processing_pipeline for preprocessing instructions using QGIS and GDAL.  Polygon masking and rasterization of solar farms were performed manually and entirely in QGIS.

01_image_processing_pipeline

This includes: 

	•	Image and mask tiling
	•	Saving indices and shape files for tiles
	•	Removal of “no data” padding around image
	•	Removal of partial image and mask tiles

## Training the Model

python 02_main_unet_pipeline.py

This includes:
	•	Stratified splitting (train/val/test)
	•	Dice Loss optimization
	•	Early stopping and LR scheduler
	•	Saves best model as best_model.pth

## Visualizing Results

Use 03_practicum_save_doublets.ipynb to:
	•	Overlay predictions and ground truth
	•	Output side-by-side comparisons
	•	Save presentation-ready .png files

 ## Estimating Solar Farm Acreage
 
04_acreage_calculation.ipynb computes:
	•	Ground truth acreage (from masks)
	•	Predicted acreage (from model outputs)
	•	Outputs total area in both sq ft and acres

## Requirements

Python 3.10+
Pytorch
Geopandas, Rasterio, Shapely
Pandas, NumPy, PIL, Matplotlib, scikit-learn, pathlib 
GDAL and QGIS

## License

For academic and research use only.  
(c) 2025 Reed Johnson| Geology | Data Science

## Results Summary

- The trained U-Net achieved an IoU of ~0.93 and Dice Coefficient of ~0.94 on the test set.
- Total detected solar farm acreage was estimated within ~10% of ground truth.
