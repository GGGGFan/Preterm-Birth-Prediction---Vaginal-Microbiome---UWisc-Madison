FROM python:3.7

# Install dependencies
RUN pip install pandas \
  numpy \
  sklearn \
  shap \
  lightgbm \
  optuna \
  scipy \
  scikit-bio

# Make folders
RUN mkdir /data_train
RUN mkdir /filtered_fts

# Copy files
COPY ./data_train/metadata.csv ./data_train
COPY ./data_train/alpha_diversity.csv ./data_train
COPY ./data_train/cst_valencia.csv ./data_train
COPY ./data_train/phylotype_nreads.1e0.csv ./data_train
COPY ./data_train/phylotype_nreads.5e_1.csv ./data_train
COPY ./data_train/taxonomy_nreads.species.csv ./data_train
COPY ./data_train/taxonomy_nreads.genus.csv ./data_train
COPY ./data_train/taxonomy_nreads.family.csv ./data_train
COPY ./filtered_fts/phylotype_05_features.txt ./filtered_fts
COPY ./filtered_fts/phylotype_1_features.txt ./filtered_fts
COPY ./filtered_fts/family_features.txt ./filtered_fts
COPY ./filtered_fts/genus_features.txt ./filtered_fts
COPY ./filtered_fts/species_features.txt ./filtered_fts
COPY ./run_model.py .

# Add executable permission
RUN chmod +x run_model.py

# Specify what file to run
ENTRYPOINT ["python", "run_model.py"]
