# HTEM DB Neural Network Project

This project aims to construct a curated dataset from the publicly available High Throughput Experimental Materials Database (HTEM DB) and to train a neural network to predict thin-film properties. The project is structured into three main components: (1) notebooks to search, compile, and clean data; (2) development and training of a neural network using the resulting dataset; and (3) a future web-based interface for model access and inference.

## 1. Notebooks

- **1_HTEM_Analysis.ipynb**  
  Performs an initial survey of the HTEM database and visualizes relevant relationships.

- **2_Search_Libraries.ipynb**  
  Identifies libraries containing thickness values, XRD angle measurements, XRF compound data, and valid deposition parameters in order to filter for usable libraries.

- **3_Validate_XRD_Angles.ipynb**  
  Determines whether libraries and their constituent samples use standardized XRD angle grids and removes outliers.

- **4_Check_Filtered_Dataset.ipynb**  
  Confirms that the remaining libraries contain standardized XRD measurements.

- **5_Download_Full_Libraries.ipynb**  
  Downloads the filtered libraries to a local system in order to limit repeated API calls.

- **6_Create_Filtered_Dataset.ipynb**  
  Constructs a flattened dataset from local data by associating deposition parameters, compositional data, and measurement outputs while standardizing cross-library documentation.

- **7_Clean_Dataset_Preliminary.ipynb**  
  Removes extraneous null values and constant or near-constant columns.

- **8_Exploratory_Data_Analysis.ipynb**  
  Exploratory data analysis, including identification of strongly correlated features using both pearson and spearman methods, as part of preliminary neural network assessment. 
  Associated features with the 'thickness' property as chosen target.
  Eliminated extreme outlier samples and libraries, decreasing total row/sample size from 9644 samples to 9554 samples.

## 2. Neural Network

## 3. Web Interface

## Configuration

This project uses a local configuration file for system-specific paths.

1. Copy `config/config.example.yaml` to `config/config.yaml`
2. Edit paths in `config/config.yaml` to match your local system
3. Make sure `uv` is installed, then run `uv sync`
4. To view notebooks, run `uv run jupyter lab`

The `config.yaml` file is intentionally ignored by version control.

## Helper Libraries Provided by HTEM

- **library.py** (see the `lib/` folder)  
  Defines a class for querying HTEM data at the library level (44 samples per library).

- **sample.py** (see the `lib/` folder)  
  Defines a class for querying HTEM data at the individual sample level.

  Original license by HTEM included within `lib/` folder.

## Usage Statement

> “Those who choose to utilize the API can download data in full to create their own visualizations and analyses beyond what is available here.”

[HTEM API Documentation](https://htem.nrel.gov/api-docs)

## Credit

The HTEM software was developed by Marcus Schwarting (marcus.schwarting@nrel.gov) and Caleb Phillips (caleb.phillips@nrel.gov) to support the Research Data Initiative and the High Throughput Experimental Materials Database (HTEM DB) at the National Renewable Energy Laboratory (NREL), Golden, Colorado, USA.

## License

This project is licensed under the MIT License. See the LICENSE file for details.