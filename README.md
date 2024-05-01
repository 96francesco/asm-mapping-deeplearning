Artisanal and Small-Scale Mining (ASM) mapping using deep learning and high-resolution satellite imagery
==============================
This project's aim was to develop deep learning models to map artisanal and small-scale mining (ASM) sites using high-resolution satellite imagery. The models were based on the U-Net architecture and were trained on a dataset of labeled images of ASM sites and non-ASM sites (binary ground truth). One important goal was to compare standalone models and data fusion models using SAR (Sentinel-1) and optical (Planet-NICFI) imagery. As data fusion approach, Late Fusion was adopted.

This project was developed as part of my MSc thesis in Geo-Information Science at Wageningen University & Research ([GRS lab](https://www.wur.nl/en/Research-Results/Chair-groups/Environmental-Sciences/Laboratory-of-Geo-information-Science-and-Remote-Sensing.htm)). The thesis was supervised by dr. [Robert Masolele](https://www.wur.nl/en/persons/robert-masolele.htm) and dr. [Johannes Reiche](https://www.wur.nl/nl/en/personen/johannes-reiche.htm). 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── predictions
    │   └── checkpoints
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to handle the data
        │   
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions and inference
        │   ├── predict_model.py
        │   └── train_model.py
        |   └── test_model.py
        │
        ├── optimization   <- Scripts to run hyperparameter optimization with Optuna
        │   
        │   
        └── visualization  <- Scripts to create exploratory and results oriented visualizations



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Installation and usage
------------

Feel free to clone this repository and use the code for your own projects. The code is structured in a way that allows you to easily train models, make predictions and visualize the results. Obviously, you will need to adapt the code to your specific use case, specifically regarding the directory structure and the data format. Some parts of the code, especially for handling the Late Fusion mode, are not optimal and could be improved; so any contribution is welcome! For any help or information, do not hesitate to contact me.

Here some steps to get you started:
- Clone the repository
    ```bash
    git clone https://github.com/96francesco/asm-mapping-deeplearning.git
    ```
    ```bash
    cd asm-mapping-deeplearning
    ```
- Use pip to install the required packages
    ```bash
    pip install -r requirements.txt
    ```
- In order to install PyTorch and torchvision with CUDA support, run the command below (check compatibility with CUDA version!)
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
- If any problem occurs during the installation of rasterio, try to run this

    ```bash
    conda install -c conda-forge rasterio
    ```