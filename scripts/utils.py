import pandas as pd
import anndata as ad
from loguru import logger 
import time

import numpy as np
import tifffile

import matplotlib.pyplot as plt
import seaborn as sns

def read_quant(csv_data_path) -> ad.AnnData:
    """
    Read the quantification data from a csv file and return an anndata object
    :param csv_data_path: path to the csv file
    :return: an anndata object
    """
    logger.info(" ---- read_quant : version number 1.0.0 ----")
    time_start = time.time()

    assert csv_data_path.endswith('.csv'), "The file should be a csv file"
    df = pd.read_csv(csv_data_path)

    meta_columns = ['CellID', 'Y_centroid', 'X_centroid',
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'Orientation', 'Extent', 'Solidity']
    assert all([column in df.columns for column in meta_columns]), "The metadata columns are not present in the csv file"

    metadata = df[meta_columns]
    data = df.drop(columns=meta_columns)
    variables = pd.DataFrame(
        index=data.columns,
        data={"math": [column_name.split("_")[0] for column_name in data.columns],
            "marker": ["_".join(column_name.split("_")[1:]) for column_name in data.columns]})

    adata = ad.AnnData(X=data.values, obs=metadata, var=variables)
    logger.info(f" {adata.shape[0]} cells and {adata.shape[1]} variables")
    logger.info(f" ---- read_quant is done, took {int(time.time() - time_start)}s  ----")
    return adata
    
def read_gates(gates_csv_path) -> pd.DataFrame:
    """
    Read the gates data from a csv file and return a dataframe
    """
    logger.info(" ---- read_gates : version number 1.0.0 ----")
    time_start = time.time()
    assert gates_csv_path.endswith('.csv'), "The file should be a csv file"
    gates = pd.read_csv(gates_csv_path)
    
    logger.info("   Filtering out all rows with value 0.0 (assuming not gated)")
    assert "gate_value" in gates.columns, "The column gate_value is not present in the csv file"
    gates = gates[gates.gate_value != 0.0]
    logger.info(f"  Found {gates.shape[0]} valid gates")

    logger.info(f" ---- read_gates is done, took {int(time.time() - time_start)}s  ----")
    return gates

def filter_adata_by_gates(adata: ad.AnnData, gates: pd.DataFrame) -> ad.AnnData:
    """ Filter the adata object by the gates """
    logger.info(" ---- filter_adata_by_gates : version number 1.0.0 ----")
    time_start = time.time()
    assert gates.marker_id.isin(adata.var.index).all(), "Some markers in the gates are not present in the adata object"
    adata = adata[:, gates.marker_id]
    logger.info(f" ---- filter_adata_by_gates is done, took {int(time.time() - time_start)}s  ----")
    return adata

def process_gates_for_sm(gates:pd.DataFrame, sample_id:int) -> pd.DataFrame:
    """ Process gates dataframe to be in log1p scale """
    logger.info(" ---- process_gates_for_sm : version number 1.0.0 ----")
    time_start = time.time()

    gates['log1p_gate_value'] = np.log1p(gates.gate_value)
    gates_for_scimap = gates[['marker_id', 'log1p_gate_value']]
    gates_for_scimap.rename(columns={'marker_id': 'marker', 'log1p_gate_value': sample_id}, inplace=True)

    logger.info(f" ---- process_gates_for_sm is done, took {int(time.time() - time_start)}s  ----")
    return gates_for_scimap


def lazy_image_check(image_path):
    """ Check the image metadata without loading the image """
    logger.info(" ---- lazy_image_check : version number 1.0.0 ----")
    time_start = time.time()

    with tifffile.TiffFile(image_path) as image:
        # Getting the metadata
        shape = image.series[0].shape
        dtype = image.pages[0].dtype

        n_elements = np.prod(shape)
        bytes_per_element = dtype.itemsize
        estimated_size_bytes = n_elements * bytes_per_element
        estimated_size_gb = estimated_size_bytes / 1024 / 1024 / 1024 
        
        logger.info(f"Image shape is {shape}")
        logger.info(f"Image data type: {dtype}")
        logger.info(f"Estimated size: {estimated_size_gb:.4g} GB")

    logger.info(f" ---- lazy_image_check is done, took {int(time.time() - time_start)}s  ----")


def filter_by_ratio(df, end_cycle, start_cycle, label="DAPI", min_ratio=0.5, max_ratio=1.05):
    """ Filter cells by ratio """

    logger.info(" ---- filter_by_ratio : version number 1.0.0 ----")
    time_start = time.time()

    df[f'{label}_ratio'] = df[end_cycle] / df[start_cycle]
    df[f'{label}_ratio_pass'] = (df[f'{label}_ratio'] > min_ratio) & (df[f'{label}_ratio'] < max_ratio)

    # print out statistics
    logger.info(f"Number of cells with {label} ratio < {min_ratio}: {sum(df[f'{label}_ratio'] < min_ratio)}")
    logger.info(f"Number of cells with {label} ratio > {max_ratio}: {sum(df[f'{label}_ratio'] > max_ratio)}")
    logger.info(f"Number of cells with {label} ratio between {min_ratio} and {max_ratio}: {sum(df[f'{label}_ratio_pass'])}")
    logger.info(f"Percentage of cells filtered out: {round(100 - sum(df[f'{label}_ratio_pass'])/len(df)*100,2)}%")

    # plot histogram

    sns.histplot(df[f'{label}_ratio'], color='blue')
    plt.yscale('log')

    plt.axvline(min_ratio, color='black', linestyle='--', alpha=0.5)
    plt.axvline(max_ratio, color='black', linestyle='--', alpha=0.5)
    plt.text(max_ratio + 0.05, 650, f"cells that gained >{int(max_ratio*100-100)}% {label}", fontsize=9, color='black')
    plt.text(min_ratio - 0.05, 650, f"cells that lost >{int(min_ratio*100-100)}% {label}", fontsize=9, color='black', horizontalalignment='right')

    plt.ylabel('cell count')
    plt.xlabel(f'{label} ratio (last/cycle)')
    plt.xlim(min_ratio-1, max_ratio+1)
    plt.show()

    logger.info(f" ---- filter_by_ratio is done, took {int(time.time() - time_start)}s  ----")

    return df