

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def prepare_data(training_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:


    mmScaler = MinMaxScaler(feature_range=(-1, 1))
    standardScaler = StandardScaler()

    # Make a copy of new_data to preserve the original
    transformed_data = new_data.copy()


    # Apply MinMaxScaler to certain columns
    for pcr in ['PCR_01', 'PCR_03', 'PCR_04', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09']:
        mmScaler.fit(pd.DataFrame(training_data, columns=[pcr]))
        transformed_pcr = mmScaler.transform(pd.DataFrame(new_data, columns=[pcr]))
        transformed_data[pcr] = transformed_pcr.flatten()

    # Apply StandardScaler to other columns
    for pcr in ['PCR_02', 'PCR_05', 'PCR_10']:
        standardScaler.fit(pd.DataFrame(training_data, columns=[pcr]))
        transformed_pcr = standardScaler.transform(pd.DataFrame(new_data, columns=[pcr]))
        transformed_data[pcr] = transformed_pcr.flatten()

    # Ensure the function returns the transformed DataFrame
    return transformed_data


