
import pandas as pd
import numpy as np
import subprocess

from llmfoundry.data import StreamingFinetuningDataset
from llmfoundry.tokenizers import ChronosTokenizerWrapper
from streaming.base import MDSWriter


def convert_to_mds(load_path: str, save_path: str, num_cols: int) -> None:
    """
    Args:
        ``load_path``: Path to the .csv file to load in the dataset from
        ``save_path``: 
    """
    
    def row_to_numpy(row: pd.Series):
        return row.to_numpy().astype(np.float64)

    df = pd.read_csv(load_path)
    one_col_data = df.apply(row_to_numpy, axis=1)
    one_col_df = pd.DataFrame(one_col_data, columns=['timeseries'])
    
    columns = {'timeseries': f'ndarray:float64:{num_cols}'}
    with MDSWriter(out=save_path, columns=columns) as out:
        for _, row in one_col_df.iterrows():
            out.write({'timeseries': row[0].astype(np.float64)})
        

if __name__ == '__main__':
    # Convert dataset to MDS format
    save_path = '/mnt/workdisk/kushal/llm-foundry/kushal-testing/mds_hospital_20_small_timeseries/'
    # subprocess.run(f'rm -rf {save_path}', shell=True, check=True)
    convert_to_mds(load_path='/mnt/workdisk/kushal/llm-foundry/kushal-testing/hospital_20_small.csv', save_path=save_path, num_cols=30)
    
    # Test that dataset can be read in
    dataset = StreamingFinetuningDataset(tokenizer=ChronosTokenizerWrapper('amazon/chronos-t5-small'), seq_len=30, local=save_path)
    print(f'dataset.num_samples: {dataset.num_samples}')
    
    # dataset = StreamingFinetuningDataset(tokenizer=ChronosTokenizerWrapper('amazon/chronos-t5-small'), max_seq_len=30, local='/mnt/workdisk/kushal/llm-foundry/kushal-testing/test_mds_function/')
    