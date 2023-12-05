import torch
from torch.utils.data import Dataset

class WeatherData(Dataset):
  """
  Wrapper used to load the weather dataset in PyTorch. 
  """

  def __init__(self, data, seq_len, filter_cols, device):
    """
    Instantiate the dataset and initialize it

    Parameters
    ----------
    data : pandas dataframe containing the entire weather dataset
    seq_len : the length of the sequence corresponding to the number of prior 
      weather samples used for prediction.
    filter_cols : weather parameters included as inputs, providing supplementary 
      information for the model.
    device : either 'cpu' or 'gpu'
    """

    # Filter the data based on the provided columns
    filtered_data = torch.tensor(data.loc[:,filter_cols].values, device=device)
    
    self.device = device
    self.X_data, self.Y_data = self.prepare_dataset(filtered_data, seq_len, filter_cols)

  def prepare_dataset(self, data, seq_len):
    """
    Prepare the dataset by creating a all the possible sequences of length
    seq_len.

    Parameters
    ----------
    data : pandas dataframe containing the entire weather dataset 
    seq_len : the length of the sequence corresponding to the number of prior 
      weather samples used for prediction.
    """
    X = torch.zeros((data.shape[0] - seq_len, seq_len, len(data.columns)), device=self.device)
    Y = []

    # Prepare the sequences of length seq_len
    for i in range(data.size(0) - seq_len):
      X[i,:] = data[i:i+seq_len,:]
      Y.append(data[i+seq_len,0])
    return X, torch.tensor(Y, device=self.device)

  def __len__(self):
    return self.X_data.size(0)

  def __getitem__(self, idx):
    return self.X_data[idx], self.Y_data[idx]