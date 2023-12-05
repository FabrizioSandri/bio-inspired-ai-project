import torch

class WeatherLSTM(torch.nn.Module):
  """
  This is the implementation of the LSTM model
  """

  def __init__(self, input_size, hidden_size, num_layers):
    """
    Instantiate a LSTM model configured with a predefined input size, hidden
    size and number of layers.

    Parameters
    ----------
    input_size : the size of the feature vector corresponding to the number of 
      weather parameters using for forecasting
    hidden_size : the number of hidden units or neurons in each LSTM layer,
      determining the capacity of the model to capture complex patterns. 
    num_layers : the number of stacked LSTM layers.
    """
    super().__init__()
    
    self.hidden_size = hidden_size

    self.lstm = torch.nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True
    )

    self.linear = torch.nn.Linear(hidden_size, 1)

  def initialize_weights(self, weight_list):
    """
    Instantiate the weights of the entire model from a given list of weights

    Parameters
    ----------
    weight_list : a list of weights used to initialize the model
    """
    if weight_list.numel() != sum(p.numel() for p in self.parameters()):
        raise ValueError("Length of weight_list must match the total number of parameters in the model.")

    # Iterate over model parameters and set values from the weight_list
    index = 0
    for param in self.parameters():
        param.data = weight_list[index:index + param.numel()].view_as(param)
        index += param.numel()
        
  def forward(self, x):
    """
    Instantiate the weights of the entire model from a given list of weights.
    Returns a N-dimensional tensor

    Parameters
    ----------
    x : tensor of shape (N,L,H) where N is the batch size, L the sequence length
      corresponding to the number of prior weather samples used for prediction
      and H is the input size(input_size)
    """
    out, (hn, cn) = self.lstm(x)
    out = self.linear(out[:, -1, :]).squeeze(-1)

    return out