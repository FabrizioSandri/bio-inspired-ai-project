import matplotlib.pyplot as plt
import torch

class Evaluate():
  """
  Class used to measure the performance of the algorithms
  """

  def __init__(self, device):
    self.device = device

  def evaluateMaeLSTM(self, model, best_individual, dataset, plot=False):
    """
    Calculate the Mean Absolute Error of the LSTM model passed as parameter by
    testing it on the provided dataset.

    Parameters
    ----------
    model : the model, in this case an instance of the WeatherLSTM class
    best_individual : the best individual found by the aforementioned model
    dataset : Pandas dataframe containing all the data of size (N,F) where N is
      the number of samples and F is the number of weather features
    plot : weather or not to plot the prediction compared to the baseline and
      the ground truth
    """
    curve_y = []        # Ground truth
    curve_lstm = []     # LSTM prediction curve
    curve_last = []     # Prediction using the last temperature recorded(baseline)


    data = torch.tensor(dataset.values, dtype=torch.float32, device=self.device)

    # Mean Absolute Error values
    mae_lstm = 0
    mae_last = 0

    starting_point = 100
    seq_len = best_individual[0]
    for j in range(data.size(0) - starting_point):
      i = (starting_point - seq_len) + j

      # Filter the necessary columns and create a sequence
      filter_cols = ["outTemp", "barometer", "dewpoint", "outHumidity", "windSpeed10"]
      col_idx = [filter_cols.index(extra_par) for extra_par in (["outTemp"] + best_individual[4])]
      x = data[i:i+seq_len, col_idx]
      y = data[i+seq_len,0].item()

      # Predict the temperature
      pred = model(x.unsqueeze(0))

      # Compute the MAE
      mae_lstm += abs(y - pred)
      mae_last += abs(y - x[-1,0].item())

      curve_y.append(y)
      curve_lstm.append(pred.item())
      curve_last.append(x[-1,0].item())

    mae_last /= (j+1)
    mae_lstm /= (j+1)

    if plot:
      plt.clf()
      plt.figure(figsize=(15,3))
      plt.plot(curve_y, label="Ground truth")
      plt.plot(curve_lstm, label="LSTM")
      plt.plot(curve_last, label="Baseline")
      plt.legend()
      plt.show()

    return mae_lstm, mae_last

  def evaluateMaeGp(self, func, dataset, seq_len, plot=False):
    """
    Calculate the Mean Absolute Error of the GP model 'func' passed as parameter
    by testing it on the provided dataset.

    Parameters
    ----------
    func : the compiled function returned by Genetic Programming
    dataset : Pandas dataframe containing all the data of size (N,F) where N is
      the number of samples and F is the number of weather features
    seq_len : the length of the sequence(lookback)
    plot : weather or not to plot the prediction compared to the baseline and
      the ground truth
    """
    curve_y = []        # Ground truth
    curve_gp = []       # Genetic Programming prediction curve
    curve_last = []     # Prediction using the last temperature recorded(baseline)

    data = torch.tensor(dataset.values, dtype=torch.float32, device=self.device)

    # Mean Absolute Error values
    mae_gp = 0
    mae_last = 0

    starting_point = 100
    for j in range(data.size(0) - starting_point):
      i = (starting_point - seq_len) + j

      # Create a sequence
      x = data[i:i+seq_len, 0]  # Get only the temperature
      y = data[i+seq_len,0].item()

      # Predict the temperature
      pred = func(*x.flatten().tolist())

      # Compute the MAE
      mae_gp += abs(y - pred)
      mae_last += abs(y - x[-1].item())

      curve_y.append(y)
      curve_gp.append(pred)
      curve_last.append(x[-1].item())

    mae_last /= (j+1)
    mae_gp /= (j+1)

    if plot:
      plt.clf()
      plt.figure(figsize=(15,3))
      plt.plot(curve_y, label="Ground truth")
      plt.plot(curve_gp, label="Genetic Programming")
      plt.plot(curve_last, label="Baseline")
      plt.legend()
      plt.show()

    return mae_gp, mae_last