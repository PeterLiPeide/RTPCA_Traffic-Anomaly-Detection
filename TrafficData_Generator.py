"""
This module load 2018 NYC taxi traffic data and provide methods for generating synthethic traffic basing on the 
real traffic data
"""

#%%
import os
import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt 


def Tensor_reshape(Tx):
    """
    reshape 4D traffic tensor into 3d.
    The dimension of days is squeezed out and concatenated on the hour dimension
    """
    if len(Tx.shape) == 3:
        print("It is already a 3-mode tensor!")
        return
    newTx = Tx[:, 0, :, :]
    for i in range(1, Tx.shape[1]):
        newTx = np.concatenate((newTx, Tx[:, i, :, :]), axis=0)
    return newTx

def Traffic_plot(Tx, week_index = None, loc_index = None):
    """
    Making plot for the 3D traffic data
    require Tx to be 3D tensor
    """
    plt.figure(figsize=(16, 9))
    for i in range(Tx.shape[1]):
        for j in range(Tx.shape[-1]):
            plt.plot(Tx[:, i, j], label = " ".join(['Week', str(i), 'Location', str(j)]))
    plt.legend()
    plt.title('Traffic Volume Plot')
    plt.xlabel('Hours in a Week')
    plt.ylabel('Traffic Volume')
    plt.show()
    return

def TrafficData_Simulate(Tx, threshold, mode = 1, const = None):
    """
    Generate synthetic traffic data and inject anomalies at specific positions
    Tx: True traffic tensor (NYC taxi data)
    p: percentage of anomalies to generate
    const: a constant for anomaly injection if desired

    return:
    synthetic data, anomaly_position
    """
    newshape = [i for i in Tx.shape]
    newshape[mode] = 1
    newshape = tuple(newshape)
    ave_Tx = np.mean(Tx, axis=mode).reshape(newshape)
    syn_Tx = np.concatenate([ave_Tx] * Tx.shape[mode], axis= mode)
    new_Tx = syn_Tx + syn_Tx * np.random.normal(0, 0.3, size=Tx.shape)
    total_element = np.prod(Tx.shape)
    anomaly_pos = np.unravel_index(np.random.choice(total_element, threshold, False), Tx.shape)
    multiplier_tensor = np.zeros(Tx.shape)
    multiplier_tensor[anomaly_pos] = 1
    if not const:
        new_Tx += 2.5 * multiplier_tensor * syn_Tx # Anomalies without constants
    else:
        new_Tx += 2.5 * multiplier_tensor * syn_Tx + const * multiplier_tensor   # Anomalies with constants

    # anomaly = set()
    # for i, j, k in zip(anomaly_pos[0], anomaly_pos[1], anomaly_pos[-1]):
    #     anomaly.add(tuple([i, j, k]))
    """
    multiplier_tensor is a tensor whose elements are either 1 or 0. 1 indicates anomalies and 0 indicates no anomaly. Thus return multiplier_tensor is enough for performance evaluation
    """
    return new_Tx, multiplier_tensor



# # %%
# def main():
#     nyc_traffic = loadmat('D:\\OneDrive - Michigan State University\\Documents\\Anomaly Detection\\RobustTensor\\Data\\NYC_Taxi.mat')

#     print(type(nyc_traffic))
#     rawdata = nyc_traffic['Y']
#     print(rawdata.shape)
#     tmp = Tensor_reshape(rawdata)
#     print(tmp.shape)
#     newTx, anomaly_loc = TrafficData_Simulate(tmp, 0.1)
#     print(newTx.shape)
#     print(anomaly_loc)
#     return

# if __name__ == "__main__":
#     main()