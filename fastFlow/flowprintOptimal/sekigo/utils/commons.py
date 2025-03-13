import numpy as np


def downSampleArray(array : np.ndarray,factor):
    """
    Array shape is (......,time_steps) needs to be atleast 2D
    Changes array inplace
    """
    assert array.shape[1] >= factor
    length = array.shape[1] - array.shape[1]%factor
    down_sampled_array = array[:,:length].reshape(array.shape[0],-1,factor).sum(axis = -1)
    return down_sampled_array



def augmentData(original_data,fraction_range = [.25,.4]):
    """
    original data is an numpy array of any dim
    this function copies and dosent just change the original array
    """
    flat_vector = original_data.reshape(-1).copy()
    indices = np.arange(flat_vector.shape[0])
    augment_size = int((np.random.uniform(fraction_range[0],fraction_range[1],1).item())*flat_vector.shape[0])
    if augment_size == 0:
        return original_data
    chosen_indices = np.random.choice(indices,size= augment_size,replace= False)
    flat_vector[chosen_indices] = np.random.uniform(low= 0, high= 1,size= chosen_indices.shape[0])
    augmented_data = flat_vector.reshape(original_data.shape)
    return augmented_data

def shuffleAugment(original_data):
    pass


def perturbData(data,alpha):
    #noise = np.random.randn(*data.shape)*alpha
    noise = np.random.uniform(low= data.min(),high= data.max(),size= data.shape)
    p_data = data*(1 - alpha) + noise*alpha
    #p_data = np.clip(p_data,a_min= -1, a_max= 1)
    return p_data




