from matplotlib import pyplot as plt
import numpy as np

def plot_band_selection(features_selected:np.ndarray, samples:list, sample_labels=None, wavelength=None, n_features=-1):
    if sample_labels is not None:
        assert(len(samples) == len(sample_labels))
    else:
        sample_labels = [None] * len(samples)

    diff = np.diff(features_selected)
    features_ranges = np.split(features_selected, np.where(diff != 1)[0]+1)

    with plt.style.context('seaborn-colorblind'):
        fig = plt.figure()
        for idx, sample in enumerate(samples):
            plt.plot(sample, label=sample_labels[idx])
            plt.scatter(features_selected, sample[features_selected], alpha=.5)

        for r in features_ranges:
            plt.axvspan(r[0]-.25, r[-1]+.25, alpha=0.25)

        if wavelength is not None:
            ticks = np.linspace(0, len(wavelength)-1, 12, dtype=int)
            plt.xlabel('Wavelength (nm)', fontsize='x-large')
            plt.xticks(ticks, wavelength[ticks], rotation=45, fontsize='large')

        plt.ylabel('Reflectance', fontsize='x-large')
        plt.yticks(fontsize='large')
        
        if sample_labels[0] is not None:
            plt.legend()
            # plt.legend(loc='upper right')
    
        if n_features != -1:
            sparse_rate = (n_features - len(features_selected)) / n_features 
            title = 'Sparsity: {:.2f}'.format(sparse_rate)
            plt.title(title, fontsize='xx-large')
 
        plt.margins(x=0.01)

    return fig