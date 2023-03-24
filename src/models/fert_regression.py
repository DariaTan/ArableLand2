import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def model(Fert, Total_area,
          prefixes, country_names,
          years, years_future_fert):
    """
    Creates a model for each country and year

    Parameters
    ----------
    Fert : dict
        Dictionary containing the fertilizers for each country and year
    Total_area : dict
        Dictionary containing the total area for each country and year
    prefixes : list
        List of country prefixes
    country_names : list
        List of country names
    years : list
        List of years
    years_future_fert : list
        List of years in the future

    Returns
    -------
    Fert_future : dict
        Dictionary containing the fertilizers forecast for each country and year

    """
    model = LinearRegression()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    Fert_future = {keys: [] for keys in prefixes}

    for prefix, name in zip(prefixes, country_names):
        values_N, values_P205, values_K2O = [], [], []
        c = np.random.rand(3,)

        for year in years:
            values_N.append(float(Fert[prefix][year][0])/Total_area[prefix][year]['total'])
            values_P205.append(float(Fert[prefix][year][1])/Total_area[prefix][year]['total'])
            values_K2O.append(float(Fert[prefix][year][2])/Total_area[prefix][year]['total'])
        model.fit(np.array(years).reshape(-1, 1), values_N)
        pred = model.predict(np.array(years_future_fert).reshape(-1, 1))
        ax[0].plot(years, values_N, label=name, color=c)
        ax[0].scatter(years_future_fert, pred, marker='.', color=c)
        Fert_future[prefix].append(pred[-1])

        model.fit(np.array(years).reshape(-1, 1), values_P205)
        pred = model.predict(np.array(years_future_fert).reshape(-1, 1))
        ax[1].plot(years, values_P205, label=name, color=c)
        ax[1].scatter(years_future_fert, pred, marker='.', color=c)
        Fert_future[prefix].append(pred[-1])

        model.fit(np.array(years).reshape(-1, 1), values_K2O)
        pred = model.predict(np.array(years_future_fert).reshape(-1, 1))
        ax[2].plot(years, values_K2O, label=name, color=c)
        ax[2].scatter(years_future_fert, pred, marker='.', color=c)
        Fert_future[prefix].append(pred[-1])

    ax[0].set_title('N')
    ax[1].set_title('P2O5')
    ax[2].set_title('K2O')
    plt.legend(bbox_to_anchor=(0.4, 0.75), ncol=2, fontsize=14)
    plt.show()

    return Fert_future
