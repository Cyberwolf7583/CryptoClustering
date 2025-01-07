
# CryptoClustering

## Source Code:
- The source code for the CryptoClustering project can be found in the [Crypto_Clustering.ipynb](Starter_Code/Crypto_Clustering.ipynb) Jupyter Notebook.

## Overview
    The CryptoClustering project aims to analyze and visualize the clustering of cryptocurrencies based on their price change percentages over various time frames. This analysis utilizes Principal Component Analysis (PCA) to reduce the dimensionality of the data and visualize it in a 2D scatter plot.

## Requirements
To run this project, you need the following Python libraries:
- `hvplot`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `numpy`

You can install the required libraries in a bash terminal using pip:

pip install hvplot matplotlib 
pandas scikit-learn numpy

## Usage
1. **Import Required Libraries**: The first step is to import the necessary libraries and dependencies.
   ```python
   import hvplot.pandas
   import matplotlib.pyplot as plt
   import pandas as pd
   from sklearn.cluster import KMeans
   from sklearn.decomposition import PCA
   from sklearn.preprocessing import StandardScaler
   import numpy as np
   ```

2. **Data Preparation**: Load your cryptocurrency data into a DataFrame and preprocess it as needed.

3. **PCA Analysis**: Perform PCA on the scaled data to reduce its dimensionality.
   ```python
   # Use the columns from the original scaled DataFrame as the index.
   scaled_data_df = scaled_data_df.set_index(market_data_df.columns.tolist())
   ```

4. **Visualization**: Create a scatter plot using `hvPlot` to visualize the PCA results.
   ```python
   df_pca_copy.hvplot.scatter(x='PCA1', y='PCA2', c='crypto_cluster', 
                               colormap='viridis', 
                               title='PCA Scatter Plot of Cryptocurrencies', 
                               xlabel='PCA1', ylabel='PCA2', 
                               size=100, alpha=0.7)
   ```

5. **Feature Influence Analysis**: Identify the strongest positive and negative influences for each principal component.
   ```python
   for i in range(loadings_df.shape[1]):
       pc_name = f'PC{i+1}'
       strongest_positive = loadings_df[pc_name].idxmax()
       strongest_negative = loadings_df[pc_name].idxmin()
       print(f"For {pc_name}:")
       print(f"  Strongest Positive Influence: {strongest_positive} ({loadings_df[pc_name].max()})")
       print(f"  Strongest Negative Influence: {strongest_negative} ({loadings_df[pc_name].min()})")
   ```

## Conclusion
This project provides insights into the clustering behavior of cryptocurrencies based on their price changes. By visualizing the data in a PCA scatter plot, users can better understand the relationships between different cryptocurrencies.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citations or Resources

1. **Pandas**: McKinney, Wes. "Data Analysis with Python and Pandas." O'Reilly Media, 2018.
   - [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

2. **Matplotlib**: Hunter, J. D. "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering, 2007.
   - [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

3. **Scikit-learn**: Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 2011.
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/)

4. **NumPy**: Harris, C. R., et al. "Array programming with NumPy." Nature, 2020.
   - [NumPy Documentation](https://numpy.org/doc/stable/)

5. **hvPlot**: Jupyter Development Team. "hvPlot: A High-Level Plotting API for the HoloViz Ecosystem." 
   - [hvPlot Documentation](https://hvplot.holoviz.org/)

6. **Principal Component Analysis (PCA)**: Jolliffe, I. T. "Principal Component Analysis." Springer Series in Statistics, 2002.
   - [PCA Overview](https://en.wikipedia.org/wiki/Principal_component_analysis)

7. **Cryptocurrency Data Source**: Specify the source of your cryptocurrency data, e.g., CoinGecko, CoinMarketCap, etc.
   - [CoinGecko API](https://www.coingecko.com/en/api)

8. **Visualization Techniques**: Any relevant articles or papers that discuss visualization techniques in data science or finance.