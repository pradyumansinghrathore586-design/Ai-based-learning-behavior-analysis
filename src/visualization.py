import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class Visualizer:
    def __init__(self):
        # PCA reduces the 7 dimensions (features) into 2 dimensions
        # so we can easily plot it on a 2D graph.
        self.pca = PCA(n_components=2)
        self.pca_fitted = False
        
    def plot_clusters_2d(self, scaled_features, category_labels, sample_size=1500):
        """
        Use PCA to map high-dimensional data into 2D for visualization.
        For large learning datasets, sampling is used to speed up rendering
        and prevent visualizing an over-cluttered graph.
        """
        print("Applying PCA for 2D visualization...")
        
        if not self.pca_fitted:
            self.pca_components = self.pca.fit_transform(scaled_features)
            self.pca_fitted = True
            
        # Sample for plotting if the dataset is too big
        if len(scaled_features) > sample_size:
            idx = np.random.choice(len(scaled_features), sample_size, replace=False)
            x = self.pca_components[idx, 0]
            y = self.pca_components[idx, 1]
            cat_sample = [category_labels[i] for i in idx]
        else:
            x = self.pca_components[:, 0]
            y = self.pca_components[:, 1]
            cat_sample = category_labels
            
        df_plot = pd.DataFrame({'PCA1': x, 'PCA2': y, 'Category': cat_sample})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a beautiful scatterplot with color mapping
        palette = {'Top Performer': '#2ca02c', 'Average': '#1f77b4', 'At Risk': '#d62728'}
        
        sns.scatterplot(
            data=df_plot, x='PCA1', y='PCA2', hue='Category', 
            palette=palette, alpha=0.7, s=50, ax=ax, edgecolor='k'
        )
        
        ax.set_title('Student Behavioral Clusters (PCA 2D Projection)')
        ax.set_xlabel('First Principal Component (Variance)')
        ax.set_ylabel('Second Principal Component (Variance)')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        return fig

    def plot_feature_importance(self, df_original, category_labels):
        """
        Show how the different original features contribute to each cluster
        by plotting average values per category.
        """
        df_plot = df_original.copy()
        
        # Remove Student ID since it's irrelevant for averages
        if 'Student_ID' in df_plot.columns:
            df_plot = df_plot.drop('Student_ID', axis=1)
            
        df_plot['Category'] = category_labels
        
        # Calculate mean of all features grouped by category
        means = df_plot.groupby('Category').mean()
        
        # Normalize the means to plot them on the same comparative scale (0 to 1)
        # purely for visualization purposes
        normalized_means = (means - means.min()) / (means.max() - means.min())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        normalized_means.T.plot(kind='bar', ax=ax, colormap='Set1')
        ax.set_title('Relative Feature Average Profile by Student Cluster')
        ax.set_ylabel('Normalized Score (Relative to Max/Min)')
        ax.set_xlabel('Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
