import pandas as pd
from sklearn.cluster import MiniBatchKMeans

class BehaviorClusteringModel:
    def __init__(self, n_clusters=3, batch_size=1024, random_state=42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        # MiniBatchKMeans is perfect for large datasets (10,000+ rows)
        # It processes the data in small chunks (batches), reducing memory usage
        # and speeding up training while giving similar results to K-Means.
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init='auto'
        )
        self.cluster_mapping = {}

    def fit_predict(self, data):
        """Fit model on scaled data and return the cluster labels."""
        print(f"Training MiniBatchKMeans with {self.n_clusters} clusters...")
        labels = self.model.fit_predict(data)
        return labels

    def interpret_clusters(self, df, labels):
        """
        Assign meaningful business logic to the clusters based on their average statistics.
        By analyzing the average exam score for each cluster, we can infer
        which cluster represents the top performers, average, and at-risk students.
        """
        df_temp = df.copy()
        df_temp['Cluster'] = labels
        
        # Calculate the mean of Exam_Score for each cluster
        cluster_means = df_temp.groupby('Cluster')['Exam_Score'].mean()
        
        # Sort the clusters by Exam Score descending
        # Index 0 will be the highest score group, Index 2 will be the lowest score group
        sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
        
        if len(sorted_clusters) == 3:
            self.cluster_mapping[sorted_clusters[0]] = "Top Performer"
            self.cluster_mapping[sorted_clusters[1]] = "Average"
            self.cluster_mapping[sorted_clusters[2]] = "At Risk"
            
        print(f"Cluster Interpretation established: {self.cluster_mapping}")
        
        # Map raw numerical labels (0, 1, 2) to their meaningful string labels
        predicted_categories = [self.cluster_mapping.get(label, f"Cluster {label}") for label in labels]
        return predicted_categories

    def get_recommendation(self, category):
        """Basic recommendation system based on the student's cluster."""
        if category == "Top Performer":
            return "Provide advanced materials, leadership opportunities, and challenging projects."
        elif category == "Average":
            return "Encourage more library usage and forum participation. Offer targeted practice."
        elif category == "At Risk":
            return "Immediate intervention required. Schedule 1-on-1 tutoring and track attendance closely."
        return "No recommendation available."
