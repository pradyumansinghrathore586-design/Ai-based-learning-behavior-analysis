import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features_to_scale = [
            'Study_Hours_Per_Week', 
            'Attendance_Rate', 
            'Assignments_Completed', 
            'Exam_Score', 
            'Forum_Participation', 
            'Library_Usage', 
            'Past_Failures'
        ]
        
    def load_data(self, filepath):
        """Load dataset from CSV."""
        try:
            df = pd.DataFrame(pd.read_csv(filepath))
            print(f"Data loaded successfully from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess(self, df):
        """
        Handle missing values (if any) and scale the features.
        We use StandardScaler because KMeans depends on distance metrics 
        (like Euclidean distance), so all features must be on the same scale.
        """
        # Create a copy to prevent changing original DataFrame
        df_processed = df.copy()
        
        # In this synthetic dataset, there are no missing values, but in a real-world scenario
        # you would handle them here:
        # df_processed.fillna(df_processed.median(), inplace=True)
        
        # Scaling numerical features
        print("Scaling features...")
        df_processed[self.features_to_scale] = self.scaler.fit_transform(df_processed[self.features_to_scale])
        
        return df_processed
        
    def get_scaler(self):
        """Return the fitted scaler for possible inverse transformation later."""
        return self.scaler
