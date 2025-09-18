import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class ChurnPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Churn Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize variables
        self.df = None
        self.model = None
        self.le_dict = {}
        self.feature_names = []

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weight
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        
    def create_data_tab(self):
        # Data Loading and Exploration Tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Exploration")
        
        # File loading section
        file_frame = ttk.LabelFrame(self.data_frame, text="Load Data", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_data).grid(row=0, column=0, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=5)
        
        # Data info section
        info_frame = ttk.LabelFrame(self.data_frame, text="Data Information", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(0, 5))
        
        self.info_text = tk.Text(info_frame, height=15, width=40)
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Plot frame
        self.plot_frame = ttk.LabelFrame(self.data_frame, text="Data Visualization", padding="10")
        self.plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for data tab
        self.data_frame.columnconfigure(0, weight=1)
        self.data_frame.columnconfigure(1, weight=2)
        self.data_frame.rowconfigure(1, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
    def create_training_tab(self):
        # Model Training Tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        
        # Training controls
        control_frame = ttk.LabelFrame(self.training_frame, text="Training Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(control_frame, text="Train Model", command=self.train_model).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Save Model", command=self.save_model).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Load Model", command=self.load_model).grid(row=0, column=2, padx=5)
        
        self.training_status = ttk.Label(control_frame, text="No model trained")
        self.training_status.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(self.training_frame, text="Training Results", padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=20, width=50)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Feature importance plot frame
        self.importance_frame = ttk.LabelFrame(self.training_frame, text="Feature Importance", padding="10")
        self.importance_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for training tab
        self.training_frame.columnconfigure(0, weight=1)
        self.training_frame.columnconfigure(1, weight=2)
        self.training_frame.rowconfigure(1, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
    def create_prediction_tab(self):
        # Prediction Tab
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="Make Predictions")
        
        # Sample data frame
        sample_frame = ttk.LabelFrame(self.prediction_frame, text="Quick Fill Options", padding="10")
        sample_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        ttk.Button(sample_frame, text="Fill Sample Data 1", command=lambda: self.fill_sample_data(1)).grid(row=0, column=0, padx=5)
        ttk.Button(sample_frame, text="Fill Sample Data 2", command=lambda: self.fill_sample_data(2)).grid(row=0, column=1, padx=5)
        ttk.Button(sample_frame, text="Fill Random Data", command=self.fill_random_data).grid(row=0, column=2, padx=5)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.prediction_frame, text="Customer Information", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=5, padx=5)
        
        # Create input fields (will be populated when data is loaded)
        self.input_widgets = {}
        self.create_input_fields(input_frame)
        
        # Prediction button and result
        button_frame = ttk.Frame(self.prediction_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(button_frame, text="Predict Churn", command=self.make_prediction).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear Fields", command=self.clear_fields).grid(row=0, column=1, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(self.prediction_frame, text="Prediction Result", padding="10")
        result_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        self.prediction_result = ttk.Label(result_frame, text="No prediction made", font=('Arial', 12, 'bold'))
        self.prediction_result.grid(row=0, column=0, pady=10)
        
        # Configure grid weights for prediction tab
        self.prediction_frame.columnconfigure(0, weight=1)
        
    def create_input_fields(self, parent):
        # All 19 fields including gender - will be updated when data is loaded
        self.input_fields = [
            ('gender', 'combobox', ['Male', 'Female']),
            ('SeniorCitizen', 'combobox', ['0', '1']),
            ('Partner', 'combobox', ['No', 'Yes']),
            ('Dependents', 'combobox', ['No', 'Yes']),
            ('tenure', 'entry', None),
            ('PhoneService', 'combobox', ['No', 'Yes']),
            ('MultipleLines', 'combobox', ['No', 'Yes', 'No phone service']),
            ('InternetService', 'combobox', ['DSL', 'Fiber optic', 'No']),
            ('OnlineSecurity', 'combobox', ['No', 'Yes', 'No internet service']),
            ('OnlineBackup', 'combobox', ['No', 'Yes', 'No internet service']),
            ('DeviceProtection', 'combobox', ['No', 'Yes', 'No internet service']),
            ('TechSupport', 'combobox', ['No', 'Yes', 'No internet service']),
            ('StreamingTV', 'combobox', ['No', 'Yes', 'No internet service']),
            ('StreamingMovies', 'combobox', ['No', 'Yes', 'No internet service']),
            ('Contract', 'combobox', ['Month-to-month', 'One year', 'Two year']),
            ('PaperlessBilling', 'combobox', ['No', 'Yes']),
            ('PaymentMethod', 'combobox', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
            ('MonthlyCharges', 'entry', None),
            ('TotalCharges', 'entry', None)
        ]
        
        row = 0
        col = 0
        for field_name, widget_type, values in self.input_fields:
            ttk.Label(parent, text=field_name + ":").grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
            
            if widget_type == 'combobox':
                widget = ttk.Combobox(parent, values=values, state='readonly')
            else:
                widget = ttk.Entry(parent)
            
            widget.grid(row=row, column=col*2+1, sticky=(tk.W, tk.E), padx=5, pady=2)
            self.input_widgets[field_name] = widget
            
            row += 1
            if row > 9:  # Create two columns with 10 fields each (19 total, so 10 + 9)
                row = 0
                col = 1
                
        # Configure column weights
        for i in range(4):
            parent.columnconfigure(i, weight=1)
    
    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.display_data_info()
                self.plot_churn_distribution()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def display_data_info(self):
        if self.df is None:
            return
            
        info_text = f"Dataset Shape: {self.df.shape}\n\n"
        info_text += "Column Information:\n"
        info_text += "-" * 30 + "\n"
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()
            info_text += f"{col}:\n"
            info_text += f"  Type: {dtype}\n"
            info_text += f"  Null values: {null_count}\n"
            info_text += f"  Unique values: {unique_count}\n\n"
        
        if 'Churn' in self.df.columns:
            churn_counts = self.df['Churn'].value_counts()
            info_text += "Churn Distribution:\n"
            info_text += "-" * 20 + "\n"
            for value, count in churn_counts.items():
                percentage = (count / len(self.df)) * 100
                info_text += f"{value}: {count} ({percentage:.1f}%)\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def plot_churn_distribution(self):
        if self.df is None or 'Churn' not in self.df.columns:
            return
            
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_counts = self.df['Churn'].value_counts()
        ax.bar(churn_counts.index, churn_counts.values, color=['lightblue', 'lightcoral'])
        ax.set_title('Churn Distribution')
        ax.set_xlabel('Churn')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(churn_counts.values):
            ax.text(i, v + 10, str(v), ha='center', va='bottom')
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.close(fig)  # Close to prevent memory leaks
    
    def preprocess_data(self):
        if self.df is None:
            return None, None
            
        df_processed = self.df.copy()
        
        # Drop customerID if it exists
        if 'customerID' in df_processed.columns:
            df_processed.drop('customerID', axis=1, inplace=True)
        
        # Convert TotalCharges to numeric
        if 'TotalCharges' in df_processed.columns:
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
        
        # Encode categorical variables
        self.le_dict = {}
        for col in df_processed.select_dtypes(include='object'):
            if col != 'Churn':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.le_dict[col] = le
        
        # Encode target variable
        if 'Churn' in df_processed.columns:
            le_target = LabelEncoder()
            df_processed['Churn'] = le_target.fit_transform(df_processed['Churn'])
            self.le_dict['Churn'] = le_target
        
        X = df_processed.drop('Churn', axis=1)
        y = df_processed['Churn']
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        try:
            X, y = self.preprocess_data()
            if X is None:
                messagebox.showerror("Error", "Failed to preprocess data!")
                return
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Train the model
            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            # Display results
            results_text = f"Training completed successfully!\n\n"
            results_text += f"Training samples: {len(X_train)}\n"
            results_text += f"Testing samples: {len(X_test)}\n\n"
            results_text += f"Accuracy: {accuracy * 100:.2f}%\n\n"
            results_text += "Confusion Matrix:\n"
            results_text += f"{cm}\n\n"
            results_text += "Classification Report:\n"
            results_text += f"{cr}"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results_text)
            
            self.training_status.config(text="Model trained successfully!")
            
            # Plot feature importance
            self.plot_feature_importance()
            
            messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy * 100:.2f}%")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def plot_feature_importance(self):
        if self.model is None:
            return
            
        # Clear previous plot
        for widget in self.importance_frame.winfo_children():
            widget.destroy()
            
        importances = self.model.feature_importances_
        features = self.feature_names
        
        # Sort features by importance
        feature_importance = list(zip(features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Show ALL features, not just top 10
        feature_names_sorted = [x[0] for x in feature_importance]
        importance_values = [x[1] for x in feature_importance]
        
        # Create a larger figure to accommodate all features
        fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.4)))
        bars = ax.barh(range(len(feature_names_sorted)), importance_values, color='skyblue')
        ax.set_yticks(range(len(feature_names_sorted)))
        ax.set_yticklabels(feature_names_sorted, fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title(f'All {len(features)} Feature Importances')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, importance_values)):
            ax.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.importance_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.close(fig)
    
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save! Please train a model first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Save model and encoders
                model_data = {
                    'model': self.model,
                    'encoders': self.le_dict,
                    'feature_names': self.feature_names
                }
                joblib.dump(model_data, file_path)
                messagebox.showinfo("Success", f"Model saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                model_data = joblib.load(file_path)
                self.model = model_data['model']
                self.le_dict = model_data['encoders']
                self.feature_names = model_data['feature_names']
                
                self.training_status.config(text="Model loaded successfully!")
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def make_prediction(self):
        if self.model is None:
            messagebox.showerror("Error", "No model available! Please train or load a model first.")
            return
            
        try:
            # Collect input values
            input_data = {}
            for field_name, widget in self.input_widgets.items():
                value = widget.get()
                if not value:
                    messagebox.showerror("Error", f"Please fill in {field_name}")
                    return
                input_data[field_name] = value
            
            # Create input array in the exact order of feature_names
            input_array = []
            for feature in self.feature_names:
                if feature in input_data:
                    value = input_data[feature]
                    
                    # Handle numeric fields
                    if feature in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
                        try:
                            input_array.append(float(value))
                        except ValueError:
                            messagebox.showerror("Error", f"{feature} must be a number")
                            return
                    else:
                        # Handle categorical fields
                        if feature in self.le_dict:
                            try:
                                encoded_value = self.le_dict[feature].transform([value])[0]
                                input_array.append(encoded_value)
                            except ValueError:
                                messagebox.showerror("Error", f"Invalid value for {feature}: {value}")
                                return
                        else:
                            input_array.append(value)
                else:
                    messagebox.showerror("Error", f"Missing feature: {feature}")
                    return
            
            # Debug information
            print(f"Expected features: {len(self.feature_names)}")
            print(f"Input array length: {len(input_array)}")
            print(f"Feature names: {self.feature_names}")
            print(f"Input array: {input_array}")
            
            # Make prediction
            prediction = self.model.predict([input_array])[0]
            prediction_proba = self.model.predict_proba([input_array])[0]
            
            # Decode prediction
            if 'Churn' in self.le_dict:
                prediction_label = self.le_dict['Churn'].inverse_transform([prediction])[0]
            else:
                prediction_label = "Yes" if prediction == 1 else "No"
            
            # Display result
            confidence = max(prediction_proba) * 100
            result_text = f"Prediction: {prediction_label}\nConfidence: {confidence:.1f}%"
            
            if prediction_label == "Yes" or prediction == 1:
                self.prediction_result.config(text=result_text, foreground="red")
            else:
                self.prediction_result.config(text=result_text, foreground="green")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
            print(f"Error details: {e}")  # Debug print
    
    def fill_sample_data(self, sample_num):
        """Fill the form with predefined sample data"""
        if sample_num == 1:
            # High churn risk customer
            sample_data = {
                'gender': 'Male',
                'SeniorCitizen': '1',
                'Partner': 'No',
                'Dependents': 'No', 
                'tenure': '1',
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': '85.0',
                'TotalCharges': '85.0'
            }
        else:  # sample_num == 2
            # Low churn risk customer
            sample_data = {
                'gender': 'Female',
                'SeniorCitizen': '0',
                'Partner': 'Yes',
                'Dependents': 'Yes',
                'tenure': '60',
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'TechSupport': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Two year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Credit card (automatic)',
                'MonthlyCharges': '55.0',
                'TotalCharges': '3300.0'
            }
        
        # Fill the widgets
        for field_name, value in sample_data.items():
            if field_name in self.input_widgets:
                widget = self.input_widgets[field_name]
                if isinstance(widget, ttk.Combobox):
                    widget.set(value)
                else:
                    widget.delete(0, tk.END)
                    widget.insert(0, value)
    
    def fill_random_data(self):
        """Fill the form with random valid data"""
        import random
        
        random_data = {
            'gender': random.choice(['Male', 'Female']),
            'SeniorCitizen': random.choice(['0', '1']),
            'Partner': random.choice(['No', 'Yes']),
            'Dependents': random.choice(['No', 'Yes']),
            'tenure': str(random.randint(1, 72)),
            'PhoneService': random.choice(['No', 'Yes']),
            'MultipleLines': random.choice(['No', 'Yes', 'No phone service']),
            'InternetService': random.choice(['DSL', 'Fiber optic', 'No']),
            'OnlineSecurity': random.choice(['No', 'Yes', 'No internet service']),
            'OnlineBackup': random.choice(['No', 'Yes', 'No internet service']),
            'DeviceProtection': random.choice(['No', 'Yes', 'No internet service']),
            'TechSupport': random.choice(['No', 'Yes', 'No internet service']),
            'StreamingTV': random.choice(['No', 'Yes', 'No internet service']),
            'StreamingMovies': random.choice(['No', 'Yes', 'No internet service']),
            'Contract': random.choice(['Month-to-month', 'One year', 'Two year']),
            'PaperlessBilling': random.choice(['No', 'Yes']),
            'PaymentMethod': random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
            'MonthlyCharges': str(round(random.uniform(18.0, 120.0), 2)),
            'TotalCharges': str(round(random.uniform(18.0, 8500.0), 2))
        }
        
        # Fill the widgets
        for field_name, value in random_data.items():
            if field_name in self.input_widgets:
                widget = self.input_widgets[field_name]
                if isinstance(widget, ttk.Combobox):
                    widget.set(value)
                else:
                    widget.delete(0, tk.END)
                    widget.insert(0, value)
    
    def clear_fields(self):
        for widget in self.input_widgets.values():
            if isinstance(widget, ttk.Combobox):
                widget.set('')
            else:
                widget.delete(0, tk.END)
        
        self.prediction_result.config(text="No prediction made", foreground="black")

def main():
    root = tk.Tk()
    app = ChurnPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()