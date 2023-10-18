import tkinter as tk
from tkinter import ttk, filedialog as fd, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create a GUI root
root = tk.Tk()

# Specify the title and dimensions to root
root.title('DNN manual tuner')
root.geometry('1800x1000')

#create labelframes
label_frame_input = ttk.Labelframe(root, text='Inputs', width=300, height=350)
label_frame_input.grid(row=0, column=0, sticky='n')
label_frame_dummies = ttk.Labelframe(root, text='Dummies', width=300, height=350)
label_frame_dummies.grid(row=0, column=1, sticky='n')
label_frame_feature_importance = ttk.Labelframe(root, text='Feature importance', width=300, height=350)
label_frame_feature_importance.grid(row=0, column=2, sticky='n')
label_frame_feature_selection = ttk.Labelframe(root, text='Feature selection', width=300, height=350)
label_frame_feature_selection.grid(row=0, column=3, sticky='n')
label_frame_feature_SMOTE_scaling = ttk.Labelframe(root, text='SMOTE & scaling', width=300, height=350)
label_frame_feature_SMOTE_scaling.grid(row=0, column=4, sticky='n')
label_frame_hyperparameters = ttk.Labelframe(root, text='Hyperparameters', width=300, height=350)
label_frame_hyperparameters.grid(row=0, column=5, sticky='n')
label_frame_results_train = ttk.Labelframe(root, text='Results train', width=1200, height=350)
label_frame_results_train.grid(row=0, column=6, sticky='n')
label_frame_results_test = ttk.Labelframe(root, text='Results test', width=1200, height=350)
label_frame_results_test.grid(row=1, column=6, sticky='n')

# Create an open file button
open_button = tk.Button(label_frame_input, text='Open database', command=lambda: OpenFile())
open_button.grid(row=0, column=0, sticky='nw')

#create column selection option
def create_label_frame_input():
    """
    Create and configure the input frame with a label and listbox for selecting the target/label column.

    Returns:
        tk.Frame: The configured input frame.
        tk.StringVar: Variable to store the selected column.
        tk.Listbox: Listbox widget for displaying available columns.
    """
    global column_selection
    global listbox_columns
    # Label for selecting target/label column
    tk.Label(label_frame_input, font="none 7 bold", text="Select target/label column:").grid(row=3, column=0, sticky='w') # place widget with empty text, will be filled later o
    # Variable to store the selected column
    column_selection = tk.StringVar()
    column_selection.set([])
    # Listbox for displaying available columns
    listbox_columns = tk.Listbox(label_frame_input, listvariable=column_selection)
    listbox_columns.grid(row=4, column=0, sticky='nw', rowspan = 10)

def OpenFile():
    """
    Open a file dialog to choose a CSV file and read the data.

    Returns:
    - str: Location of the selected database file.
    - list: Columns of the database.
    """
    global name
    global data
    global entry_observations
    name = fd.askopenfilename(initialdir="", filetypes=(("Text File", "*.csv"), ("All Files", "*.*")), title="Choose a file.")
    data = pd.read_csv(name, error_bad_lines=False)
    list(data.columns)
    column_selection.set(list(data.columns))
    tk.Button(label_frame_dummies, text='process', command=lambda: dummifying()).grid(row=0, column=0, sticky='w')
    total_rows = data.shape[0]
    tk.Label(label_frame_input, font="none 7 bold", text="Total observations: " + str(total_rows)).grid(row=14, column=0, sticky='w')  # place widget with empty text, will be filled late

    # Create an Entry widget
    tk.Label(label_frame_input, font="none 7 bold", text="Restict observations (random):").grid(row=15, column=0, sticky='w')
    entry_observations = tk.Entry(label_frame_input, text="")
    entry_observations.grid(row=16, column=0, padx=10, pady=10)

def clear_label_frame(name_label_frame):
    """
    Clear all label widgets in the specified label frame.

    Args:
    - name_label_frame (ttk.Labelframe): The label frame to be cleared.
    """
    for widget in name_label_frame.grid_slaves():
        if widget.winfo_class() == "Label":
            widget.destroy()

def dummifying():
    """
    Quick process of the database using a Chunk function and return some basic details of the selected column.

    Returns:
    - pd.DataFrame: Used database file.
    - str: Selected column.
    - float: Process time.
    - int: Observations in the database.
    - float: Average of the selected column.
    - float: Maximum value of the selected column.
    - float: Minimum value of the selected column.
    """
    clear_label_frame(label_frame_dummies)
    global data
    global X
    global y
    global X_encoded

    try:# checks if there was a column selected
        selection = listbox_columns.get(listbox_columns.curselection())
    except:
        tk.messagebox.showerror("warning", "Select column then Process database")
        return

    # check if the entry of the restrict observations is correct.
    value = int(entry_observations.get())
    if value == "":
        pass
    if isinstance(value, int) == True:
        data = data.sample(n=value, random_state=42).copy()
    else:
        tk.messagebox.showerror("Insert interger or leave empty")
        pass

    X = data.drop([selection], axis=1)
    y = data[selection]

    # Because we are trying to find the most significant correlations with another categorical variable ('Default'), it is very important to ensure we encode our categorical to ensure accurate feature selection.
    # One-hot encode all object (categorical) columns
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)
    tk.Label(label_frame_dummies, font="none 7 bold", text="Target column: " + str(selection)).grid(row=2, column=0, sticky='w') # place widget with empty text, will be filled later o
    tk.Label(label_frame_dummies, font="none 7 bold", text="Dummyfied columns:").grid(row=4, column=0, sticky='w')  # place widget with empty text, will be filled later

    row_number = 8
    for dummy in X.select_dtypes(include=['object']):
        tk.Label(label_frame_dummies, font="none 7", text=str(dummy)).grid(row=row_number, column=0,sticky='w') # place widget with empty text, will be filled later
        row_number = row_number + 1

    tk.Button(label_frame_feature_importance, text='process', command=lambda: feature_importance()).grid(row=0, column=0, sticky='w')

def feature_importance():
    """
    Calculate and display feature importances using RandomForestRegressor.
    """
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_regressor.fit(X_encoded, y)
    feature_importances = rf_regressor.feature_importances_
    global importance_df
    importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index()

    for index, row in importance_df.iterrows():
        feature = row["Feature"]
        importance = round(row["Importance"],3)
        tk.Label(label_frame_feature_importance, font="none 7", text=str(index) + " " + str(importance) + " " + feature).grid(row=index +1, column=0,sticky='w') # place widget with empty text, will be filled later
    # Create a StringVar to hold the default value
    tk.Button(label_frame_feature_selection, text='process', command=lambda: feature_selection()).grid(row=0, column=0, sticky='w')
    global default_value_selection
    default_value_selection = tk.StringVar()
    default_value_selection.set(0.01)  # Set the default value to "0.01"

    # Create an Entry widget and set its textvariable to the default value
    entry = tk.Entry(label_frame_feature_selection, textvariable=default_value_selection)
    entry.grid(row=1, column=0, sticky='w')


def feature_selection():
    """
    Perform feature selection based on the specified importance threshold.
    """

    value_selection = float(default_value_selection.get())
    clear_label_frame(label_frame_feature_selection)
    df_selection = importance_df[importance_df['Importance'] > value_selection]
    columns_to_filter= df_selection['Feature'].tolist()
    global X_encoded_filtered
    X_encoded_filtered = X_encoded.loc[:, columns_to_filter]
    tk.Label(label_frame_feature_selection, font="none 7 bold", text="Selected features:").grid(row=2, column=0, sticky='w')  # place widget with empty text, will be filled later o
    for index, row in df_selection.iterrows():
        feature = row["Feature"]
        importance = round(row["Importance"], 3)
        tk.Label(label_frame_feature_selection, font="none 7", text=str(index) + " " + str(importance) + " " + feature).grid(row=index + 3, column=0, sticky='w')  # place widget with empty text, will be filled later
    tk.Button(label_frame_feature_SMOTE_scaling, text='process', command=lambda: SMOTE_scaling()).grid(row=0, column=0, sticky='w')

    global checkbox_scaling_var
    checkbox_scaling_var = tk.BooleanVar(value=True)
    tk.Checkbutton(label_frame_feature_SMOTE_scaling, text="scaling", variable=checkbox_scaling_var).grid(row=2, column=0, sticky='w')
    global checkbox_SMOTE_var
    checkbox_SMOTE_var = tk.BooleanVar()
    tk.Checkbutton(label_frame_feature_SMOTE_scaling, text="SMOTE", variable=checkbox_SMOTE_var).grid(row=3, column=0, sticky='w')
    unique_features = y.unique()
    value_counts = y.value_counts()
    # Iterate over unique values and their counts using a for loop
    row = 4
    for value, count in value_counts.items():
        tk.Label(label_frame_feature_SMOTE_scaling, font="none 7", text=f"Feature value {value}: {count} occurrences").grid(row=row, column=0, sticky='w')
        row = row + 1


class Dropdown:
    """
    Create a dropdown menu in the specified label frame.

    Args:
    - label_frame (ttk.Labelframe): The label frame where the dropdown will be created.
    - options (tuple): Options for the dropdown.
    - default_value: Default value for the dropdown.
    - row (int): Row position in the label frame.
    - column (int): Column position in the label frame.
    - label (str, optional): Label for the dropdown.

    Attributes:
    - label_frame (ttk.Labelframe): The label frame where the dropdown is created.
    - options (tuple): Options for the dropdown.
    - default_value: Default value for the dropdown.
    - combo_var (tk.StringVar): StringVar to store the selected item.
    - combo (ttk.Combobox): Combobox widget.
    """
    def __init__(self, label_frame, options, default_value, row, column, label=None):
        self.label_frame = label_frame
        self.options = options
        self.default_value = default_value

        # Create a Label if specified
        if label:
            self.label = ttk.Label(label_frame, text=label)
            self.label.grid(row=row, column=column, padx=5, pady=5)

        # Create a StringVar to store the selected item
        self.combo_var = tk.StringVar()

        # Create a Combobox widget
        self.combo = ttk.Combobox(label_frame, textvariable=self.combo_var)
        self.combo['values'] = options
        self.combo.set(default_value)
        self.combo.grid(row=row, column=column + 1, padx=5, pady=5)

    def get_selected_value(self):
        return self.combo_var.get()

def SMOTE_scaling():
    """
    Apply SMOTE and scaling to the selected features.

    This function applies SMOTE (Synthetic Minority Over-sampling Technique) and scaling to the selected features.
    It resamples the dataset using SMOTE if the checkbox_SMOTE_var is selected, and scales the data if checkbox_scaling_var is selected.

    Global Variables:
    - X_scaled: Scaled feature matrix
    - y_resampled: Resampled target variable

    Parameters:
    None

    Returns:
    None
    """
    global X_scaled
    global y_resampled
    if checkbox_SMOTE_var.get():
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_encoded_filtered, y)
        y_resampled.value_counts(normalize=True)
    else:
        X_resampled, y_resampled = X_encoded_filtered, y
    if checkbox_scaling_var.get():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
    else:
        X_scaled = X_resampled

    tk.Button(label_frame_hyperparameters, text='process & show results', command=lambda: run_models()).grid(row=0, column=0, sticky='w')
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Deep Neural Network").grid(row=1, column=0, sticky='w')

    global layers_entry
    tk.Label(label_frame_hyperparameters, text="Number of Layers:").grid(row=10, column=0, sticky='w')
    layers_var = tk.StringVar(value="3")
    layers_entry = tk.Entry(label_frame_hyperparameters, textvariable=layers_var)
    layers_entry.grid(row=10, column=1)
    global units_entry

    tk.Label(label_frame_hyperparameters, text="Units per Layer:").grid(row=11, column=0, sticky='w')
    units_var = tk.StringVar(value="64,64,64")
    units_entry = tk.Entry(label_frame_hyperparameters, textvariable=units_var)
    units_entry.grid(row=11, column=1)
    global activation_entry
    tk.Label(label_frame_hyperparameters, text="Activations (relu,tanh,sigmoid):").grid(row=12, column=0, sticky='w')
    activation_var = tk.StringVar(value="relu,tanh,sigmoid")
    activation_entry = tk.Entry(label_frame_hyperparameters, textvariable=activation_var)
    activation_entry.grid(row=12, column=1)

    learning_rates = [0.00002, 0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1.0]
    epochs = list(range(1, 21))
    global dropdown_1
    global dropdown_2
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Learning rates").grid(row=13, column=0, sticky='w')
    dropdown_1 = Dropdown(label_frame_hyperparameters, learning_rates, learning_rates[5], row=14, column=0, label="")
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Epochs").grid(row=15, column=0, sticky='w')
    dropdown_2 = Dropdown(label_frame_hyperparameters, epochs, epochs[10], row=16, column=0, label="")


def create_neural_network(input_dim):
    """
    Create a deep neural network model.

    This function creates a deep neural network model based on the specified hyperparameters.

    Parameters:
    - input_dim (int): Dimensionality of the input features.

    Returns:
    - model (Sequential): Compiled Keras Sequential model.
    """
    num_layers = int(layers_entry.get())
    units_per_layer = list(map(int, units_entry.get().split(',')))
    activations = activation_entry.get().split(',')

    if len(units_per_layer) != num_layers or len(activations) != num_layers:
        raise ValueError("Number of layers, units_per_layer, and activations must have the same length.")

    model = Sequential()
    model.add(Dense(units_per_layer[0], input_dim=input_dim, activation=activations[0]))

    for i in range(1, num_layers):
        model.add(Dense(units_per_layer[i], activation=activations[i]))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


def run_models():
    """
    Run multiple classifiers and display the evaluation metrics.
    """
    classifiers = {'Neural Network': create_neural_network(X_scaled.shape[1])}

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

    # Initialize dictionaries to store evaluation metric results
    results_train = {
        'Classifier': [],
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'ROC AUC': []
    }

    results_test = {
        'Classifier': [],
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'ROC AUC': []
    }

    # Loop through classifiers and calculate various evaluation metrics
    for classifier_name, classifier in classifiers.items():

            # Assuming X_scaled and y_resampled are already defined for the DNN
        input_dim = X_scaled.shape[1]
        classifier = create_neural_network(input_dim)

            # Train the DNN model
        classifier.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1)
        y_train_pred = classifier.predict(X_train)
        y_train_pred = (y_train_pred > 0.5).astype(int)

        # Evaluate on the training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)

        # Make predictions on the test set
        y_test_pred = classifier.predict(X_test)
        y_test_pred = (y_test_pred > 0.5).astype(int)

        # Evaluate the model on the test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)

        # Store the results for both training and test sets
        results_train['Classifier'].append(classifier_name)
        results_train['Accuracy'].append(train_accuracy)
        results_train['F1 Score'].append(train_f1)
        results_train['Precision'].append(train_precision)
        results_train['Recall'].append(train_recall)
        results_train['ROC AUC'].append(train_roc_auc)

        results_test['Classifier'].append(classifier_name)
        results_test['Accuracy'].append(test_accuracy)
        results_test['F1 Score'].append(test_f1)
        results_test['Precision'].append(test_precision)
        results_test['Recall'].append(test_recall)
        results_test['ROC AUC'].append(test_roc_auc)

        tree2 = ttk.Treeview(label_frame_results_train, show="headings")

        # Include 'Classifier' in the list of columns
        columns = tuple(results_train.keys())
        tree2["columns"] = columns

        # Configure columns using grid
        for i, metric in enumerate(columns):
            tree2.column(metric, anchor="w", width=100)
            tree2.heading(metric, text=metric)
            tree2.grid(row=0, column=i, sticky="nsew")

        # Insert data into the Treeview
        for i in range(len(results_train['Classifier'])):
            classifier = results_train['Classifier'][i]
            row_data = [results_train[metric][i] for metric in columns]
            tree2.insert("", "end", values=tuple(row_data))

        # Configure treeview to expand with the window
        tree2.grid(row=1, column=0, sticky="nsew")

        #fill test:
        tree3 = ttk.Treeview(label_frame_results_test, show="headings")

        # Include 'Classifier' in the list of columns
        columns = tuple(results_test.keys())
        tree3["columns"] = columns

        # Configure columns using grid
        for i, metric in enumerate(columns):
            tree3.column(metric, anchor="w", width=100)
            tree3.heading(metric, text=metric)
            tree3.grid(row=0, column=i, sticky="nsew")

        # Insert data into the Treeview
        for i in range(len(results_test['Classifier'])):
            classifier = results_test['Classifier'][i]
            row_data = [results_test[metric][i] for metric in columns]
            tree3.insert("", "end", values=tuple(row_data))

        # Configure treeview to expand with the window
        tree3.grid(row=1, column=0, sticky="nsew")

create_label_frame_input()
root.mainloop()