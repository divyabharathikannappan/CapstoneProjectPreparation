from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the dataset from the given file path.
    Parameters:
    - file_path: The path to the CSV file containing the data.
    
    Returns:
    - data: Loaded DataFrame containing the dataset.
    """
    data = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
    return data  # Return the loaded DataFrame

def preprocess_data(data):
    """
    Preprocess the data by handling missing values, encoding categorical variables, 
    and normalizing features.

    Parameters:
    - data: DataFrame containing the raw dataset.
    
    Returns:
    - data: Processed DataFrame ready for training.
    """
    #*******************************************************************************
    # Fill missing values based on specific rules and domain knowledge.

    # Fill marital status based on age groups
    data.loc[(data['marital'].isnull()) & (data['age'] < 30), 'marital'] = 'single'  # Assign 'single' to young people with missing marital status
    data.loc[(data['marital'].isnull()) & (data['age'] >= 30), 'marital'] = 'married'  # Assign 'married' to older people with missing marital status
    data['marital'].fillna('unknown', inplace=True)  # Fill any remaining missing marital statuses with 'unknown'

    # Filling jobs with 'unknown' if missing
    data['job'].fillna('unknown', inplace=True)  # Fill missing job values with 'unknown'

    # Assuming people in management have a university degree
    data.loc[(data['education'].isnull()) & (data['job'] == 'management'), 'education'] = 'university.degree'  # Fill missing education for management role with university degree

    # Fill remaining education values by the mode within each job group
    data['education'] = data.groupby('job')['education'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown'))  # Fill missing education by the most frequent value in the job group

    # Fill defaults, housing, and loan columns with default values
    data['education'].fillna('unknown', inplace=True)  # Fill any remaining missing education values with 'unknown'
    data['default'].fillna('no', inplace=True)  # Fill missing default status with 'no'
    data['housing'].fillna('yes', inplace=True)  # Fill missing housing status with 'yes'
    data['loan'].fillna('yes', inplace=True)  # Fill missing loan status with 'yes'

    #****************************************************************************
    # Drop unnecessary columns to simplify the dataset.

    columns_to_drop = [
        'pdays',  # Number of days since the client was last contacted (not useful for prediction)
        'previous',  # Number of contacts performed before this campaign (not useful for prediction)
        'contact',  # Type of communication used (not useful for prediction)
        'day_of_week',  # Day of the week the campaign was conducted (does not significantly affect outcome)
        'duration'  # Duration of the last contact (depends on outcome, should not be included in training)
    ]
    data.drop(columns=columns_to_drop, inplace=True)  # Drop the unnecessary columns from the dataset

    #******************************************************************************

    # Convert categorical variables to dummy/one-hot encoded variables.
    # One-hot encoding creates binary columns for each category in the categorical features.

    data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'poutcome'], drop_first=True)  # Apply one-hot encoding for categorical variables and drop the first column for each feature to avoid multicollinearity

    #********************************************************************************
    # Normalize features using StandardScaler (standardization to have mean=0, variance=1)

    scaler = StandardScaler()  # Create an instance of StandardScaler
    x = scaler.fit_transform(data.drop(columns=['y']))  # Normalize all features except the target column 'y'
    y = LabelEncoder().fit_transform(data['y'])  # Encode the target labels into numerical values (0 or 1 for binary classification)

    # The features (x) and labels (y) are now ready for training

    return data  # Return the processed DataFrame
