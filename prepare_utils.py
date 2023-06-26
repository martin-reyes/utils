from sklearn.model_selection import train_test_split

def identify_cols_with_white_space(df):
    '''
    Takes in a DataFrame
    Prints columns with any values that are whitespace
    Returns columns in a list
    '''
    
    cols_w_white_space = []
    
    for col in df.columns:
        # check string/object columns
        if df[col].dtype == 'O':
            # check for any values in the column that are empty or whitespace
            is_whitespace = df[col].str.isspace()
            has_whitespace = is_whitespace.any()
            if has_whitespace:
                print(f'{col} has whitespace')
                cols_w_white_space.append(col)
    return cols_w_white_space

# # replace empty space with np.nan and convert column to float
# df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).astype(float)
# # impute median
# df[col] = df[col].fillna( df[col].median())

def split_data(df, test_size=.2, validate_size=.2, stratify_col=None, random_state=None):
    '''
    take in a DataFrame and return train, validate, and test DataFrames;
    return train, validate, test DataFrames.
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, test_size=test_size, random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, test_size=validate_size/(1-test_size),
                                                                           random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df, test_size=test_size,
                                                random_state=random_state, stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, test_size=validate_size/(1-test_size),
                                           random_state=random_state, stratify=train_validate[stratify_col])       
    return train, validate, test

