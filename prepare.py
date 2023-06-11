import acquire as a
from sklearn.model_selection import train_test_split

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

