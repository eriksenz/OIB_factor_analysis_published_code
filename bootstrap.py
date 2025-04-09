import pandas as pd

# define bootstrap function
def resample(df, n_group, n_bootstrap):
    # initialize a list to hold the bootstrapped dataframes
    bootstraps = []
    
    # determine the groups that have more than 50 samples
    group_sizes = df['Group'].value_counts()
    large_groups = group_sizes[group_sizes > n_group].index
    
    # separate the data into large groups and small groups
    large_groups_df = df[df['Group'].isin(large_groups)]
    small_groups_df = df[~df['Group'].isin(large_groups)]
    
    # perform bootstrapping
    for _ in range(n_bootstrap):
        # resample the large groups with replacement to have exactly n samples each
        bootstrapped = large_groups_df.groupby('Group', group_keys=False).apply(lambda x: x.sample(n=n_group, replace=True))
        
        # re-sort by the order of islands
        order = df['Island'].unique()
        bootstrapped['Island'] = pd.Categorical(bootstrapped['Island'], categories=order, ordered=True)
        bootstrapped = bootstrapped.sort_values('Island')
        
        # for each island, re-sort by index number
        bootstrapped = bootstrapped.groupby('Island', group_keys=False).apply(lambda x: x.sort_index())
        
        # combine bootstrapped data with non-bootstrapped data
        bootstrapped_df = pd.concat([bootstrapped, small_groups_df])
        
        # append each bootstrapped dataset
        bootstraps.append(bootstrapped_df)
    
    return bootstraps