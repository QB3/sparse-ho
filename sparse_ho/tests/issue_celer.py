from celer.datasets.simulated import make_correlated_data

y = make_correlated_data(10, 10, random_state=42)[1]
print("y.sum() = %5f " % y.sum())

y = make_correlated_data(10, 10, random_state=42)[1]
print("y.sum() = %5f " % y.sum())
