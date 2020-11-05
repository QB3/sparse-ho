"""File to download and load real data from libsvm, using libsvmdata.
"""

from libsvmdata import fetch_libsvm
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# def _get_20newsgroup():
#     print("Loading data...")
#     newsgroups = fetch_20newsgroups()
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(newsgroups.data)
#     y = newsgroups.target

#     X = X.tocsc()
#     # NNZ = np.diff(X.indptr)  # number of non zero elements per feature
#     # # keep only features with >=3 non zero values
#     # X = X[:, NNZ >= 3]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.5, random_state=42)

#     return X_train, X_val, X_test, y_train, y_val, y_test


# def _get_finance():
#     X, y = fetch_libsvm("finance")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.5, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test


# def _get_rcv1():
#     X, y = fetch_libsvm("rcv1_train")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.5, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test


# def _get_leukemia():
#     X, y = fetch_libsvm("leu")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.5, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test


# def _get_real_sim():
#     X, y = fetch_libsvm("real-sim")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.5, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test


# def _get_news20():
#     X, y = fetch_libsvm("news20")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.5, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test


# def get_data_old(dataset_name, csr=False):
#     if dataset_name == "finance":
#         out = _get_finance()
#     elif dataset_name == "20newsgroups":
#         out = _get_20newsgroup()
#     elif dataset_name == "rcv1":
#         out = _get_rcv1()
#     elif dataset_name == "leukemia":
#         out = _get_leukemia()
#     elif dataset_name == "real-sim":
#         out = _get_real_sim()
#     elif dataset_name == "20news":
#         out = _get_news20()
#     else:
#         raise ValueError("dataset_name %s does not exist" % dataset_name)

#     X_train, X_val, X_test, y_train, y_val, y_test = out

#     if csr:
#         X_train = X_train.tocsr()
#         X_val = X_val.tocsr()
#         X_test = X_test.tocsr()
#     else:
#         X_train = X_train.tocsc()
#         X_val = X_val.tocsc()
#         X_test = X_test.tocsc()

#     print("Finished loading data: %s ..." % dataset_name)

#     return X_train, X_val, X_test, y_train, y_val, y_test


def get_data(dataset_name, csr=False):
    X, y = fetch_libsvm(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    # be careful train_test_split returns crs matrices

    if csr:
        X_train = X_train.tocsr()
        X_val = X_val.tocsr()
        X_test = X_test.tocsr()
    else:
        X_train = X_train.tocsc()
        X_val = X_val.tocsc()
        X_test = X_test.tocsc()

    print("Finished loading data: %s ..." % dataset_name)

    return X_train, X_val, X_test, y_train, y_val, y_test
