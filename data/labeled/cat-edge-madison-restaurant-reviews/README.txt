Original data obtained from https://www.kaggle.com/yelp-dataset/yelp-dataset.

Here nodes are Yelp reviewers, hyperedges corresponds to reviewers who reviewed
a particular type of establishment within a month. The establishment types are
different types of restaurants in Madison, WI.

The file hyperedges.txt contains lists of reviewers that reviewed a certain type
of establishment within a month. Each line lists the reviewer numbers that
appeared together in a hyperedge.

The file hyperedge-labels.txt lists the category type label (establishment type)
corresponding to each line in the hyperedges.txt file.

The file hyperedge-label-identities.txt lists the names of categories, with
order in the list corresponding to the number label used in the file
hyperedge-labels.txt.

The file temporal-list.txt has format "(reviewer id) (category id) (timestamp)\n".
