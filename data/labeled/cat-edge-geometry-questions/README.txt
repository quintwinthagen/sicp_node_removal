Original data obtained from the mathoverflow.net data dumps.

Here nodes are users and hyperedges correspond to users who answered a
particular type of question within a month. The type of question is determined
by a tag, and this dataset looks at tags involving geometry.

The file hyperedges.txt contains lists of users that answered a certain type of
question within a month. Each line lists the users that appeared
together in a hyperedge.

The file hyperedge-labels.txt lists the category type label (question tags)
corresponding to each line in the hyperedges.txt file.

The file hyperedge-label-identities.txt lists the names of tags, with order in
the list corresponding to the number label used in the file
hyperedge-labels.txt.

The file temporal-list.txt has the format "(user id) (category id) (timestamp)\n".
