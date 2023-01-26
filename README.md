# Fitting_PCA_model

The code uses the Fisher Iris dataset, which contains measurements of the sepal and petal length and width for 150 iris flowers of three different species:
setosa, versicolor, and virginica. The code is applying principal component analysis (PCA) to reduce the dimensionality of the data and then visualizing the results
by plotting the first two principal components. It then uses the my_fitpca function to fit a PCA model to the data and the plot_ellipse function to plot the boundary
of the ellipse. The my_predictpca function then predicts the class of each data point.

The code first loads the Fisher Iris dataset, then uses the pca function to perform PCA on the measurements of the flowers. After that, it plots the first two
principal components. Then, it creates a function called my_fitpca that takes in the PCA data and the classes of the data as input. This function calculates the
eigenvectors, eigenvalues,and mean of the PCA data for each class.
Next, it calls the plot_ellipse function, which plots the ellipse boundary of each class using the eigenvectors and eigenvalues of the data.
Then, it defines the MahalanobisDistance function, which calculates the distance between a data point and the mean of the data in the feature space defined by the
eigenvectors.
Finally, it calls the my_predictpca function which takes in the model and data as input. This function uses the MahalanobisDistance function to predict the class
of each data point and return the class and score of each data point.
