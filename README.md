Malaria Detection Project using LeNet Model
===========================================

This notebook implements a Malaria Detection Project using the LeNet model architecture, TensorFlow Datasets (TFDS) malaria dataset, and various advanced techniques for model training and evaluation. The project includes data preprocessing, data augmentation, Mix-Up data augmentation, resizing, and early stopping callbacks. It also integrates TensorBoard for visualizing the training process and WandB.ai for experiment tracking.

Project Structure
-----------------

-   `images/`: Directory for images, including screenshots and visualizations.
-   `logs/`: Directory to store TensorBoard logs.

Getting Started
---------------

1.  Install required dependencies:

    bashCopy code

    `pip install tensorflow tensorflow-datasets matplotlib scikit-learn opencv-python seaborn tensorflow-probability`

2.  Clone the repository:

    bashCopy code

    `git clone https://github.com/zain-0/Malaria-Diagnosis-using-LeNet-and-WandB.AI.git
    cd malaria-diagnosis-using-LeNet-and-WandB.AI `

3.  Download the TFDS malaria dataset and place it in the `data/` directory.

4.  Run the notebook `malaria_detection.ipynb` in your preferred environment.

Project Notebooks
-----------------

-   `malaria_diagnosis_cnn.ipynb`: Main notebook containing the implementation of the Malaria Detection Project.

Data Preprocessing and Augmentation
-----------------------------------

-   The dataset is loaded using TFDS, split into training, validation, and test sets.
-   Data augmentation techniques such as random rotation, random flip, and random contrast are applied to the training dataset.
-   Mix-Up data augmentation is implemented to enhance model generalization.

Model Architecture
------------------

-   LeNet model architecture is used for malaria detection.
-   The model includes Conv2D layers, BatchNormalization, MaxPool2D, Dropout, and Dense layers.
-   Various hyperparameters such as learning rate, dropout rate, regularization rate, etc., can be tuned.

Callbacks
---------

-   Early stopping is implemented to prevent overfitting.
-   CSV Logger.
-   TensorBoard integration is utilized for visualizing training progress and logs.
-   WandB.ai is integrated for experiment tracking and visualization.

Hyperparameter Tuning
---------------------

-   Hyperparameter tuning is performed using TensorBoard and WandB.ai.
-   Multiple hyperparameters such as learning rate, batch size, and architecture parameters are explored.

Model Evaluation and Visualization
----------------------------------

-   Model evaluation metrics, including accuracy, precision, recall, and AUC, are tracked.
-   Confusion matrix and ROC curve are visualized to assess model performance.

Results
-------

-   The training progress, hyperparameter tuning results, and visualization images are stored in the `logs/` directory.
-   Model weights can be saved in the `model_weights.h5` file.
-   Visualizations and screenshots can be found in the `images/` directory.

Feel free to explore and modify the notebook based on your needs. If you encounter any issues or have suggestions, please open an issue on GitHub. Happy coding!
