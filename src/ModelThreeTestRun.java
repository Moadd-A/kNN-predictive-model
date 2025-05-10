import java.io.IOException;

public class ModelThreeTestRun {
    static void Assert(boolean res) {
        if (!res) {
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    // Calculate mean for a specific feature for a specific class
    static double calculateMean(double[][] data, int[] labels, int classLabel, int featureIndex) {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            if (labels[i] == classLabel) {
                sum += data[i][featureIndex];
                count++;
            }
        }
        return count > 0 ? sum / count : 0;
    }

    // Calculate standard deviation for a specific feature for a specific class
    static double calculateStdDev(double[][] data, int[] labels, int classLabel, int featureIndex, double mean) {
        double sumSquaredDiff = 0;
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            if (labels[i] == classLabel) {
                sumSquaredDiff += Math.pow(data[i][featureIndex] - mean, 2);
                count++;
            }
        }
        // Add small constant to prevent division by zero
        return count > 1 ? Math.sqrt(sumSquaredDiff / (count - 1)) + 0.0001 : 1;
    }

    // Calculate Gaussian likelihood
    static double gaussianLikelihood(double x, double mean, double stdDev) {
        // Handle edge cases
        if (stdDev == 0) return x == mean ? 1.0 : 0.0;

        double exponent = -Math.pow(x - mean, 2) / (2 * Math.pow(stdDev, 2));
        // Use log-space calculations to prevent underflow
        return exponent - Math.log(stdDev * Math.sqrt(2 * Math.PI));
    }

    // Main classification method using Gaussian Naive Bayes
    static int gaussianNaiveBayesClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature) {
        // Prior probabilities
        double logPriorLike = Math.log(0.61);
        double logPriorDislike = Math.log(0.39);

        // Initialize log probabilities with priors
        double logProbLike = logPriorLike;
        double logProbDislike = logPriorDislike;

        // For each feature
        for (int feature = 8; feature < 14; feature++) {  // Only use numerical features
            // Calculate means and standard deviations for both classes
            double meanLike = calculateMean(trainingData, trainingLabels, 1, feature);
            double stdDevLike = calculateStdDev(trainingData, trainingLabels, 1, feature, meanLike);

            double meanDislike = calculateMean(trainingData, trainingLabels, 0, feature);
            double stdDevDislike = calculateStdDev(trainingData, trainingLabels, 0, feature, meanDislike);

            // Add log likelihoods (already in log space from gaussianLikelihood)
            logProbLike += gaussianLikelihood(testFeature[feature], meanLike, stdDevLike);
            logProbDislike += gaussianLikelihood(testFeature[feature], meanDislike, stdDevDislike);
        }

        // Return 1 for "like" if its log probability is higher, 0 for "dislike" otherwise
        return logProbLike > logProbDislike ? 1 : 0;
    }

    public static void main(String[] args) {
        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];

        try {
            ModelOne.loadData("training-set.csv", trainingData, trainingLabels);
            ModelOne.loadData("testing-set.csv", testingData, testingLabels);
        } catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        // Print some debug information
        System.out.println("Training Data Analysis:");
        for (int feature = 8; feature < 14; feature++) {
            double meanLike = calculateMean(trainingData, trainingLabels, 1, feature);
            double stdDevLike = calculateStdDev(trainingData, trainingLabels, 1, feature, meanLike);
            double meanDislike = calculateMean(trainingData, trainingLabels, 0, feature);
            double stdDevDislike = calculateStdDev(trainingData, trainingLabels, 0, feature, meanDislike);

            System.out.printf("Feature %d:\n", feature);
            System.out.printf("  Like - Mean: %.2f, StdDev: %.2f\n", meanLike, stdDevLike);
            System.out.printf("  Dislike - Mean: %.2f, StdDev: %.2f\n", meanDislike, stdDevDislike);
        }

        // Calculate accuracy on test set
        int correctPredictions = 0;
        for (int i = 0; i < testingData.length; i++) {
            int prediction = gaussianNaiveBayesClassify(trainingData, trainingLabels, testingData[i]);
            if (prediction == testingLabels[i]) {
                correctPredictions++;
            }
        }

        double accuracy = (double) correctPredictions / testingData.length * 100;
        System.out.printf("Gaussian Naive Bayes Accuracy: %.2f%%\n", accuracy);
    }
}