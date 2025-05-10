import java.io.IOException;

public class ModelThree {
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
        return count > 1 ? Math.sqrt(sumSquaredDiff / (count - 1)) + 1.0 : 1;
    }

    // Calculate Gaussian likelihood
    static double gaussianLikelihood(double x, double mean, double stdDev) {
        if (stdDev <= 0) return 0.0;
        double exponent = -Math.pow(x - mean, 2) / (2 * Math.pow(stdDev, 2));
        return exponent;
    }

    // Calculate genre likelihood
    static double genreLikelihood(double[][] data, int[] labels, int classLabel, double[] testFeature) {
        double likelihood = 0;
        // Look at genre features (0-7)
        for (int i = 0; i < 8; i++) {
            if (testFeature[i] > 0) {  // If this genre is present
                int genreCount = 0;
                int totalClassCount = 0;
                for (int j = 0; j < data.length; j++) {
                    if (labels[j] == classLabel) {
                        totalClassCount++;
                        if (data[j][i] > 0) {
                            genreCount++;
                        }
                    }
                }
                if (totalClassCount > 0) {
                    double genreProb = (genreCount + 1.0) / (totalClassCount + 2.0); // Laplace smoothing
                    likelihood += Math.log(genreProb);
                }
            }
        }
        return likelihood;
    }

    // Feature weights based on observed separation
//    static double getFeatureWeight(int featureIndex) {
//        switch(featureIndex) {
//            case 8:  return 2.0;    // budget
//            case 9:  return 1.0;    // year
//            case 10: return 1.5;    // IMDB
//            case 11: return 2.0;    // box office
//            case 12: return 1.5;    // RT score
//            case 13: return 1.5;    // runtime
//            default: return 1.0;
//        }
//    }
    static double getFeatureWeight(int featureIndex) {
        switch(featureIndex) {
            case 8:  return 2.0;    // budget
            case 9:  return 2.0;    // year
            case 10: return 0.5;    // IMDB
            case 11: return 3.0;    // box office
            case 12: return 0.5;    // RT score
            case 13: return 1.0;    // runtime
            default: return 1.0;
        }
    }


    // Main classification method using Gaussian Naive Bayes
    static int gaussianNaiveBayesClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature) {
        double logPriorLike = Math.log(0.61); // log the prior probability of liking
        double logPriorDislike = Math.log(0.39); // log the prior probability of disliking

        // Make variables for the above
        double logProbLike = logPriorLike;
        double logProbDislike = logPriorDislike;

        // Add genre likelihoods
        logProbLike += genreLikelihood(trainingData, trainingLabels, 1, testFeature);
        logProbDislike += genreLikelihood(trainingData, trainingLabels, 0, testFeature);

        // For each numerical feature
        for (int feature = 8; feature < 14; feature++) {
            // mean and standard deviation for like and dislike
            double meanLike = calculateMean(trainingData, trainingLabels, 1, feature);
            double stdDevLike = calculateStdDev(trainingData, trainingLabels, 1, feature, meanLike);

            double meanDislike = calculateMean(trainingData, trainingLabels, 0, feature);
            double stdDevDislike = calculateStdDev(trainingData, trainingLabels, 0, feature, meanDislike);

            double weight = getFeatureWeight(feature);
            // Update the log-probabilities using the weight metric
            logProbLike += weight * gaussianLikelihood(testFeature[feature], meanLike, stdDevLike);
            logProbDislike += weight * gaussianLikelihood(testFeature[feature], meanDislike, stdDevDislike);
        }
            // compare the log probabilities and return the predicted case. if like has higher probability return 1 and otherwise 0
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

        // Calculate and print confusion matrix
        int truePositives = 0, falsePositives = 0;
        int falseNegatives = 0, trueNegatives = 0;

        for (int i = 0; i < testingData.length; i++) {
            int prediction = gaussianNaiveBayesClassify(trainingData, trainingLabels, testingData[i]);
            if (prediction == 1 && testingLabels[i] == 1) truePositives++;
            if (prediction == 1 && testingLabels[i] == 0) falsePositives++;
            if (prediction == 0 && testingLabels[i] == 1) falseNegatives++;
            if (prediction == 0 && testingLabels[i] == 0) trueNegatives++;
        }


        double accuracy = (double)(truePositives + trueNegatives) / testingData.length * 100;
        System.out.printf("\nGaussian Naive Bayes Accuracy: %.2f%%\n", accuracy);
    }
}