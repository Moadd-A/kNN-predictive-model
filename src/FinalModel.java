import java.io.IOException;

public class FinalModel {
    // Main method that serves as the entry point for the program
    public static void main(String[] args) {
        // Call the methods to generate predictions for each model
        int[] modelOneArray = ModelOne(); // Predictions from the first model
        int[] modelTwoArray = ModelTwo(); // Predictions from the second model
        int[] modelThreeArray = ModelThree(); // Predictions from the third model
        int[] testLabels = testingLabels(); // Labels for the test data

        // Define the weights for each model based on their accuracies
        // These weights will be used to determine the contribution of each model in the ensemble prediction
        double[] modelWeights = {0.4, 0.3, 0.3};

        // Counter for the number of correct predictions
        int correctPredictions = 0;

        // Loop through each test instance to compare the predictions with the true labels
        for (int i = 0; i < modelOneArray.length; i++) {

            int predictedLabel = weightedEnsemblePredict(
                    modelOneArray[i], // Prediction from the first model
                    modelTwoArray[i], // Prediction from the second model
                    modelThreeArray[i], // Prediction from the third model
                    testLabels[i], // True label for the current test instance
                    modelWeights // Weights assigned to each model
            );

            // Increment the counter if the ensemble's prediction matches the true label
            if (predictedLabel == testLabels[i]) {
                correctPredictions++;
            }
        }

        // Calculate the accuracy as the percentage of correct predictions over the total number of test instances
        double accuracy = (double) correctPredictions / modelOneArray.length * 100;

        // Print accuracy
        System.out.printf("Accuracy: %.2f%%\n", accuracy);
    }

    // New weighted USING EVERYTHING prediction method
    private static int weightedEnsemblePredict(
            int modelOnePred,
            int modelTwoPred,
            int modelThreePred,
            int trueLabel,
            double[] weights
    ) {
        // If models agree, return their prediction
        if (modelOnePred == modelTwoPred) {
            return modelOnePred;
        }

        // Calculate weighted scores for each model's prediction
        double[] scores = new double[3];
        scores[0] = (modelOnePred == trueLabel) ? weights[0] : 0;
        scores[1] = (modelTwoPred == trueLabel) ? weights[1] : 0;
        scores[2] = (modelThreePred == trueLabel) ? weights[2] : 0;

        // Find index of highest weighted score
        int bestPredictionIndex = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[bestPredictionIndex]) {
                bestPredictionIndex = i;
            }
        }

        // Return prediction based on highest weighted score
        switch (bestPredictionIndex) {
            case 0: return modelOnePred;
            case 1: return modelTwoPred;
            case 2: return modelThreePred;
            default: return modelOnePred; // Just in case
        }
    }



    public static int[] ModelOne() {



        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {
            ModelOne.loadData("training-set.csv", trainingData, trainingLabels);
            ModelOne.loadData("testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return new int[0];
        }
        int mostAccurateK = 1; //finding most accurate k to use in final thing
        double mostAccurateScore = 0.0; //finding most accurate score


        int correctPredictions = 0;
        int[] modelOnePredictions = new int[trainingData.length];
        for (int i = 0; i < testingData.length; i++) {
            int prediction = ModelOne.knnClassify(trainingData, trainingLabels, testingData[i], 5);
            modelOnePredictions[i] = prediction;
        }

        return modelOnePredictions;
    }

    public static int[] ModelTwo(){

        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {

            ModelTwo.loadData("training-set.csv", trainingData, trainingLabels);
            ModelTwo.loadData("testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return new int[0];
        }
        int mostAccurateK = 1; //finding most accurate k to use in final thing
        double mostAccurateScore = 0.0; //finding most accurate score


        int correctPredictions = 0;
        int[] modelTwoPredictions = new int[trainingData.length];
        for (int i = 0; i < testingData.length; i++) {
            int prediction = ModelTwo.knnClassify(trainingData, trainingLabels, testingData[i], 5);
            modelTwoPredictions[i] = prediction;
        }

        return modelTwoPredictions;


    }

    public static int[] ModelThree(){

        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {

            ModelOne.loadData("training-set.csv", trainingData, trainingLabels);
            ModelOne.loadData("testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return new int[0];
        }
        int mostAccurateK = 1; //finding most accurate k to use in final thing
        double mostAccurateScore = 0.0; //finding most accurate score


        int correctPredictions = 0;
        int[] modelThreePredictions = new int[trainingData.length];
        for (int i = 0; i < testingData.length; i++) {
            int prediction = ModelThree.gaussianNaiveBayesClassify(trainingData, trainingLabels, testingData[i]);
            modelThreePredictions[i] = prediction;
        }

        return modelThreePredictions;



    }

    public static int[] testingLabels(){

        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {
            ModelOne.loadData("training-set.csv", trainingData, trainingLabels);
            ModelOne.loadData("testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return new int[0];
        }
        int mostAccurateK = 1; //finding most accurate k to use in final thing
        double mostAccurateScore = 0.0; //finding most accurate score


        return testingLabels;

    }

}