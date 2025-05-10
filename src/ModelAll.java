import java.io.IOException;

public class ModelAll {


    public static void main(String[] args) {
        int[] modelOneArray = ModelOne();
        int[] modelTwoArray = ModelTwo();
        int[] modelThreeArray = ModelThree();
        int[] testLabels = testingLabels();


        int correctPredictions = 0;
        for (int i = 0; i < modelOneArray.length; i++) {

            if ((modelOneArray[i] == modelTwoArray[i])) {
                if(modelOneArray[i] == testLabels[i]){
                    correctPredictions ++;
                }

            }
            else{
                if(modelThreeArray[i] == testLabels[i]) {
                    correctPredictions ++;
                }
            }

        }
        System.out.println("Accuracy: " +correctPredictions + "%");

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

                // Compute accuracy on the testing set
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

        // Compute accuracy on the testing set
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

       // Compute accuracy on the testing set
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
            // You may need to change the path:
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








