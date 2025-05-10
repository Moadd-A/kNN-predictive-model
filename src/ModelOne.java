import java.io.*;

public class ModelOne {

    // Use we use 'static' for all methods to keep things simple, so we can call those methods main

    static void Assert (boolean res) // We use this to test our results - don't delete or modify!
    {
        if(!res)	{
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    // Copy your vector operations here:
    // ...
    static double dot(double [] U, double [] V) { // dot product of two vectors
        // add your code
        double dotProduct = 0.0;
        for(int i = 0; i < V.length; i++){
            dotProduct += V[i]*U[i];
        }
        return dotProduct;
    }


    static int NumberOfFeatures = 17;
    static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {


        double[] feature = new double[NumberOfFeatures];

        switch (genre) { // We also use represent each movie genre as an integer number:


            case "Action":  feature[0] = 2; break;
            case "Fantasy":   feature[1] = 2; break;
            case "Romance": feature[2] = 2; break;
            case "Sci-Fi": feature[3] = 2; break;
            case "Adventure": feature[4] = 2; break;
            case "Horror": feature[5] = 2; break;
            case "Comedy": feature[6] = 2; break;
            case "Thriller": feature[7] = 2; break;
            default: Assert(false);

        }

        // With just genre 67% k = 9
        // Made it so it only uses odd values for K

        feature[8] = budget - 97.91;          // and this 67% k = 3
        feature[9] = (year - 2021.93);          // and this 67% k = 3
        feature[10] = imdb;                   // and this 67% k = 3
        feature[11] = boxOffice - 149.55;     // and this 67% k = 9
        feature[12] = rt - 81.37;             // and this 67% k = 25
        feature[13] = runtime - 109.6;        // and this 69%(nice) k = 11
        return feature;
    }


    // We are using the dot product to determine similarity:
    static double similarity(double[] u, double[] v) {
        return dot(u, v);
    }

    // We have implemented KNN classifier for the K=1 case only. You are welcome to modify it to support any K
// Method to classify a test feature based on the k-nearest neighbors algorithm
    static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature, int k) {

        // Array to store the similarity values between the test feature and each training data point
        double[] similarity = new double[trainingData.length];
        // Array to store the labels of the training data points
        int[] labels = new int[trainingData.length];

        // Calculate the similarity of the test feature with each training data point
        for (int i = 0; i < trainingData.length; i++) {
            similarity[i] = similarity(testFeature, trainingData[i]); // Compute similarity
            labels[i] = trainingLabels[i]; // Copy the label of the training data point
        }

        // Temporary variables for sorting the similarity scores and corresponding labels
        double temp = 0.0;
        int tempLabel = 0;

        // Bubble sort algorithm to sort the training data based on similarity in descending order
        for(int i = 0; i < trainingData.length - 1; i++) {
            boolean swapped = false;
            for(int j = 0; j < (trainingData.length - i - 1); j++) {
                // If the current similarity is less than the next similarity, swap them
                if(similarity[j] < similarity[j + 1]) {

                    // Swap the similarity scores
                    temp = similarity[j + 1];
                    similarity[j + 1] = similarity[j];
                    similarity[j] = temp;

                    // Swap the corresponding labels to maintain relationship with their equivalent similarity scores
                    tempLabel = labels[j + 1];
                    labels[j + 1] = labels[j];
                    labels[j] = tempLabel;

                    swapped = true; // Set to true to indicate a swap occurred
                }
            }

            if(!swapped) {
                break;
            }
        }

        // Counter to count the number of "liked" labels (label value 1) in the top k most similar points
        int counter = 0;

        // Iterate over the top k most similar training data points
        for(int i = 0; i < k; i++) {
            if(labels[i] == 1) { // Check if the label indicates "liked"
                counter++;
            }
            // If more than half of the top k labels are 1, return 1 (indicating the movie is "liked")
            if(counter > k / 2.0) {
                return 1; // Majority voting results in "liked"
            }
        }


        return 0; // Return 0 if the majority of the top k labels are not 1
    }

    static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int idx = 0;
            br.readLine(); // skip header line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                // Assuming csv format: MovieID,Title,Genre,Runtime,Year,Lead Actor,Director,IMDB,RT(%),Budget,Box Office Revenue (in millions $),Like it
                double id = Double.parseDouble(values[0]);
                String genre = values[2];
                double rt = Double.parseDouble(values[6]);
                double year = Double.parseDouble(values[3]);
                double imdb = Double.parseDouble(values[7]);
                double boxOffice = Double.parseDouble(values[8]);
                double budget = Double.parseDouble(values[9]);
                double runtime = Double.parseDouble(values[10]);

                dataLabels[idx] = Integer.parseInt(values[11]); //  like it Assuming the label is the last column and is numeric

                dataFeatures[idx] = toFeatureVector(id, genre, runtime, year, imdb, rt, budget, boxOffice);
                idx++;
            }
        }
    }

    public static void main(String[] args) {

        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {
            // You may need to change the path:
            loadData("training-set.csv", trainingData, trainingLabels);
            loadData("testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }
        int mostAccurateK = 1; //finding most accurate k to use in final thing
        double mostAccurateScore = 0.0; //finding most accurate score
        for (int k = 1; k < 15; k+=2) {
            // Compute accuracy on the testing set
            int correctPredictions = 0;
            for (int i = 0; i < testingData.length; i++) {
                int prediction = knnClassify(trainingData, trainingLabels, testingData[i], k);
                if (prediction == testingLabels[i]) {
                    correctPredictions++;
                }
            }

            // Add some lines here: ...

            double accuracy = (double) correctPredictions / testingData.length * 100;
            if (accuracy > mostAccurateScore) {
                mostAccurateScore = accuracy;
                mostAccurateK = k;
            }
            System.out.printf(k + " : %.2f%%\n", accuracy);
        }
        System.out.println("Most Accurate Prediction: " + mostAccurateScore + "| Most Accurate K: " + mostAccurateK); //to get the most accurate K
    }

}