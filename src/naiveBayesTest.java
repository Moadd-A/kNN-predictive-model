import java.io.*;

public class naiveBayesTest {
    static void Assert (boolean res){
        if(!res){
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    static int NumberOfFeatures = 14;
    static double[] toFeatureVector(String title, String director, String leadActor, boolean likeIt ,double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {


        double[] feature = new double[NumberOfFeatures];
        feature[0] = id;  // We use the movie id as a numeric attribute.

        switch (genre) { // We also use represent each movie genre as an integer number:


            case "Action":  feature[0] = 1; break;
            case "Fantasy":   feature[1] = 1; break;
            case "Romance": feature[2] = 1; break;
            case "Sci-Fi": feature[3] = 1; break;
            case "Adventure": feature[4] = 1; break;
            case "Horror": feature[5] = 1; break;
            case "Comedy": feature[6] = 1; break;
            case "Thriller": feature[7] = 1; break;
            default: Assert(false);

        }

        if (runtime >= 111.705) {
            feature[8] = 1;
        }else {
            feature[9] = 0;
        }
        if(budget >= 111) {
            feature[10] = 1;
        }else {
            feature[11] = 0;
        }

        if(boxOffice >= 170.918) {
            feature[12] = 1;
        }else {
            feature[13] = 0;
        }

//        if(imdb >= 7.333){
//            feature[12] = 1;
//        }else{
//            feature[12] = 0;
//        }

//        if(rt >= 81.934){
//            feature[13] = 1;
//        }else{
//            feature[13] = 0;
//        }

//        if(year >= 2021){
//            feature[12] = 1;
//        }else{
//            feature[12] = 0;
//        }


        return feature;
    }
    static  int pos_count = 0, neg_count = 0; // variables for counting positive and negative
    static double [] FeatureCountsPos = new double [NumberOfFeatures];
    static double [] FeatureCountsNeg = new double [NumberOfFeatures];
    static class NaiveBayesModel {

        public NaiveBayesModel(){ }

        double estimate(double[] X) { // method to estimate the probability of a positive outcome based on the feature vector
            double s = Math.log((double) pos_count / neg_count);
            Assert(neg_count > 0);
            Assert(pos_count > 0);
            // calculate conditional probabilities for each feature
            for (int x = 0; x < NumberOfFeatures; x++) {
                if (X[x] > 0) {

                    double p_feature_cond_pos = FeatureCountsPos[x] / pos_count;
                    if (p_feature_cond_pos == 0)
                        p_feature_cond_pos = 0.01;

                    double p_feature_cond_neg = FeatureCountsNeg[x] / neg_count;
                    if (p_feature_cond_neg == 0)
                        p_feature_cond_neg = 0.01;

                    double feature_strength = p_feature_cond_pos / p_feature_cond_neg;

                    s = s + Math.log(feature_strength);
                }
            }
            return 1 / (1 + Math.exp((-s)));
        }

        public void Update(double X[], int label) { // method to update feature count based on training data and labels

            Assert(NumberOfFeatures == X.length);
            for (int x = 0; x < NumberOfFeatures; x++) {

                if (label > 0) {
                    if (X[x] > 0)
                        FeatureCountsPos[x]++;
                } else if (X[x] > 0)
                    FeatureCountsNeg[x]++;
            }

            if (label > 0)
                pos_count++;
            else
                neg_count++;
        }

        public int[] ReportAccuracy(double data[][], int labels[]) {
            int[] bayesLabels = new int[data.length];
            double number_correct_predictions = 0;

            for (int j = 0; j < data.length; j++) {
                int prediction;
                if (estimate(data[j]) >= 0.643) {
                    prediction = 1;
                    bayesLabels[j] = 1;

                } else {
                    prediction = 0;
                    bayesLabels[j] = 0;
                }

                if (prediction == labels[j])
                    number_correct_predictions++;
            }
            System.out.printf(" Naive Bayes Accuracy: %.2f%%\n", (float) (number_correct_predictions / data.length * 100));
            return bayesLabels;
        }
        static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels) throws IOException {
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                int idx = 0;
                br.readLine(); // skip header line
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    // Assuming csv format: MovieID,Title,Genre,Runtime,Year,Lead Actor,Director,IMDB,RT(%),Budget,Box Office Revenue (in million $),Like it
                    double id = Double.parseDouble(values[0]);
                    String title = values[1];
                    String genre = values[2];
                    double year = Double.parseDouble(values[3]);
                    String director = values[4];
                    String leadActor = values[5];
                    double rt = Double.parseDouble(values[6]);
                    double imdb = Double.parseDouble(values[7]);
                    double boxOffice = Double.parseDouble(values[8]);
                    double budget = Double.parseDouble(values[9]);
                    double runtime = Double.parseDouble(values[10]);
                    boolean likeIt = Integer.parseInt(values[11]) == 1;
                    dataFeatures[idx] = toFeatureVector(title, director, leadActor, likeIt, id, genre, runtime, year, imdb, rt, budget, boxOffice);           dataLabels[idx] = Integer.parseInt(values[11]); // Assuming the label is the last column and is numeric
                    idx++;
                }
            }
        }



    }

    public static void main(String[] args) {


        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];

        try{
            NaiveBayesModel.loadData("training-set.csv", trainingData, trainingLabels);
            NaiveBayesModel.loadData("testing-set.csv", testingData, testingLabels);
        }
        catch(IOException e){
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        NaiveBayesModel model = new NaiveBayesModel(); // instance of model

        for(int x = 0; x < NumberOfFeatures; x++){
            FeatureCountsPos[x] = 0;
            FeatureCountsNeg[x] = 0;
        }

        for(int j = 0; j < trainingData.length; j++){
            model.Update(trainingData[j], trainingLabels[j]); // use update method
        }

        model.ReportAccuracy(testingData, testingLabels); // accuracy report


    }
    }
