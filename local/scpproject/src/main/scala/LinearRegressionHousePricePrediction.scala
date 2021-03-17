import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.log4j._


/**
 *
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * In this implementation we use Spark machine learning libraries ml and mllib
 * to create a machine learning model that uses Linear Regression to predict the price
 * of the houses based on a set of features.
 * The dataset has 6 columns and 5000 rows of all numeric values.
 * The purpose of the implementation is to predict the price of the house based on the given
 * list of features.
 * Before training we have split the dataset into 2 sets (train / test)
 *
 */
object LinearRegressionHousePricePrediction {

  def main(args: Array[String]): Unit = {
    //log level setup
    Logger.getLogger("org").setLevel(Level.ERROR)

    //create the spark session object
    val spark = SparkSession
      .builder()
      .appName("HousePricePrediction")
      .master("local[*]")
      .getOrCreate()

    //read the data from the file
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("data/housing.csv")

    //selecting the needed columns and renaming the "Price" column that we need to predict
    import spark.implicits._
    val df = (data.select(data("Price").as("label"),
      $"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms",
      $"Avg Area Number of Bedrooms", $"Area Population"))

    //creating the feature vector with all the rows except the one we want to predict (Price)
    val assembler = (new VectorAssembler()
      .setInputCols(Array("Avg Area Income", "Avg Area House Age",
        "Avg Area Number of Rooms", "Avg Area Number of Bedrooms",
        "Area Population")).setOutputCol("features"))

    //transform the dataframe in 2 columns: lable <----> features
    val output = assembler.transform(df).select($"label", $"features")

    //training and test data
    val Array(train, test) = output.select("label", "features").randomSplit(Array(0.7, 0.3), seed = 12345)

    //create the Linear Regression model
    val lr = new LinearRegression()

    //create the Parameter Grid
    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(10000, 100, 0.1)).build()

    //train split
    //here we can use different evaluators like:
    // - rmse (root mean squared error) - default value,
    // - mse (mean squared error),
    // - r2 (R2 metric),
    // - mae (mean absolute error)
    val trainValidationSplit = (new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator().setMetricName("rmse"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      )

    //fit the model
    val model = trainValidationSplit.fit(train)

    println(model.bestModel)

    //show the results of the prediction
    model.transform(test).select("features", "label", "prediction").show()

  }

}

