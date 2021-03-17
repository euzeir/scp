import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

/**
 *
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * In this implementation we make use of ml and mllib libraries of spark to
 * predict the if a customer will or will not subscribe in a bank service.
 * In particular we have used Logistic Regression to predict the outcome based on
 * a set of column values.
 * The dataset used in this project can be found in this link:
 * https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv
 * The dataset has 41,188 rows and 21 columns (fields) numeric and non-numeric.
 * We use a reducted version of the dataset 10K
 */
object LogisticRegressionBankSubscriptionPrediction {

  def main(args: Array[String]): Unit = {

    //log level setup
    Logger.getLogger("org").setLevel(Level.ERROR)

    //creation of SparkSession (in previous Spark versions was SparkContext)
    val spark = SparkSession
      .builder()
      .appName("BankSubscription")
      .master("local[*]")
      .getOrCreate()

    //reading the local dataset
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("data/banking300.csv")

    //print the dataset schema
    data.printSchema()

    //show the first 10 rows in order to have a better view of the data we are working with
    data.show(10)

    //selecting the needed columns and renaming the "Price" column that we need to predict
    import spark.implicits._
    val subscriptionData = (data.select(data("y").as("label"), $"age", $"job", $"marital",
      $"education", $"default", $"housing", $"loan", $"contact", $"month", $"day_of_week",
      $"duration", $"campaign", $"pdays", $"previous", $"poutcome", $"emp_var_rate",
      $"cons_price_idx", $"cons_conf_idx", $"euribor3m", $"nr_employed"))

    //covert categorical columns that are in string format into numeric values
    val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
    val defaultIndexer = new StringIndexer().setInputCol("default").setOutputCol("defaultIndex")
    val housingIndexer = new StringIndexer().setInputCol("housing").setOutputCol("housingIndex")
    val loanIndexer = new StringIndexer().setInputCol("loan").setOutputCol("loanIndex")
    val contactIndexer = new StringIndexer().setInputCol("contact").setOutputCol("contactIndex")
    val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
    val dayOfWeekIndexer = new StringIndexer().setInputCol("day_of_week").setOutputCol("dayOfWeekIndex")
    val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")

    //convert numerical values into One Hot Encoding (0-1 strings)
    val jobOHE = new OneHotEncoder().setInputCol("jobIndex").setOutputCol("jobVector")
    val maritalOHE = new OneHotEncoder().setInputCol("maritalIndex").setOutputCol("maritalVector")
    val educationOHE = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationVector")
    val defaultOHE = new OneHotEncoder().setInputCol("defaultIndex").setOutputCol("defaultVector")
    val housingOHE = new OneHotEncoder().setInputCol("housingIndex").setOutputCol("housingVector")
    val loanOHE = new OneHotEncoder().setInputCol("loanIndex").setOutputCol("loanVector")
    val contactOHE = new OneHotEncoder().setInputCol("contactIndex").setOutputCol("contactVector")
    val monthOHE = new OneHotEncoder().setInputCol("monthIndex").setOutputCol("monthVector")
    val dayOfWeekOHE = new OneHotEncoder().setInputCol("dayOfWeekIndex").setOutputCol("dayOfWeekVector")
    val poutcomeOHE = new OneHotEncoder().setInputCol("poutcomeIndex").setOutputCol("poutcomeVector")

    //create the (label, features vector) with the columns to use from the model
    val assembler = (new VectorAssembler().setInputCols(Array("age", "jobVector",
      "maritalVector", "educationVector", "defaultVector", "housingVector", "loanVector",
      "contactVector", "monthVector", "dayOfWeekVector", "duration", "campaign", "pdays",
      "previous", "poutcomeVector", "emp_var_rate", "cons_price_idx", "cons_conf_idx",
      "euribor3m", "nr_employed")).setOutputCol("features"))

    //creating the training and testing sets in 70% - 30%
    val Array(train, test) = subscriptionData.randomSplit(Array(0.7, 0.3), seed = 12345)

    //create the logistic regression object
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")

    //create the pipeline - all the stages the data has to pass
    val pipeline = (new Pipeline().setStages(Array(jobIndexer, maritalIndexer, educationIndexer,
      defaultIndexer, housingIndexer, loanIndexer, contactIndexer, monthIndexer, dayOfWeekIndexer,
      poutcomeIndexer, jobOHE, maritalOHE, educationOHE, defaultOHE, housingOHE, loanOHE, contactOHE,
      monthOHE, dayOfWeekOHE, poutcomeOHE, assembler, lr)))

    //create the model and train it on the "train" data
    val lrModel = pipeline.fit(train)

    //test the model using the "test" data
    val predictions = lrModel.transform(test)

    //show the schema of the results
    predictions.printSchema()

    // Select example rows to display.
    predictions.select("label", "features").show(50)

    //prediction
    val predictionAndLabels = predictions.select($"prediction", $"label").as[(Double, Double)].rdd

    //used to show the difference between predicted value and real value
    val predictionAndActual = predictions.select($"prediction", $"label").collect()

    val metrics1 = new MulticlassMetrics(predictionAndLabels)
    val metrics2 = new BinaryClassificationMetrics(predictionAndLabels)

    //printing out the accuracy
    println(s"The accuracy of the prediction: ${metrics1.accuracy}")
    println((s"ROC : ${metrics2.roc().toString()}"))

    for(pred <- predictionAndActual) {
      println(s"Prediction - Actual : ${pred}")
    }
  }

}

