import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._
import org.apache.spark.ml.evaluation.ClusteringEvaluator


/**
 *
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * In this implementation we use Sparks ml library and in particular the KMeans class
 * to cluster a the data contained in a dataset of census data.
 * The dataset contain 51 columns (features) of numeric data and 10000 rows.
 * We cluster the data and then evaluate the result using the 'siluhete metrics'.
 * For more information about the siluhete metrics please look the site:
 * /https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
 * The dataset can be found here:
 * https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
 * https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/
 */
object KMeansClusteringCensusData {

  def main(args: Array[String]): Unit = {
    //log level setup
    Logger.getLogger("org").setLevel(Level.ERROR)

    //create the spark session object
    val spark = SparkSession
      .builder()
      .appName("KMeansClusteringCensusData")
      .master("local[*]")
      .getOrCreate()

    //read the data from the file
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("data/census.csv")

    data.printSchema()

    import spark.implicits._
    val features = (data.select($"dAge", $"dAncstry1", $"dAncstry2", $"iAvail", $"iCitizen", $"iClass",
      $"dDepart", $"iDisabl1", $"iDisabl2", $"iEnglish", $"iFeb55", $"iFertil", $"dHispanic", $"dHour89", $"dHours",
      $"iImmigr", $"dIncome1", $"dIncome2", $"dIncome3", $"dIncome4", $"dIncome5", $"dIncome6", $"dIncome7",
      $"dIncome8", $"dIndustry", $"iKorean", $"iLang1", $"iLooking", $"iMarital", $"iMay75880", $"iMeans", $"iMilitary",
      $"iMobility", $"iMobillim", $"dOccup", $"iOthrserv", $"iPerscare", $"dPOB", $"dPoverty", $"dPwgt1", $"iRagechld",
      $"dRearning", $"iRelat1", $"iRelat2", $"iRemplpar", $"iRiders", $"iRlabor", $"iRownchld", $"dRpincome", $"iRPOB",
      $"iRrelchld", $"iRspouse", $"iRvetserv", $"iSchool", $"iSept80", $"iSex", $"iSubfam1", $"iSubfam2", $"iTmpabsnt",
      $"dTravtime", $"iVietnam", $"dWeek89", $"iWork89", $"iWorklwk", $"iWWII", $"iYearsch", $"iYearwrk", $"dYrsserv"))

    val assembler = (new VectorAssembler().setInputCols(Array("dAge", "dAncstry1", "dAncstry2", "iAvail",
      "iCitizen", "iClass", "dDepart", "iDisabl1", "iDisabl2", "iEnglish", "iFeb55", "iFertil", "dHispanic",
      "dHour89", "dHours", "iImmigr", "dIncome1", "dIncome2", "dIncome3", "dIncome4", "dIncome5", "dIncome6",
      "dIncome7", "dIncome8", "dIndustry", "iKorean", "iLang1", "iLooking", "iMarital", "iMay75880", "iMeans",
      "iMilitary", "iMobility", "iMobillim", "dOccup", "iOthrserv", "iPerscare", "dPOB", "dPoverty", "dPwgt1",
      "iRagechld", "dRearning", "iRelat1", "iRelat2", "iRemplpar", "iRiders", "iRlabor", "iRownchld", "dRpincome",
      "iRPOB", "iRrelchld", "iRspouse", "iRvetserv", "iSchool", "iSept80", "iSex", "iSubfam1", "iSubfam2",
      "iTmpabsnt", "dTravtime", "iVietnam", "dWeek89", "iWork89", "iWorklwk", "iWWII", "iYearsch", "iYearwrk", "dYrsserv"))
      .setOutputCol("features"))

    val trainingData = assembler.transform(features).select("features")

    val kmeans = new KMeans()
      .setK(10)
      .setSeed(1L)
      .setMaxIter(30)
      .setInitMode("random")

    val model = kmeans.fit(trainingData)

    println(model.summary)

    // Make predictions
    val predictions = model.transform(trainingData)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    //printing out the centroids
    for(centoid <- model.clusterCenters) {
      println(s"Centroid : ${centoid} \n")
    }

    println(s"Number of features : ${model.numFeatures}")
    println(s"Distance Measure : ${model.getDistanceMeasure}")

  }
}

