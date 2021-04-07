import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._
import org.apache.spark.ml.evaluation.ClusteringEvaluator

/**
 *
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * This is an implementation of the KMeans clustering algorithm using new Spark APIs
 * based on DataFrames
 * The purpose here is to cluster a set of data points taken from a dataset of Earthquakes
 * happened between 1970 to 2014. The dataset can be found in the following link:
 * https://data.humdata.org/dataset/catalog-of-earthquakes1970-2014
 * The dataset contains the following fields 12 fields consisting of numeric and non numeric values
 * (DateTime,Latitude,Longitude,Depth,Magnitude,MagType,NbStations,Gap,Distance,RMS,Source,EventID)
 * Here we are going to consider only for numeric (Double) fields to create 4 dimensional feature vector.
 * The for dimensions are: (Latitude,Longitude,Depth,Magnitude)
 * We have used a number of clusters K = 51 based on this document:
 * https://www.sciencedirect.com/science/article/pii/B9780444410764500213
 * that states that in the planet are present 51 seismic regions.
 */
object KMeansClusteringEarthquakeMLImpl {

  def main(args: Array[String]): Unit = {
    //log level setup
    Logger.getLogger("org").setLevel(Level.ERROR)

    //create the spark session object
    val spark = SparkSession
      .builder()
      .appName("KMeansClusteringEarthquakeMLImpl")
      .master("local[*]")
      .getOrCreate()

    //read the data from the file
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("data/earthquakes.csv")

    data.printSchema()

    import spark.implicits._
    val features = (data.select($"Latitude", $"Longitude", $"Depth", $"Magnitude"))

    val assembler = (new VectorAssembler().setInputCols(Array("Latitude", "Longitude", "Depth", "Magnitude"))
      .setOutputCol("features"))

    val trainingData = assembler.transform(features).select("features")

    val kMeans = new KMeans()
      .setK(51)
      .setSeed(1L)
      .setMaxIter(Int.MaxValue)
      .setInitMode("random")

    val model = kMeans.fit(trainingData)

    println(model.summary)

    // Make predictions
    val predictions = model.transform(trainingData)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    //printing out the centroids
    for(centroid <- model.clusterCenters) {
      println(s"Centroid : ${centroid}")
    }

    println(s"Number of features : ${model.numFeatures}")
    println(s"Distance Measure : ${model.getDistanceMeasure}")

  }
}


