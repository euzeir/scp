import scala.math._
import org.apache.spark.SparkContext
import org.apache.log4j._

/**
 *
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * This is an implementation of the KMeans clustering algorithm using old Spark APIs
 * based on RDDs.
 * The purpose here is to cluster a set of data points taken from a dataset of Earthquakes
 * happened between 1970 to 2014. The dataset can be found in the following link:
 * https://data.humdata.org/dataset/catalog-of-earthquakes1970-2014
 * The dataset contains the following fields 12 fields consisting of numeric and non numeric values
 * (DateTime,Latitude,Longitude,Depth,Magnitude,MagType,NbStations,Gap,Distance,RMS,Source,EventID)
 * Here we are going to consider only for numeric (Double) fields to create 4 dimensional points.
 * Each point is 4 coordinates P(Latitude, Longitude, Depth, Magnitude)
 * We have used a number of clusters K = 51 based on this document:
 * https://www.sciencedirect.com/science/article/pii/B9780444410764500213
 * that states that in the planet are present 51 seismic regions.
 * Algorithm implementation based on theory book: ISLR - free edition (chapter 10.3.1)
 * https://www.statlearning.com/
 */

object KMeansClusteringEarthquakeDataAWS {

  // Define the Point custom data type as a list of Doubles : P(x, y, z, w)
  type Point = (Double, Double, Double, Double)

  // The distances between two points (Euclidean distance)
  // https://en.wikipedia.org/wiki/Euclidean_distance
  def distanceBetweenTwoPoints(pointA: Point, pointB: Point): Double = {
    sqrt(pow(pointB._1 - pointA._1, 2) + pow(pointB._2 - pointA._2, 2) + pow(pointB._3 - pointB._3, 2) + pow(pointB._4 - pointB._4, 2))
  }

  // The sum of the coordinates of two points
  // http://mathandmultimedia.com/2011/06/29/addition-of-coordinates/
  def coordinateAddition(pointA: Point, pointB: Point): Point = {
    (pointA._1 + pointB._1, pointA._2 + pointB._2, pointA._3 + pointB._3, pointA._4 + pointB._4)
  }

  //given a single point Pi(x, y, z, w) and an array of points A = [P1, P2, .... Pn] the function returns the
  //index of the point in the array that has the shortest distance to the original point Pi.
  def closestPoint(singlePoint: Point, arrayOfPoints: Array[Point]): Int = {
    var pointIndex = 0
    var isTheClosestPoint = Double.PositiveInfinity

    for (i <- 0 until arrayOfPoints.length) {
      val distance = distanceBetweenTwoPoints(singlePoint, arrayOfPoints(i))
      if (distance < isTheClosestPoint) {
        isTheClosestPoint = distance
        pointIndex = i
      }
    }
    pointIndex
  }

  def main(args: Array[String]) {

    //setting up the logger to avoid warning logs
    Logger.getLogger("org").setLevel(Level.ERROR)

    //creating the spark context in the old fashion (Spark 1)
    val sparkContext = new SparkContext()

    // K is the number of centroids (center points of clusters) to find - 51 Major Earthquake Zones in the Planet
    val K = 51

    // stoppingCondition : this value is used as stopping condition for the KMeans
    val stoppingCondition = .1

    // Load the data file and remove the Header of the 'csv'.
    // by calling textFile on SparkContext i have created an RDD
    // we had previously upload the data in a S3 bucket called 'scpUzeirCoridale'
    val data = sparkContext.textFile("s3n://scp-uzeir-coridale/earthquakes.csv")
      .mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

    // Create the feature list with the columns that we are interested for:
    // Latitude, Longitude, Depth, Magnitude, that represent the coordinates of the points
    // Note: we use the persist() methode to keep in cache the data, in order to reduce
    // the processing time
    val features = data.map(line => line.split(','))
      .map(feature => (feature(1).toDouble, feature(2).toDouble, feature(3).toDouble, feature(4).toDouble))
      .filter(point => !((point._1 == 0) && (point._2 == 0) && (point._3 == 0) && (point._4 == 0)))
      .persist()

    println("Starting the computation...")
    println(s"Number of data points to process: ${features.count()}")
    println("Printing out the shape of a single feature: ")
    for ((a, b, c, d) <- features.take(1)) {
      println("Latitude: " + a + " Longitude : " + b + " Depth: " + c + " Magnitude: " + d)
    }

    // First Step of the Algorithm:
    // Select randomly K = 51 points on the set of all points as temporary centroids
    // The coordinates of the points will change as the iterations goes on
    // Invoking the action: 'takeSample' to extract 51 random points
    println(s"Initializing $K random Centroids")
    val kPoints: Array[Point] = features.takeSample(withReplacement = false, K)

    println("**************************************************")
    println(s"Following $K random Centroids have been initialized : ")
    for ((a, b, c, d) <- kPoints) {
      println("Latitude: " + a + " Longitude : " + b + " Depth: " + c + " Magnitude: " + d)
    }

    // loop until the total distance between one iteration's points and the next is
    // less than the value specified in the 'stoppingCondition'

    // tempDistance: Double is the variable that control the iterations of KMeans
    // when this variable reach a value less than the stoppingCondition then the KMeans stops
    // and returns the computed centroids
    var tempDistance = Double.PositiveInfinity
    var numIterations: Int = 0

    println("**************************************************")
    while (tempDistance > stoppingCondition) {
      numIterations += 1

      // For each point, find the index of the closest centroid
      // The result is a tuple of this shape: (index, (point,1))
      val closestCentroidRDD = features.map(point => (closestPoint(point, kPoints), (point, 1)))

      // For each centroid add the coordinates of all point close to the centroid using 'coordinateAddition'
      // Sum the number of those points for each centroid as the last value of the tuple
      val closestPointsRDD = closestCentroidRDD.reduceByKey { case ((point1, n1), (point2, n2)) => (coordinateAddition(point1, point2), n1 + n2) }

      // For each centroid find a new point by calculating the average of each closest point
      val newPoints = closestPointsRDD.map { case (i, (point, n)) => (i, (point._1 / n, point._2 / n, point._3 / n, point._4 / n)) }.collectAsMap()

      // Calculate the total of the distance between the current centroid points and new points
      tempDistance = 0.0

      for (i <- 0 until K) {
        tempDistance += distanceBetweenTwoPoints(kPoints(i), newPoints(i))
      }

      println(s"Iteration number: " + numIterations)
      println("Distance between iterations: " + tempDistance)

      // Copy the new points to the kPoints array for the next iteration
      for (i <- 0 until K) {
        kPoints(i) = newPoints(i)
      }
    }

    // Display the final center points
    println("**************************************************")
    println("Final Clusters:")
    for (point <- kPoints) {
      println(s"Centroid ::: $point")
    }

    println("**************************************************")

    // take 10 randomly selected device from the dataset and recall the model
    val earthquakeRdd = data.map(line => line.split(','))
      .map(feature => (feature(0), (feature(1).toDouble, feature(2).toDouble, feature(3).toDouble, feature(4).toDouble)))
      .filter(earthquake => !((earthquake._2._1 == 0) && (earthquake._2._2 == 0) && (earthquake._2._3 == 0) && (earthquake._2._4 == 0)))
      .persist()

    var points = earthquakeRdd.takeSample(withReplacement = false, 10)

    for ((earthquakeDate, point) <- points) {
      val EarthquakeZone = closestPoint(point, kPoints)
      println("Earthquake happened on " + earthquakeDate + " belongs to " + EarthquakeZone + " cluster!")
    }
    println("**************************************************")

    sparkContext.stop()
  }
}


