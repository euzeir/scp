import scala.math.pow
import org.apache.spark.SparkContext
import org.apache.log4j._

object KMeansClusteringEarthquakeData {


  // The squared distances between two points
  def distanceSquared(pointA: (Double, Double, Double, Double), pointB: (Double, Double, Double, Double)): Double = {
    pow(pointB._1 - pointA._1, 2) + pow(pointB._2 - pointA._2, 2) + pow(pointB._3 - pointB._3, 2) + pow(pointB._4 - pointB._4, 2)
  }

  // The sum of two points
  def addPoints(pointA: (Double, Double, Double, Double), pointB: (Double, Double, Double, Double)): (Double, Double, Double, Double) = {
    (pointA._1 + pointB._1, pointA._2 + pointB._2, pointA._3 + pointB._3, pointA._4 + pointB._4)
  }

  // for a point p and an array of points, return the index in the array of the point closest to p
  def closestPoint(singlePoint: (Double, Double, Double, Double), arrayOfPoints: Array[(Double, Double, Double, Double)]): Int = {
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- 0 until arrayOfPoints.length) {
      val dist = distanceSquared(singlePoint, arrayOfPoints(i))
      if (dist < closest) {
        closest = dist
        bestIndex = i
      }
    }
    bestIndex
  }

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkContext = new SparkContext("local[*]", "KMeansClusteringEarthquakeData")

    // K is the number of means (center points of clusters) to find - 20 Major Earthquake Zones in the Planet
    val K = 20

    // ConvergeDist -- the threshold "distance" between iterations at which we decide we are done
    val convergeDist = .1

    // Parse the device status data file into pairs

    val data = sparkContext.textFile("data/earthquakes.csv")

    //create the feature list
    val features = data.map(line => line.split(','))
      .map(feature => (feature(1).toDouble, feature(2).toDouble, feature(3).toDouble, feature(4).toDouble))
      .filter(point => !((point._1 == 0) && (point._2 == 0) && (point._3 == 0) && (point._4 == 0)))
      .persist()

    println("Starting the computation...")
    println(s"Number of features to process: ${features.count()}")
    println("Printing out the shape of a single feature: ")
    for ((a, b, c, d) <- features.take(1)) {
      println("Latitude: " + a + " Longitude : " + b + " Depth: " + c + " Magnitude: " + d)
    }

    //start with K randomly selected points from the dataset as center points

    var kPoints = features.takeSample(withReplacement = false, K)

    println("**************************************************")
    println(s"Initialize $K random Centroids: ")
    for ((a, b, c, d) <- kPoints) {
      println("Latitude: " + a + " Longitude : " + b + " Depth: " + c + " Magnitude: " + d)
    }

    // loop until the total distance between one iteration's points and the next is less than the convergence distance specified
    var tempDistance = Double.PositiveInfinity

    println("**************************************************")
    while (tempDistance > convergeDist) {

      // For each key (k-point index), find a new point by calculating the average of each closest point

      // for each point, find the index of the closest kpoint.
      // map to (index, (point,1)) as follow:
      // (1, ((2.1,-3.4),1))
      // (0, ((5.1,-7.4),1))
      // (1, ((8.1,-4.4),1))
      val closestToKpointRdd = features.map(point => (closestPoint(point, kPoints), (point, 1)))

      // For each key (k-point index), reduce by sum (addPoints) the latitudes and longitudes of all the points closest to that k-point, and the number of closest points
      // E.g.
      // (1, ((4.325,-5.444),2314))
      // (0, ((6.342,-7.532),4323))
      // The reduced RDD should have at most K members.

      //val pointCalculatedRdd = closestToKpointRdd.reduceByKey((v1, v2) => ((addPoints(v1._1, v2._1), v1._2 + v2._2)))
      val pointCalculatedRdd = closestToKpointRdd.reduceByKey { case ((point1, n1), (point2, n2)) => (addPoints(point1, point2), n1 + n2) }

      // For each key (k-point index), find a new point by calculating the average of each closest point
      // (index, (totalX,totalY),n) to (index, (totalX/n,totalY/n))

      //val newPointRdd = pointCalculatedRdd.map(center => (center._1, (center._2._1._1 / center._2._2, center._2._1._2 / center._2._2))).sortByKey()
      val newPoints = pointCalculatedRdd.map { case (i, (point, n)) => (i, (point._1 / n, point._2 / n, point._3 / n, point._4 / n)) }.collectAsMap()

      // calculate the total of the distance between the current points (kPoints) and new points (localAverageClosestPoint)

      tempDistance = 0.0

      for (i <- 0 until K) {
        // That distance is the delta between iterations. When delta is less than convergeDist, stop iterating
        tempDistance += distanceSquared(kPoints(i), newPoints(i))
      }

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
      .filter(earthquake => !((earthquake._2._1 == 0) && (earthquake._2._2 == 0)))
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

