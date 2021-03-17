import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.util.LongAccumulator
import org.apache.log4j._
import scala.collection.mutable.ArrayBuffer

/**
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * This in an implementation of the well-known "Six Degree of Separation Problem".
 * The problem states that the maximum number of hops one node has to go across to reach anyone of
 * the nodes in a social network is at maximum 6.
 * https://en.wikipedia.org/wiki/Six_degrees_of_separation
 * From an algorithmic prospective this can be seen also as the BFS (Breadth First Algorithm)
 * https://en.wikipedia.org/wiki/Breadth-first_search
 */
object DegreesOfSeparation {

  // The nodes we want to find the separation between.
  val startNodeID = 5988
  val targetNodeID = 14

  // We make our accumulator a "global" Option so we can reference it in a mapper later.
  var hitCounter:Option[LongAccumulator] = None

  // Some custom data types
  // BFSData contains an array of hero ID connections, the distance, and color.
  type BFSData = (Array[Int], Int, String)
  // A BFSNode has a heroID and the BFSData associated with it.
  type BFSNode = (Int, BFSData)

  /** Converts a line of raw input into a BFSNode */
  def convertToBFS(line: String): BFSNode = {

    // Split up the line into fields
    val fields = line.split(",")

    // Extract this hero ID from the first field
    val mainNodeID = fields(0).toInt

    // Extract subsequent hero ID's into the connections array
    var connections: ArrayBuffer[Int] = ArrayBuffer()
    for ( connection <- 1 until (fields.length - 1)) {
      connections += fields(connection).toInt
    }

    // Default distance and color is 9999 and white
    var color:String = "WHITE"
    var distance:Int = 9999999

    // Unless this is the character we're starting from
    if (mainNodeID == startNodeID) {
      color = "GRAY"
      distance = 0
    }

    (mainNodeID, (connections.toArray, distance, color))
  }

  /** Create "iteration 0" of our RDD of BFSNodes */
  def createStartingRdd(sc:SparkContext): RDD[BFSNode] = {
    val inputFile = sc.textFile("data/graph.csv")
    inputFile.map(convertToBFS)
  }

  /** Expands a BFSNode into this node and its children */
  def bfsMap(node:BFSNode): Array[BFSNode] = {

    // Extract data from the BFSNode
    val nodeID:Int = node._1
    val data:BFSData = node._2

    val connections:Array[Int] = data._1
    val distance:Int = data._2
    var color:String = data._3

    // This is called from flatMap, so we return an array
    // of potentially many BFSNodes to add to our new RDD
    var results:ArrayBuffer[BFSNode] = ArrayBuffer()

    // Gray nodes are flagged for expansion, and create new
    // gray nodes for each connection
    if (color == "GRAY") {
      for (connection <- connections) {
        val newNodeID = connection
        val newDistance = distance + 1
        val newColor = "GRAY"

        // Have we stumbled across the character we're looking for?
        // If so increment our accumulator so the driver script knows.
        if (targetNodeID == connection) {
          if (hitCounter.isDefined) {
            hitCounter.get.add(1)
          }
        }

        // Create our new Gray node for this connection and add it to the results
        val newEntry:BFSNode = (newNodeID, (Array(), newDistance, newColor))
        results += newEntry
      }

      // Color this node as black, indicating it has been processed already.
      color = "BLACK"
    }

    // Add the original node back in, so its connections can get merged with
    // the gray nodes in the reducer.
    val thisEntry:BFSNode = (nodeID, (connections, distance, color))
    results += thisEntry

    results.toArray
  }

  /** Combine nodes for the same heroID, preserving the shortest length and darkest color. */
  def bfsReduce(data1:BFSData, data2:BFSData): BFSData = {

    // Extract data that we are combining
    val edges1:Array[Int] = data1._1
    val edges2:Array[Int] = data2._1
    val distance1:Int = data1._2
    val distance2:Int = data2._2
    val color1:String = data1._3
    val color2:String = data2._3

    // Default node values
    var distance:Int = 9999
    var color:String = "WHITE"
    var edges:ArrayBuffer[Int] = ArrayBuffer()

    // See if one is the original node with its connections.
    // If so preserve them.
    if (edges1.length > 0) {
      edges ++= edges1
    }
    if (edges2.length > 0) {
      edges ++= edges2
    }

    // Preserve minimum distance
    if (distance1 < distance) {
      distance = distance1
    }
    if (distance2 < distance) {
      distance = distance2
    }

    // Preserve darkest color
    if (color1 == "WHITE" && (color2 == "GRAY" || color2 == "BLACK")) {
      color = color2
    }
    if (color1 == "GRAY" && color2 == "BLACK") {
      color = color2
    }
    if (color2 == "WHITE" && (color1 == "GRAY" || color1 == "BLACK")) {
      color = color1
    }
    if (color2 == "GRAY" && color1 == "BLACK") {
      color = color1
    }
    if (color1 == "GRAY" && color2 == "GRAY") {
      color = color1
    }
    if (color1 == "BLACK" && color2 == "BLACK") {
      color = color1
    }

    (edges.toArray, distance, color)
  }

  /** Our main function where the action happens */
  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Create a SparkContext using every core of the local machine
    val sc = new SparkContext("local[*]", "DegreesOfSeparation")

    // Our accumulator, used to signal when we find the target
    // character in our BFS traversal.
    hitCounter = Some(sc.longAccumulator("Hit Counter"))

    var iterationRdd = createStartingRdd(sc)

    for (iteration <- 1 to 20) {
      println("Running BFS Iteration# " + iteration)

      // Create new vertices as needed to darken or reduce distances in the
      // reduce stage. If we encounter the node we're looking for as a GRAY
      // node, increment our accumulator to signal that we're done.
      val mapped = iterationRdd.flatMap(bfsMap)

      // Note that mapped.count() action here forces the RDD to be evaluated, and
      // that's the only reason our accumulator is actually updated.
      println("Processing " + mapped.count() + " values.")

      if (hitCounter.isDefined) {
        val hitCount = hitCounter.get.value
        if (hitCount > 0) {
          println("Hit the target character! From " + hitCount +
            " different direction(s).")
          return
        }
        if (iteration == 20 && hitCount == 0) {
          println("No connection found...")
          return
        }
      }

      // Reducer combines data for each character ID, preserving the darkest
      // color and shortest path.
      iterationRdd = mapped.reduceByKey(bfsReduce)
    }
  }
}
