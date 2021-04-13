import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.util.LongAccumulator
import org.apache.log4j._
import scala.collection.mutable.ArrayBuffer

/**
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * This in an implementation of the well-known "Six Degree of Separation Problem" using BFS
 * The problem states that the maximum number of hops one node has to go across to reach anyone of
 * the nodes in a social network is at maximum 6.
 * https://en.wikipedia.org/wiki/Six_degrees_of_separation
 * From an algorithmic prospective this can be seen also as the BFS (Breadth First Algorithm)
 * https://en.wikipedia.org/wiki/Breadth-first_search
 */
object GraphBfsAWS {

  // The nodes we want to find the separation between.
  val startNodeID = 5988
  val targetNodeID = 15

  // Use accumulator where multiple threads can write.
  var hitCounter:Option[LongAccumulator] = None

  // Some custom data types
  // Data contains an array of hero ID connections, the distance, and color.
  type Data = (Array[Int], Int, String)
  // Node has a startNodeID and the Data associated with it.
  type Node = (Int, Data)

  // convert from files line to Node
  def convertToNode(line: String): Node = {

    // Split up the line into fields
    val fields = line.split(",")

    // Extract this ID from the first field
    val mainNodeID = fields(0).toInt

    // Extract subsequent ID's into the connections array
    val connections: ArrayBuffer[Int] = ArrayBuffer()
    for ( connection <- 1 until (fields.length - 1)) {
      connections += fields(connection).toInt
    }

    // Default distance and color is Infinity and White
    var color:String = "WHITE"
    var distance:Int = Double.PositiveInfinity.toInt

    if (mainNodeID == startNodeID) {
      color = "GRAY"
      distance = 0
    }

    (mainNodeID, (connections.toArray, distance, color))
  }

  // Load the data file and create first iteration
  def initialRDD(sc:SparkContext): RDD[Node] = {
    val inputFile = sc.textFile("graph.csv")
    inputFile.map(convertToNode)
  }

  // Expands a BFSNode into this node and its children
  def expandBFSMap(node:Node): Array[Node] = {

    // Extract data from the Node
    val nodeID:Int = node._1
    val data:Data = node._2

    val connections:Array[Int] = data._1
    val distance:Int = data._2
    var color:String = data._3

    // This is called from flatMap, so we return an array of Nodes to add to RDD
    val results:ArrayBuffer[Node] = ArrayBuffer()

    // Gray nodes need to be expanded and create new gray nodes
    if (color == "GRAY") {
      for (connection <- connections) {
        val newNodeID = connection
        val newDistance = distance + 1
        val newColor = "GRAY"

        // Check if we have reach the target node
        // If so increment our accumulator so the driver script knows.
        if (targetNodeID == connection) {
          if (hitCounter.isDefined) {
            hitCounter.get.add(1)
          }
        }

        // Create our new Gray node for this connection and add it to the results
        val newEntry:Node = (newNodeID, (Array(), newDistance, newColor))
        results += newEntry
      }

      // Color this node as black = has been processed
      color = "BLACK"
    }

    // Add the original node back in, so its connections can get merged with
    // the gray nodes in the reducer.
    val thisEntry:Node = (nodeID, (connections, distance, color))
    results += thisEntry

    results.toArray
  }

  // Combine nodes for the same ID, preserving the shortest length and darkest color
  def reduceBFS(data1:Data, data2:Data): Data = {

    // Extract data that we are combining
    val edges1:Array[Int] = data1._1
    val edges2:Array[Int] = data2._1
    val distance1:Int = data1._2
    val distance2:Int = data2._2
    val color1:String = data1._3
    val color2:String = data2._3

    // Default node values
    var distance:Int = Double.PositiveInfinity.toInt
    var color:String = "WHITE"
    val edges:ArrayBuffer[Int] = ArrayBuffer()

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

  // Main function
  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Create a SparkContext
    val sparkContext = new SparkContext("local[*]", "GraphBfsAWS")

    // Used to signal when we find the target
    hitCounter = Some(sparkContext.longAccumulator("Hit Accumulator"))

    var iterationRdd = initialRDD(sparkContext)

    // We have set the number of iterations up to 20 for security a more reasonable value would be 10
    for (iteration <- 1 to 20) {
      println("************************************")
      println("Running BFS Iteration# " + iteration)

      // Apply expandBFSMap
      val mapped = iterationRdd.flatMap(expandBFSMap)

      // Number of node that are being processed
      println("Processing " + mapped.count() + " values.")

      if (hitCounter.isDefined) {
        val hitCount = hitCounter.get.value
        if (hitCount > 0) {
          println("************************************")
          println(s"The degree of separation between starting node ${startNodeID} and target node ${targetNodeID} is :: ${iteration} hop(s)")
          println("Hit the target character! From " + hitCount + " different direction(s).")
          return
        }
        if (iteration == 20 && hitCount == 0) {
          println("No connection found...")
          return
        }
      }

      // Apply reduceBFS to combine the data for each node
      iterationRdd = mapped.reduceByKey(reduceBFS)
    }
    sparkContext.stop()
  }
}

