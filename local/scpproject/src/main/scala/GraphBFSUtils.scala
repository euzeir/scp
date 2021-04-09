import org.apache.spark._
import org.apache.log4j._
import java.io._

/**
 * Scalable and Cloud Programming - e.uzeir & d.coriale
 *
 * This file is used to transform our dataset into a suitable form in order to
 * be able to process it.
 * The dataset that we are going to use can be found in this url
 * http://networksciencebook.com/translations/en/resources/data.html
 * The data is in the form of 2 columns A B and we are going to transform it into a list of numbers
 * where the first number represent the first node and the other number the list of nodes connected to it.
 */
object GraphBFSUtils {
  def main(args: Array[String]): Unit = {

    //log level setup - we want only to get Error logs if any
    Logger.getLogger("org").setLevel(Level.ERROR)

    //create the spark context
    val sc = new SparkContext("local[*]", "GraphBFSUtils")

    //load the data
    val lines = sc.textFile("data/road.csv")

    //parse the file and take out the needed fields (NodeId, ConnectedTo)
    def prepareFile(line: String) = {
      val fields = line.split(",")
      val nodeId = fields(0)
      val connectedTo = fields(1)
      (nodeId, connectedTo)
    }

    //convert each line to a string, split it on different words and take the first column
    val node = lines.map(prepareFile)

    //group the nodes based on unique keys and their list of connections
    val listOfNodes = node.groupByKey().map(x => (x._1, x._2.toList)).collect()

    //print the file in console - just to show them!!
    val printListOfNodes = listOfNodes.foreach(println)

    //clean the file and write it locally using javas write utilities
    val file = new File("data/roadT.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    for(item <- listOfNodes) {
      val a1 = item._1
      val a2 = item._2
      bw.write(a1.toString() + a2.toString()
        .replace("List", "")
        .replace("(", ",")
        .replace(")","")
        .replace(", ",",") + "\n")
    }
    bw.close()
  }
}

