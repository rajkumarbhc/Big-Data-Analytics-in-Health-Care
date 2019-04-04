package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {
  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    val initializePageRank = patientID
    val vertexSource: VertexId = patientID
    var graphRandom: Graph[Double, Double] = graph.outerJoinVertices(graph.outDegrees) { (x, y, z) => z.getOrElse(0) }.mapTriplets(line => 1.0 / line.srcAttr, TripletFields.Src).mapVertices { (x, y) => if (!(x != vertexSource)) 1.0 else 0.0 }
    def subtractDiffrence(sourceN: VertexId, sourceV: VertexId): Double = { if (sourceN == sourceV) 1.0 else 0.0 }
    var rept = 0
    var graphRandomVOne: Graph[Double, Double] = null
    while (rept < numIter) {
      graphRandom.cache()
      val prep = graphRandom.aggregateMessages[Double](spd => spd.sendToDst(spd.srcAttr * spd.attr), _ + _, TripletFields.Src)
      graphRandomVOne = graphRandom
      val prep2 = { (vertexSource: VertexId, vertexDest: VertexId) => alpha * subtractDiffrence(vertexSource, vertexDest) }
      graphRandom = graphRandom.outerJoinVertices(prep) { (x, y, z) => prep2(vertexSource, x) + (1.0 - alpha) * z.getOrElse(0.0) }.cache()
      graphRandom.edges.foreachPartition(vertices => {})
      graphRandomVOne.vertices.unpersist(false)
      graphRandomVOne.edges.unpersist(false)
      rept = rept + 1
    }
    val randomWalkGraph = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(index => index._1).collect().toSet
    val randomWalkGraphFinal = graphRandom.vertices.filter(n => randomWalkGraph.contains(n._1)).takeOrdered(11)(Ordering[Double].reverse.on(n => n._2)).map(_._1)
    randomWalkGraphFinal.slice(1, randomWalkGraphFinal.length).toList
  }
}
