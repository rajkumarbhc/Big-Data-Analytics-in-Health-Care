/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {
  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    val biDirectionalGraph = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(index => index._1).collect().toSet
    val biDirectionalGraphNeighbors = graph.collectNeighborIds(EdgeDirection.Out)
    val subsetGraphNeighbors = biDirectionalGraphNeighbors.filter(subSet => biDirectionalGraph.contains(subSet._1) && subSet._1.toLong != patientID)
    val selectSpecificGraphNeighbors = biDirectionalGraphNeighbors.filter(subSet => subSet._1.toLong == patientID).map(subSet => subSet._2).flatMap(subSet => subSet).collect().toSet
    val GraphNeighborsPatientValues = subsetGraphNeighbors.map(subSet => (subSet._1, jaccard(selectSpecificGraphNeighbors, subSet._2.toSet)))
    GraphNeighborsPatientValues.takeOrdered(10)(Ordering[Double].reverse.on(subSet => subSet._2)).map(_._1.toLong).toList
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    val patientSourceNode = graph.edges.sparkContext
    val biDirectionalGraph = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(subSet => subSet._1).collect().toSet
    val biDirectionalGraphPatients = graph.collectNeighborIds(EdgeDirection.Out).filter(subSet => biDirectionalGraph.contains(subSet._1))
    val biDirectionalGraphPatients2 = biDirectionalGraphPatients.cartesian(biDirectionalGraphPatients).filter(subSet => subSet._1._1 < subSet._2._1)
    biDirectionalGraphPatients2.map(subSet => (subSet._1._1, subSet._2._1, jaccard(subSet._1._2.toSet, subSet._2._2.toSet)))
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    if (a.isEmpty || b.isEmpty) { return 0.0 }
    a.intersect(b).size.toDouble / a.union(b).size.toDouble
  }
}

