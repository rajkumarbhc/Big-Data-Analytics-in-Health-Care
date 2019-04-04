/**
 * @author Sungtae An <stan84@gatech.edu>.
 */

package edu.gatech.cse6250.clustering

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{ PowerIterationClustering => PIC }

/**
 * Power Iteration Clustering (PIC), a scalable graph clustering algorithm developed by
 * [[http://www.icml2010.org/papers/387.pdf Lin and Cohen]]. From the abstract: PIC finds a very
 * low-dimensional embedding of a dataset using truncated power iteration on a normalized pair-wise
 * similarity matrix of the data.
 *
 * @see [[http://en.wikipedia.org/wiki/Spectral_clustering Spectral clustering (Wikipedia)]]
 */

object PowerIterationClustering {
  def runPIC(similarities: RDD[(Long, Long, Double)]): RDD[(Long, Int)] = {
    val sc = similarities.sparkContext
    val implementPowerIterationClustering = new PIC().setK(3).setMaxIterations(100).setInitializationMode("degree")
    val assignPIC = implementPowerIterationClustering.run(similarities)
    assignPIC.assignments.map(index => (index.id, index.cluster))
  }
}