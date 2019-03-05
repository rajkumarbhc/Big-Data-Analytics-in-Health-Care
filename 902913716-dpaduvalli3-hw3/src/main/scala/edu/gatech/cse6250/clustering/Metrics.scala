/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.clustering

import breeze.linalg.max
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   * \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   *
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    val sizeofSamples = clusterAssignmentAndLabel.count().toDouble
    val clusterPurity = clusterAssignmentAndLabel.map(key => (key, 1))
      .keyBy(_._1)
      .reduceByKey((key, value) => (key._1, key._2 + value._2))
      .map(key => (key._2._1._1, key._2._2))
      .keyBy(_._1)
      .reduceByKey((key, value) => (1, max(key._2, value._2)))
      .map(key => key._2._2)
      .reduce((key, value) => key + value)
    clusterPurity / sizeofSamples
  }
}
