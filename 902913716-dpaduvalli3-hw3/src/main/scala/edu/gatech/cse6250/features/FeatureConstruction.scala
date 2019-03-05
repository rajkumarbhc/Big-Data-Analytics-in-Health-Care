package edu.gatech.cse6250.features

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su
 */
object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   *
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val df = diagnostic.map(index => ((index.patientID, index.code), 1.0)).reduceByKey(_ + _)
    df
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   *
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val mf = medication.map(index => ((index.patientID, index.medicine), 1.0)).reduceByKey(_ + _)
    mf
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   *
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */

    val ls = labResult.map(index => ((index.patientID, index.testName), index.value)).reduceByKey(_ + _)
    val lc = labResult.map(index => ((index.patientID, index.testName), 1.0)).reduceByKey(_ + _)
    val lf = ls.join(lc).map(index => (index._1, index._2._1 / index._2._2))
    lf
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   *
   * @param diagnostic   RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val df = diagnostic.map(index => ((index.patientID, index.code), 1.0)).reduceByKey(_ + _)
    val dt = df.filter(index => (candiateCode.contains(index._1._2)))
    dt
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   *
   * @param medication          RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val mf = medication.map(index => ((index.patientID, index.medicine), 1.0)).reduceByKey(_ + _)
    val mt = mf.filter(index => (candidateMedication.contains(index._1._2)))
    mt
  }

  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   *
   * @param labResult    RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val ls = labResult.map(index => ((index.patientID, index.testName), index.value)).reduceByKey(_ + _)
    val lc = labResult.map(index => ((index.patientID, index.testName), 1.0)).reduceByKey(_ + _)
    val lf = ls.join(lc).map(index => (index._1, index._2._1 / index._2._2))
    val lt = lf.filter(index => (candidateLab.contains(index._1._2)))
    lt
  }

  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   *
   * @param sc      SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    val fmap = feature.map(_._1._2).distinct().collect.zipWithIndex.toMap
    val fmap2 = sc.broadcast(fmap)
    val finalValues = feature.map(index => (index._1._1, (index._1._2, index._2))).groupByKey()
    val result = finalValues.map {
      case (key, value) =>
        val countofFinalValues = fmap2.value.size
        val finalValuesIndecies = value.toList.map { case (key2, value2) => (fmap2.value(key2), value2) }
        val finalValuesVectors = Vectors.sparse(countofFinalValues, finalValuesIndecies)
        val soln = (key, finalValuesVectors)
        soln
    }
    result
  }
}

