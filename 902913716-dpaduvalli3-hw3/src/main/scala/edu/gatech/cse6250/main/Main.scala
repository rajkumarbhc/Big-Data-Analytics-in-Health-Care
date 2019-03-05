package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.clustering.Metrics
import edu.gatech.cse6250.features.FeatureConstruction
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import edu.gatech.cse6250.phenotyping.T2dmPhenotype
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{ GaussianMixture, KMeans, StreamingKMeans }
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ DenseMatrix, Matrices, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Yu Jing <yjing43@gatech.edu>,
 * @author Ming Liu <mliu302@gatech.edu>
 */
object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(spark)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    // =========== USED FOR AUTO GRADING CLUSTERING GRADING =============
    // phenotypeLabel.map{ case(a,b) => s"$a\t$b" }.saveAsTextFile("data/phenotypeLabel")
    // featureTuples.map{ case((a,b),c) => s"$a\t$b\t$c" }.saveAsTextFile("data/featureTuples")
    // return
    // ==================================================================

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamingPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKmeans is: $streamingPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamingPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKmeans is: $streamingPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures: RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    println("phenotypeLabel: " + phenotypeLabel.count)
    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray))) })
    println("features: " + features.count)
    val rawFeatureVectors = features.map(_._2).cache()
    println("rawFeatureVectors: " + rawFeatureVectors.count)

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]

    def transform(feature: Vector): Vector = {
      val scaled = scaler.transform(Vectors.dense(feature.toArray))
      Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
    }

    /**
     * TODO: K Means Clustering using spark mllib
     * Train a k means model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */

    val knnCluster = new KMeans().setSeed(6250L).setK(3).setMaxIterations(20).run(featureVectors).predict(featureVectors)
    val knnClusterPrediction = features.map(_._1).zip(knnCluster).join(phenotypeLabel).map(_._2)

    //val knnClusterOne = knnClusterPrediction.filter(l => (l._1 == 0 && l._2 == 3)).count()
    //val knnClusterTwo = knnClusterPrediction.filter(l => (l._1 == 1 && l._2 == 3)).count()
    //val knnClusterThree = knnClusterPrediction.filter(l => (l._1 == 2 && l._2 == 3)).count()
    //println(knnClusterOne)
    //println(knnClusterTwo)
    //println(knnClusterThree)

    val kMeansPurity = Metrics.purity(knnClusterPrediction)

    /**
     * TODO: GMMM Clustering using spark mllib
     * Train a Gaussian Mixture model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    val clusterGaussian = new GaussianMixture().setSeed(6250L).setK(3).setMaxIterations(20).run(featureVectors).predict(featureVectors)
    val clusterGaussianPrediction = features.map(_._1).zip(clusterGaussian).join(phenotypeLabel).map(_._2)

    //val GausClusterOne = clusterGaussianPrediction.filter(l => (l._1 == 0 && l._2 == 1)).count()
    //val GausClusterTwo = clusterGaussianPrediction.filter(l => (l._1 == 1 && l._2 == 1)).count()
    //val GausClusterThree = clusterGaussianPrediction.filter(l => (l._1 == 2 && l._2 == 1)).count()
    //println(GausClusterOne)
    //println(GausClusterTwo)
    //println(GausClusterThree)
    val gaussianMixturePurity = Metrics.purity(clusterGaussianPrediction)

    /**
     * TODO: StreamingKMeans Clustering using spark mllib
     * Train a StreamingKMeans model using the variabe featureVectors as input
     * Set the number of cluster K = 3, DecayFactor = 1.0, number of dimensions = 10, weight for each center = 0.5, seed as 6250L
     * In order to feed RDD[Vector] please use latestModel, see more info: https://spark.apache.org/docs/2.2.0/api/scala/index.html#org.apache.spark.mllib.clustering.StreamingKMeans
     * To run your model, set time unit as 'points'
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    val knnStream = new StreamingKMeans().setK(3).setDecayFactor(1.0).setRandomCenters(10, 0.5, 6250L).latestModel()
    val knnStreamPrediction = knnStream.update(featureVectors, 1.0, "points").predict(featureVectors)
    val knnStreamPredictionGrouping = features.map(_._1).zip(knnStreamPrediction).join(phenotypeLabel).map(_._2)
    //val ksClusterOne = knnStreamPredictionGrouping.filter(l => (l._1 == 0 && l._2 == 3)).count()
    //val ksClusterTwo = knnStreamPredictionGrouping.filter(l => (l._1 == 1 && l._2 == 3)).count()
    //val ksClusterThree = knnStreamPredictionGrouping.filter(l => (l._1 == 2 && l._2 == 3)).count()
    //println(ksClusterOne)
    //println(ksClusterTwo)
    //println(ksClusterThree)

    val streamKmeansPurity = Metrics.purity(knnStreamPredictionGrouping)

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity)

  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
   *
   * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
    import spark.implicits._
    val sqlContext = spark.sqlContext

    /* a helper function sqlDateParser may useful here */

    /**
     * load data using Spark SQL into three RDDs and return them
     * Hint:
     * You can utilize edu.gatech.cse6250.helper.CSVHelper
     * through your sparkSession.
     *
     * This guide may helps: https://bit.ly/2xnrVnA
     *
     * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
     * Be careful when you deal with String and numbers in String type.
     * Ignore lab results with missing (empty or NaN) values when these are read in.
     * For dates, use Date_Resulted for labResults and Order_Date for medication.
     *
     */

    /**
     * TODO: implement your own code here and remove
     * existing placeholder code below
     */

    //medication
    val medication_orders_INPUT = CSVHelper.loadCSVAsTable(spark, "data/medication_orders_INPUT.csv", "medication_orders_INPUT")
    val medicationDataFrame = spark.sql("SELECT Member_ID AS patientID, Order_Date, Drug_Name AS medicine FROM medication_orders_INPUT")
    val medication: RDD[Medication] = medicationDataFrame.map(line => Medication(line(0).toString, sqlDateParser(line(1).toString), line(2).toString.toLowerCase)).rdd

    //Lab Results
    val lab_results_INPUT = CSVHelper.loadCSVAsTable(spark, "data/lab_results_INPUT.csv", "lab_results_INPUT")
    val labResultDataFrame = spark.sql("SELECT Member_ID AS patientID, Date_Resulted, Result_Name AS testName, Numeric_Result AS value FROM lab_results_INPUT WHERE Numeric_Result != ''")
    val labResult: RDD[LabResult] = labResultDataFrame.map(line => LabResult(line(0).asInstanceOf[String], sqlDateParser(line(1).toString), line(2).asInstanceOf[String].toLowerCase, line(3).asInstanceOf[String].filterNot(",".toSet).toDouble)).rdd

    //Diagnostics
    val encounter_INPUT = CSVHelper.loadCSVAsTable(spark, "data/encounter_INPUT.csv", "encounter_INPUT")
    val encounter_dx_INPUT = CSVHelper.loadCSVAsTable(spark, "data/encounter_dx_INPUT.csv", "encounter_dx_INPUT")
    val diagnosisDataFrame = spark.sql("SELECT encounter_INPUT.Member_ID AS patientID, encounter_INPUT.Encounter_DateTime AS date, encounter_dx_INPUT.code FROM encounter_INPUT JOIN encounter_dx_INPUT ON encounter_INPUT.Encounter_ID = encounter_dx_INPUT.Encounter_ID")
    val diagnostic: RDD[Diagnostic] = diagnosisDataFrame.map(line => Diagnostic(line(0).toString, sqlDateParser(line(1).toString), line(2).toString)).rdd

    (medication, labResult, diagnostic)
  }

}
