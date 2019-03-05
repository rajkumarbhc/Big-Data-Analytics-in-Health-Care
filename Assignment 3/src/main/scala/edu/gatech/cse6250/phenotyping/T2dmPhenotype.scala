package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43", "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")
  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6", "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")
  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")
  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl", "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl", "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose", "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide", "avandia", "actos", "actos", "glipizide")
  val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] =

    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * You don't need to follow the example placeholder code below exactly, but do have the same return type.
     *
     * Hint: Consider case sensitivity when doing string comparisons.
     */

    {
      val sc = medication.sparkContext
      /** Hard code the criteria */
      //val type1_dm_dx = Set("code1", "250.03")
      //val type1_dm_med = Set("med1", "insulin nph")
      // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well

      val patientIDs = diagnostic.map(_.patientID).union(labResult.map(_.patientID)).union(medication.map(_.patientID)).distinct()

      //////Case//////
      val type1DiagnosisGroup = diagnostic.filter(index => T1DM_DX.contains(index.code)).map(_.patientID).distinct()
      val type1DiagnosisGroupExcultion = patientIDs.subtract(type1DiagnosisGroup).distinct()
      val type2DiagnosisGroup = diagnostic.filter(index => T2DM_DX.contains(index.code)).map(_.patientID).distinct()
      val type2DiagnosisGroupExcultion = patientIDs.subtract(type2DiagnosisGroup).distinct()
      val type1MedicationGroup = medication.filter(index => T1DM_MED.contains(index.medicine.toLowerCase)).map(_.patientID).distinct()
      val type1MedicationGroupExclution = patientIDs.subtract(type1MedicationGroup).distinct()
      val type2MedicationGroup = medication.filter(index => T2DM_MED.contains(index.medicine.toLowerCase)).map(_.patientID).distinct()
      val type2MedicationGroupExclution = patientIDs.subtract(type2MedicationGroup).distinct()
      val cp_1 = type1DiagnosisGroupExcultion.intersection(type2DiagnosisGroup).intersection(type1MedicationGroupExclution)
      val cp_2 = type1DiagnosisGroupExcultion.intersection(type2DiagnosisGroup).intersection(type1MedicationGroup).intersection(type2MedicationGroupExclution)
      val cp_3_1 = type1DiagnosisGroupExcultion.intersection(type2DiagnosisGroup).intersection(type1MedicationGroup).intersection(type2MedicationGroup)
      val cm_3 = medication.map(index => (index.patientID, index)).join(cp_3_1.map(index => (index, 0))).map(index => Medication(index._2._1.patientID, index._2._1.date, index._2._1.medicine))
      val cm_1 = cm_3.filter(index => T1DM_MED.contains(index.medicine.toLowerCase)).map(index => (index.patientID, index.date.getTime())).reduceByKey(Math.min)
      val cm_2 = cm_3.filter(index => T2DM_MED.contains(index.medicine.toLowerCase)).map(index => (index.patientID, index.date.getTime())).reduceByKey(Math.min)
      val cp_3 = cm_1.join(cm_2).filter(index => index._2._1 > index._2._2).map(_._1)
      val casePatients_v1 = sc.union(cp_1, cp_2, cp_3).distinct()

      //////Control//////
      val fg1 = labResult.filter(index => index.testName.toLowerCase.contains("glucose"))
      val glucose = fg1.map(_.patientID).distinct()
      val gs = glucose.collect.toSet
      val glucosePositivePatients = labResult.filter(index => gs(index.patientID))
      val labValuesAbnormal = labResultsAbnormal(glucosePositivePatients).distinct()
      val gps2 = glucose.subtract(labValuesAbnormal).distinct()
      val gps3_1 = diagnostic.filter(index => DM_RELATED_DX.contains(index.code)).map(index => index.patientID).distinct()
      val gps3_2 = diagnostic.filter(index => index.code.startsWith("250.")).map(index => index.patientID).distinct()
      val gps3 = patientIDs.subtract(gps3_1.union(gps3_2)).distinct()
      val controlPatients_v1 = gps2.intersection(gps3)

      //////Other//////
      val patientsofOtherCategory = patientIDs.subtract(casePatients_v1).subtract(controlPatients_v1).distinct()

      /** Find CASE Patients */
      val casePatients = casePatients_v1.map((_, 1))

      /** Find CONTROL Patients */
      val controlPatients = controlPatients_v1.map((_, 2))

      /** Find OTHER Patients */
      val others = patientsofOtherCategory.map((_, 3))

      /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
      val phenotypeLabel = sc.union(casePatients, controlPatients, others)

      /** Return */
      phenotypeLabel

    }

  def labResultsAbnormal(abnormalLab: RDD[LabResult]): RDD[String] = {
    val sc = abnormalLab.sparkContext
    val fhx1 = abnormalLab.filter(index => index.testName.equals("hba1c") && index.value >= 6.0).map(index => index.patientID)
    val fhx2 = abnormalLab.filter(index => index.testName.equals("hemoglobin a1c") && index.value >= 6.0).map(index => index.patientID)
    val fhx3 = abnormalLab.filter(index => index.testName.equals("fasting glucose") && index.value >= 110).map(index => index.patientID)
    val fhx4 = abnormalLab.filter(index => index.testName.equals("fasting blood glucose") && index.value >= 110).map(index => index.patientID)
    val fhx5 = abnormalLab.filter(index => index.testName.equals("fasting plasma glucose") && index.value >= 110).map(index => index.patientID)
    val fhx6 = abnormalLab.filter(index => index.testName.equals("glucose") && index.value > 110).map(index => index.patientID)
    val fhx7 = abnormalLab.filter(index => index.testName.equals("glucose, serum") && index.value > 110).map(index => index.patientID)
    val fhx = sc.union(fhx1, fhx2, fhx3, fhx4, fhx5, fhx6, fhx7)
    fhx
  }

}

