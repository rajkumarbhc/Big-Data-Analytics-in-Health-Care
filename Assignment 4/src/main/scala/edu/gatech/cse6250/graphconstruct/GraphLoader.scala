/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult], medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {
    val patientSourceNode = patients.sparkContext
    val patientVertex: RDD[(VertexId, VertexProperty)] = patients.map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    val countOfPatients = patients.map(patientColumn => patientColumn.patientID).distinct().count()

    val diagnosticEvents = diagnostics.map(columnNames => ((columnNames.patientID, columnNames.icd9code), columnNames)).reduceByKey((columnName1, columnName2) => if (columnName1.date > columnName2.date) columnName1 else columnName2).map { case (key, x) => x }
    val diagnosticNode = diagnosticEvents.map(_.icd9code).distinct().zipWithIndex().map { case (sol, sol2) => (sol, sol2 + countOfPatients + 1) }
    val diagnosticVertixIdentifier = diagnosticNode.collect.toMap
    val diagnosticVertex: RDD[(VertexId, VertexProperty)] = diagnosticNode.map { case (icd9code, values) => (values, DiagnosticProperty(icd9code)) }
    val diagnosticNodeCount = diagnosticNode.count()

    val labEvents = labResults.map(columnNames => ((columnNames.patientID, columnNames.labName), columnNames)).reduceByKey((columnName1, columnName2) => if (columnName1.date > columnName2.date) columnName1 else columnName2).map { case (key, x) => x }
    val labNode = labEvents.map(_.labName).distinct().zipWithIndex().map { case (sol, sol2) => (sol, sol2 + countOfPatients + 1 + diagnosticNodeCount) }
    val labVertixIdentifier = labNode.collect.toMap
    val labVertex: RDD[(VertexId, VertexProperty)] = labNode.map { case (labName, values) => (values, LabResultProperty(labName)) }
    val labNodeCount = labNode.count()

    val medicationEvents = medications.map(columnNames => ((columnNames.patientID, columnNames.medicine), columnNames)).reduceByKey((columnName1, columnName2) => if (columnName1.date > columnName2.date) columnName1 else columnName2).map { case (key, x) => x }
    val medicationNode = medicationEvents.map(_.medicine).distinct().zipWithIndex().map { case (sol, sol2) => (sol, sol2 + countOfPatients + 1 + diagnosticNodeCount + labNodeCount) }
    val medicationVertixIdentifier = medicationNode.collect.toMap
    val medicationVertex: RDD[(VertexId, VertexProperty)] = medicationNode.map { case (medicine, values) => (values, MedicationProperty(medicine)) }

    //Define Graph Edges
    val graphLabVertixIdentifier = patientSourceNode.broadcast(labVertixIdentifier)
    val paientLabEdges = labEvents.map(columnNames => (columnNames.patientID, columnNames.labName, columnNames)).map { case (patientID, labName, index) => Edge(patientID.toLong, graphLabVertixIdentifier.value(labName), PatientLabEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val labPaientEdges = labEvents.map(columnNames => (columnNames.patientID, columnNames.labName, columnNames)).map { case (patientID, labName, index) => Edge(graphLabVertixIdentifier.value(labName), patientID.toLong, PatientLabEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val finalPaientLabEdges = patientSourceNode.union(paientLabEdges, labPaientEdges)

    val graphDiagnosticVertixIdentifier = patientSourceNode.broadcast(diagnosticVertixIdentifier)
    val paientDiagnosticEdges = diagnosticEvents.map(columnNames => (columnNames.patientID, columnNames.icd9code, columnNames)).map { case (patientID, icd9code, index) => Edge(patientID.toLong, graphDiagnosticVertixIdentifier.value(icd9code), PatientDiagnosticEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val diagnosticPaientEdges = diagnosticEvents.map(columnNames => (columnNames.patientID, columnNames.icd9code, columnNames)).map { case (patientID, icd9code, index) => Edge(graphDiagnosticVertixIdentifier.value(icd9code), patientID.toLong, PatientDiagnosticEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val finalPaientDiagnosticEdges = patientSourceNode.union(paientDiagnosticEdges, diagnosticPaientEdges)

    val graphMedicationVertixIdentifier = patientSourceNode.broadcast(medicationVertixIdentifier)
    val paientMedicationEdges = medicationEvents.map(columnNames => (columnNames.patientID, columnNames.medicine, columnNames)).map { case (patientID, medicine, index) => Edge(patientID.toLong, graphMedicationVertixIdentifier.value(medicine), PatientMedicationEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val medicationPaientEdges = medicationEvents.map(columnNames => (columnNames.patientID, columnNames.medicine, columnNames)).map { case (patientID, medicine, index) => Edge(graphMedicationVertixIdentifier.value(medicine), patientID.toLong, PatientMedicationEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val finalPaientMedicationEdges = patientSourceNode.union(paientMedicationEdges, medicationPaientEdges)

    // Making Graph
    val finalVertices = patientSourceNode.union(patientVertex, diagnosticVertex, labVertex, medicationVertex)
    val finalEdges = patientSourceNode.union(finalPaientDiagnosticEdges, finalPaientLabEdges, finalPaientMedicationEdges)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](finalVertices, finalEdges)
    graph
  }
}
