from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as charts
import numpy as numpy
import itertools as tools

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	graph, labels = charts.subplots(1, 2, figsize = (20, 10))
	labels[0].set_title('Loss Curves')
	labels[0].plot(train_losses, 'C0', label ='Training Loss')
	labels[0].plot(valid_losses, 'C1', label ='Validation Loss')
	labels[0].legend(loc ="upper right")
	labels[0].set_xlabel("Epoch")
	labels[0].set_ylabel("Loss")
	labels[1].set_title('Accuracy Curves')
	labels[1].plot(train_accuracies, 'C0', label ='Training Accuracy')
	labels[1].plot(valid_accuracies, 'C1', label ='Validation Accuracy')
	labels[1].legend(loc ="upper left")
	labels[1].set_xlabel("Epoch")
	labels[1].set_ylabel("Accuracy")
	graph.savefig('Learning_Curve.png')

def plot_confusion_matrix(results, class_names):
	def plot_matrix(cm, labels, chart_name ='conf_matrix_internal_function'):
		cm = cm.astype('float')/cm.sum(axis = 1)[:, numpy.newaxis]
		charts.imshow(cm, interpolation='nearest', cmap = charts.cm.Blues)
		charts.title(chart_name)
		charts.colorbar()
		checkered_boxes = numpy.arange(len(labels))
		charts.xticks(checkered_boxes, labels, rotation = 45)
		charts.yticks(checkered_boxes, labels)
		formatType = '.2f'
		maxLimit = cm.max() / 2.
		for x, y in tools.product(range(cm.shape[0]), range(cm.shape[1])):
			charts.text(y, x, format(cm[x, y], formatType), horizontalalignment = "center", color = "white" if cm[x, y] > maxLimit else "black")
		charts.ylabel('True')
		charts.xlabel('Predicted')
		charts.tight_layout()
	true_y_label, pred_y_label = zip(* results)
	#print(true_y_label)
	#print("_______________________________________________________________________")
	#print(pred_y_label)
	internalMatrixFunction = confusion_matrix(true_y_label, pred_y_label)
	numpy.set_printoptions(precision = 2)
	charts.figure()
	plot_matrix(internalMatrixFunction, labels = class_names, chart_name ='Normalized Confusion Matrix')
	charts.savefig("Confusion_Matrix.png")