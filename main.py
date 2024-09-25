import os 
import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from data_process import get_df, train_val_test_split
#from utils_model import my_model, predictions_metrics, train_val_loss_plot
from utils_model import my_model, predictions_metrics, train_val_loss_plot

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

# parse args
parser = argparse.ArgumentParser(description='ofm-magpie model inputs')
parser.add_argument('--ofm_channels', default=[32, 32, 64], nargs='+', type=int)
parser.add_argument('--ofm_kernels', default=[5, 3, 3], nargs='+', type=int)
parser.add_argument('--magpie_channels', default=[32, 48, 64], nargs='+', type=int)
parser.add_argument('--magpie_kernels', default=[3, 3, 3], nargs='+', type=int)

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--output_dir', default='results', type=str, help='output directory')

parser.add_argument('--test_ratio', default=0.1, type=float,)
parser.add_argument('--num_kfolds', default=9, type=int,)

args = parser.parse_args(sys.argv[1:])

# dataset
data_path=os.path.join(os.getcwd(), 'data')
df=get_df(csv_path=os.getcwd(), data_path=data_path)
train_ratio=1 - args.test_ratio 
train_valid_df, test_df=train_val_test_split(df, train_ratio=train_ratio, test_ratio=args.test_ratio)

test_ids=test_df['id'].values
test_data=(np.asarray(test_df.ofd.values.tolist()), np.asarray(test_df.magpie_matrix.values.tolist()))
test_labels=test_df['prop'].values

ids=train_valid_df['id'].values
labels=train_valid_df['prop'].values


def plot_roc_curve(fpr, tpr, name, dpi=400):
	fig, ax = plt.subplots(figsize=(6,5))
	ax.plot(fpr, tpr, color='orange', label='ROC')
	ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
		#ax.set_title('Receiver Operating Characteristic (ROC) Curve')
	ax.legend()
	fig.tight_layout()
	fig.savefig(name, dpi=dpi)



def create_model():
	# model 
	model=my_model(
			ofm_channels=args.ofm_channels, 
			ofm_kernels=args.ofm_kernels, 
			magpie_channels=args.magpie_channels, 
			magpie_kernels=args.magpie_kernels,
			)

	optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=args.lr)

	model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
			optimizer=optimizer,
			metrics=[
				tf.keras.metrics.BinaryAccuracy(name='accuracy'),
				tf.keras.metrics.Precision(name='precision'),
				tf.keras.metrics.Recall(name='recall'),
				]
			)
	return model


#output directory
out_dir=os.path.join(os.getcwd(), args.output_dir)
if not os.path.isdir(out_dir):
	os.mkdir(out_dir)


# Cross validation (CV)
num_folds=args.num_kfolds
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1

iterations=[]; 
min_train_loss=[]; min_val_loss=[]; 
train_acc=[]; val_acc=[]; test_acc=[];
train_prec=[]; val_prec=[]; test_prec=[];
train_recl=[]; val_recl=[]; test_recl=[];
train_f1=[]; val_f1=[]; test_f1=[];
test_roc_auc=[]
for train, valid in kfold.split(train_valid_df.prop.values):
	model=create_model()

	# train and valid data
	train_data=(np.asarray(train_valid_df.ofd.values.tolist())[train], np.asarray(train_valid_df.magpie_matrix.values.tolist())[train])
	valid_data=(np.asarray(train_valid_df.ofd.values.tolist())[valid], np.asarray(train_valid_df.magpie_matrix.values.tolist())[valid])

	# model training and obtaining the best model
	filename="weights.best_"+str(fold_no)+".hdf5"
	filepath=os.path.join(out_dir, filename)
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min', save_weights_only=True)
	callbacks_list = [checkpoint]
	# validation_data=(X_test, Y_test)
	history = model.fit(
				train_data, 
				labels[train], 
				batch_size=args.batch_size,
				callbacks=callbacks_list, 
				validation_data=(valid_data, labels[valid]),  
				validation_batch_size=32, 
				epochs=args.epochs,
				)
	model.load_weights(filepath)

	
	# train val loss curve for each kfold
	#train_val_loss_plot(history, os.path.join(out_dir, 'train_val_loss_'+str(fold_no)+'.pdf'))

	name=os.path.join(out_dir, 'train_val_loss_'+str(fold_no)+'.jpg')
	dpi=400
	fig, ax = plt.subplots(figsize=(6,5))
	ax.plot(history.history['loss'], label='train loss')
	ax.plot(history.history['val_loss'], label='val loss')
	ax.set_title("model loss")
	ax.set_xlabel('epoch')
	ax.set_ylabel('loss')
	#ax.legend(['train', 'valid'], loc='upper right')
	ax.legend(loc='upper right')
	fig.tight_layout()
	fig.savefig(name, dpi=dpi)


	predictions=model.predict(test_data).squeeze()
	print(predictions, test_labels)
	# Computing manually fpr, tpr, thresholds and roc auc 
	fpr, tpr, thresholds = roc_curve(test_labels, predictions)
	roc_auc = auc(fpr, tpr)
	print("ROC_AUC Score : ",roc_auc)
	print("Function for ROC_AUC Score : ",roc_auc_score(test_labels, predictions)) # Function present
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	print("Threshold value is:", optimal_threshold)
	plot_roc_curve(fpr, tpr, name=os.path.join(out_dir, 'ROC_AUC_'+str(fold_no)+'.jpg'))


	#predictions_metrics(model, ids[valid], valid_data, labels[valid], 
	#os.path.join(out_dir, 'valid_results_'+str(fold_no)+'.csv'), 
	#os.path.join(out_dir, 'valid_metrics_'+str(fold_no)+'.txt'))

	# def predictions_metrics(model, ids, data, labels, results_csv_filename, metrics_filename):
	def predictions_metrics_cm_cls(model, ids, data, labels, results_csv_filename, metrics_filename, cm_filename, dpi=400, threshold=0.5):	
		optimal_threshold=threshold
		predictions=model.predict(data).squeeze()
		prediction_classes = [
			1 if prob > optimal_threshold else 0 for prob in np.ravel(predictions)
			]
		evaluated_predictions=model.evaluate(data, labels)
		preds=pd.DataFrame({'id': ids, 'target':labels, 'predicted':prediction_classes})
		preds.to_csv(results_csv_filename, index=False)
		
		with open(metrics_filename, 'w') as out:
			#R2_test=r2_score(labels, new_predictions)
			#mae_test=mean_absolute_error(labels, new_predictions)
			#mse_test=mean_squared_error(labels, new_predictions)
			out.write('loss:'+str(evaluated_predictions[0])+'\n')
			out.write('accuracy_eval: '+str(evaluated_predictions[1])+'\n')
			out.write('precision_eval: '+str(evaluated_predictions[2])+'\n')
			out.write('recall_eval: '+str(evaluated_predictions[3])+'\n')

			out.write('accuracy: '+str(accuracy_score(labels, prediction_classes))+'\n')
			out.write('precision: '+str(precision_score(labels, prediction_classes))+'\n')
			out.write('recall: '+str(recall_score(labels, prediction_classes))+'\n')
			out.write('f1: '+str(f1_score(labels, prediction_classes))+'\n')

		# Plot the confusion matrix using a heatmap
		confusion_matrix = tf.math.confusion_matrix(labels, prediction_classes)
		#print(confusion_matrix)

		fig, ax = plt.subplots(figsize=(6,5))
		sns.heatmap(confusion_matrix, annot=True, cmap='viridis')
		ax.set_xlabel('Predicted Labels')
		ax.set_ylabel('True Labels')
		#plt.xlabel('Predicted Labels')
		#plt.ylabel('True Labels')
		#ax.legend()
		fig.tight_layout()
		fig.savefig(cm_filename, dpi=dpi)

		return evaluated_predictions

	# output predictions, metrics, and confusion matrix for each kfold 
	train_eval=predictions_metrics_cm_cls(model, ids[train], train_data, labels[train], 
	os.path.join(out_dir, 'train_results_'+str(fold_no)+'.csv'), 
	os.path.join(out_dir, 'train_metrics_'+str(fold_no)+'.txt'),
	os.path.join(out_dir, 'train_cm_'+str(fold_no)+'.jpg'))

	val_eval=predictions_metrics_cm_cls(model, ids[valid], valid_data, labels[valid], 
	os.path.join(out_dir, 'valid_results_'+str(fold_no)+'.csv'), 
	os.path.join(out_dir, 'valid_metrics_'+str(fold_no)+'.txt'),
	os.path.join(out_dir, 'valid_cm_'+str(fold_no)+'.jpg'))

	test_eval=predictions_metrics_cm_cls(model, test_ids, test_data, test_labels, 
	os.path.join(out_dir, 'test_results_'+str(fold_no)+'.csv'), 
	os.path.join(out_dir, 'test_metrics_'+str(fold_no)+'.txt'),
	os.path.join(out_dir, 'test_cm_'+str(fold_no)+'.jpg'))


	# metrics for each kfold 
	iterations.append(fold_no)
	min_train_loss.append(min(history.history['loss']));
	min_val_loss.append(min(history.history['val_loss']));

	train_acc.append(train_eval[1]); 
	train_prec.append(train_eval[2])
	train_recl.append(train_eval[3])

	val_acc.append(val_eval[1]); 
	val_prec.append(val_eval[2])
	val_recl.append(val_eval[3])

	test_acc.append(test_eval[1]); 
	test_prec.append(test_eval[2])
	test_recl.append(test_eval[3])

	test_roc_auc.append(roc_auc_score(test_labels, predictions))

	
	fold_no += 1
	
	#break


# summary of all major results
summary_df=pd.DataFrame({'iterations': iterations, 'min_train_loss':min_train_loss, 'min_val_loss':min_val_loss, 
			'train_accuracy':train_acc, 'val_accuracy':val_acc, 'test_accuracy':test_acc,
			'train_precison':train_prec, 'val_precison':val_prec, 'test_precison':test_prec,
			'train_recall':train_recl, 'val_recall':val_recl, 'test_recall':test_recl,
			'test_roc_auc':test_roc_auc,
			})
summary_df.to_csv(os.path.join(out_dir, 'CV_results_summary.csv'), index=False)
