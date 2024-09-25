# OFM-magpie-CNN-model-PF_class
Classification by CNN model that utilizes orbital field matrix (OFM) and magpie features. It was used to classify thermoelectric power factor.

## Usage
you have to have "props.csv" file that have 'id' and 'prop' columns. You also have to put the POSCAR files inside "directory" and each file must end with ".POSCAR"</br>
Note: the code has cross-validation.</br>
Here is a sample of how to run the code.
<code>
python main.py \
  --ofm_channels 32 32 64 \
  --ofm_kernels 5 3 3 \
  --magpie_channels 32 48 64 \
  --magpie_kernels 3 3 3 \
  -b 32 \
  --lr 0.001 \
  --epochs 50 \
  --output_dir "results" \
  --test_ratio 0.1 \
  --num_kfolds 9 \
</code>

the code will generate these files: </br>
1. **CV_results_summary.csv**: cross-validation results from all iterations</br>
2. **weights.best_#.hdf5**: weights of best model (lowest validation loss) where #: cross-validation iteration</br>
3. **ROC_AUC_#.jpg**: image of Receiver-Operating Characteristic curve with area under the curve value where #: cross-validation iteration</br>
4. **train_val_loss_#.jpg**: training and validaition losses where #: cross-validation iteration</br>

5. **test_cm_#.jpg**: </br>
6. **test_metrics_#.txt**: </br>
7. **test_results_#.csv**: </br>

8. **train_cm_#.jpg**: </br>
9. **train_metrics_#.txt**: </br>
10. **train_results_#.csv**: </br>

11. **valid_cm_#.jpg**: </br>
12. **valid_metrics_#.txt**: </br>
13. **valid_results_#.csv**: </br>

