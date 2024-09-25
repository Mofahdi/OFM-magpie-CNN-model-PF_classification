# OFM-magpie-CNN-model-PF_class
Classification by CNN model that utilizes orbital field matrix (OFM) and magpie features. It was used to classify thermoelectric power factor.

## Usage
you have to have "props.csv" file that have 'id' and 'prop' columns. You also have to put the POSCAR files inside "directory" and each file must end with ".POSCAR"</br>
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
