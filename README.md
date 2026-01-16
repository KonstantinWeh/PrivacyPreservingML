# PrivacyPreservingML

Repository for the **2025 Integrated Masters Project** by *Cedric Bohni* and *Konstantin Wehmeyer*.

This project provides Jupyter notebooks and source files for testing different **privacy-preserving machine learning** techniques, along with a demo application that showcases **secure inference using functional encryption**. The encryption approach implements **Inner-Product Functional Encryption (IPFE)**.

### Setup

Before running the notebooks, tests, or demo, install the Python dependencies:

```bash
pip install -r requirements.txt
```

### Notebooks

The Jupyter notebooks walk through all experimental steps of the project, each building upon the previous one. It is recommended to open and execute them **in order**:

1. **`00_basic_cnn.ipynb`** – Trains and evaluates a simple CNN on MNIST, visualizing filters and activations.  
2. **`01_modified_cnn.ipynb`** – Explores CNN variations (architecture, preprocessing) and trains weights for IPFE.  
   - This notebook saves the trained model in the `models/` folder:
     ```
     cnn_model_1.pth
     ```
3. **`02_basic_ipfe_cnn.ipynb`** – Introduces IPFE applied to the first CNN layer.  
4. **`03_extracted_encrypt_ipfe_cnn.ipynb`** – Extracts and encrypts image patches separately, using an optimized prime.  
5. **`04_0_kernel_ipfe_cnn.ipynb`** – Applies IPFE at the kernel level.  
6. **`04_1_kernel_patch_ipfe_cnn.ipynb`** – Tests the kernel-patch approach and compares results.  
7. **`04_2_batch_ipfe_cnn.ipynb`** – Implements batch-based performance measurements and parallel decryption with compiler optimizations.  
8. **`04_3_batch_kernel_ipfe_cnn.ipynb`** – Extends the previous batching approach to kernel-level IPFE.

**Notes and tips**
- Some notebooks require pretrained weights from the `models/` directory. If missing, the notebook may retrain or raise an error; review the top cells for a `load model` option.  
- To only visualize filters and activations, you can skip training and load a provided model.  
- Long-running cells (e.g., training or large-batch experiments) show progress bars — interrupt the kernel to stop execution.

**Reproducing results**
- Run each notebook from top to bottom to reproduce figures and results.  
- For long runtimes, checkpoints or flags are provided to load precomputed outputs from the `results/` folder.

---

### Test Suite

The file `src/main.py` provides a full test suite for evaluating various **functional encryption schemes and CNN architectures**, requiring minimal manual interaction.

**Quick start**
1. Edit the configuration file (`configs/base.yaml`) with your desired parameters.  
2. Run the test suite:
   ```bash
   python -m src.main -r 1
    ```
   Use the -r flag to repeat the run n times.

### Workflow

1. **Edit the configuration file**  
   Configure parameters in `configs/base.yaml`:
   - Random seed for reproducibility  
   - Prime value for IPFE and vector size (e.g., 4 for a 2×2 kernel)  
   - Batch size and dataset location  
   - Number of workers for parallelization  
   - `max_test_samples` to limit test images (empty = all)  
   - `name` to toggle between `"ipfe"` (encrypted) and `"plain"` (unencrypted)`  
     - Encrypted runs require a prior unencrypted run with identical configs.  

   **Architecture parameters**
   - `in_channels`: Input channels (1 for grayscale, 3 for RGB)  
   - `num_classes`: For MNIST, use 10  
   - Convolutional layer parameters (each list must have the same length):  
     - `c`: Kernel sizes  
     - `k`: Kernel strides  
     - `p`: Pooling options (0 = none, 1 = enabled)  
     - `stride`: Strides per layer  
     - `padding`: Padding per layer  
   - `dropout`: Dropout rate in the fully connected layer  
   - **Optimization options:**  
     - `precrypt`: Usually set to `true`  
     - Only one optimization option should be `true` at a time (set all to `false` for a baseline run).  
   - Additional fields define training parameters and model save paths.

2. **Train the base model**  
   Run a `"plain"` configuration with `max_test_samples: None`.  
   This will start model training and display batch, loss, and accuracy per epoch, then automatically save the model:

        k2_conv8_16_32_64_stride3_pad1_dropout0.5.pt

3. **Run a baseline test**  
   After training, run the model with a chosen number of test images.  
   Example terminal output:
       
        Run ========================= 1
        Maximum absolute inner product over all slices: 313.33173
        Scaled maximum absolute inner product (x10000): 3.1333172e+06
        Suggested prime minimum (scaled max * 2): 6.2666345e+06
        Evaluating on test set...

    From the suggested minimum, find (manually or with a tool) the next greater prime.  
    Use this prime value in the IPFE configuration.

4. **Run the encrypted version**  
Execute the same configuration with `name: ipfe`.  
All results will be saved in the `results/` folder.

   
### Encrypted Convolution Inference Demo (Server–Client)

This demo illustrates **privacy-preserving CNN inference** using **Inner-Product Functional Encryption (IPFE)**.  
A client encrypts image data, sends it to the server, and receives predictions — all without exposing plaintext data.

#### Overview

- **`server.py`**
  - Runs a TCP server that loads a pretrained CNN on MNIST.
  - Handles model selection, IPFE setup, and encrypted inference.
  - Decrypts IPFE ciphertexts only enough to compute first-layer inner products before completing the remaining layers in plaintext.

- **`client.py`**
  - Connects to the server and fetches first-layer CNN weights.
  - Initializes an IPFE instance using the selected model configuration.
  - Encrypts unfolded MNIST patches and sends them for encrypted inference.

#### Demo Workflow
1. **Start the server**
   ```bash
   python Demo.server
    ```  
The server listens on 127.0.0.1:5000 and loads models:

    models/demo_model_{1,2,3}.pth

2. **Run the client**
    ```bash
    python Demo.client
    ```  
- Interactive prompt options
  - Command 0 – Close Demo
  - Command 1 – Choose a CNN variant and initialize IPFE parameters (prime, generator, vector length, and keys). 
  - Command 2 – Select the number of MNIST test images to encrypt (the client unfolds and encrypts each patch). 
  - Command 3 – Send encrypted patches to the server and receive model predictions.
