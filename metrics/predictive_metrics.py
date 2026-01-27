# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tf_slim.layers import layers as _layers
tf.compat.v1.disable_eager_execution()
def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len

def predictive_score_metrics (ori_data, generated_data):
  """Report the performance of Post-hoc RNN one-step ahead prediction.
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
  Returns:
    - predictive_score: MAE of the predictions on the original data
  """
  ori_data = np.array(ori_data)
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()
  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
  ## Builde a post-hoc RNN predictive network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128
  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim], name = "myinput_x")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")    
  Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim], name = "myinput_y")
  # Predictor function
  def predictor (x, t):
    """Simple predictor function.
    Args:
      - x: time-series data
      - t: time information
    Returns:
      - y_hat: prediction
      - p_vars: predictor variables
    """
    with tf.compat.v1.variable_scope("predictor", reuse=tf.compat.v1.AUTO_REUSE) as vs:
      p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
      p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = _layers.fully_connected(p_outputs, dim, activation_fn=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]
    return y_hat, p_vars
  y_pred, p_vars = predictor(X, T)
  # Loss for the predictor
  p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
  # optimizer
  p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
  ## Training    
  # Session start
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
  # Training using Synthetic dataset
  for itt in range(iterations):
    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]     
    # X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    

    X_mb = list(generated_data[i][:-1] for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:],[len(generated_data[i][1:]),dim]) for i in train_idx)

    T_mb = list(generated_time[i]-1 for i in train_idx)
    # Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
    # Train predictor
    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        
  ## Test the trained model on the original data
  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]
  X_mb = list(ori_data[i][:-1,:(dim)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:], [len(ori_data[i][1:]),dim]) for i in train_idx)
  # Prediction
  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
  # Compute the performance in terms of MAE
  MAE_temp = 0
  MSE_temp = 0
  R2_temp = 0 
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
  predictive_score = MAE_temp / no
  return predictive_score    


def predictive_score_metrics2 (ori_data, generated_data):
  """Report the performance of Post-hoc RNN one-step ahead prediction.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - predictive_score: MAE of the predictions on the original data
  """

  ori_data = np.array(ori_data)
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
  ## Builde a post-hoc RNN predictive network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128
    
  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")    
  Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")
    
  # Predictor function
  def predictor (x, t):
    """Simple predictor function.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat: prediction
      - p_vars: predictor variables
    """
    with tf.compat.v1.variable_scope("predictor", reuse=tf.compat.v1.AUTO_REUSE) as vs:
      p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
      p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = _layers.fully_connected(p_outputs, 1, activation_fn=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]
    
    return y_hat, p_vars
    
  y_pred, p_vars = predictor(X, T)
  # Loss for the predictor
  p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
  # optimizer
  p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
        
  ## Training    
  # Session start
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # Training using Synthetic dataset
  for itt in range(iterations):
          
    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]     
            
    X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(generated_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
          
    # Train predictor
    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        
    
  ## Test the trained model on the original data
  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]
    
  X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)
    
  # Prediction
  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
    
  # Compute the performance in terms of MAE
  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
  predictive_score = MAE_temp / no
    
  return predictive_score
    