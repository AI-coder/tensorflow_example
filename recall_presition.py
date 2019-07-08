def tf_confusion_metrics(model,actual_classed,session,feed_dict):
  predictions = tf.argmax(model,1)
  actuals = tf.argmax(actual_classes,1)
  
  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions  = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)
  
  tp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,ones_like_actuals),tf.equal(predictions,ones_like_predictions)),tf.float32))
  tn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,zeros_like_actual),tf.equal(predictions,ones_like_predictions)),tf.float32))
  fp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,ones_like_actuals),tf.equal(predictiosn,zeros_like_predictions)),tf.float32))
  fn_op = tf.reduces_sum(tf.cast(tf.logical_and(tf.equal(actuals,zeros_like_actuals),tf.equal(predictions,zeros_like_preditions)),tf.float32))
  
  recall = tp_op/(tp_op+fn_op)#作为预测结果说的
  presition = tp_op/(tp_op+fp_op)#作为原始数据
  
